#!/usr/bin/env python3
"""
CNN-based Tactile-Vision RL Training
Uses CNN for both vision and tactile inputs, then embeds them properly
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import wandb
import imageio
from pathlib import Path
import tactile_envs
from pyvirtualdisplay import Display
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


class CNNVisionEncoder(nn.Module):
    """
    CNN-based vision encoder for RGB images
    Standard robotic vision pipeline: Conv layers -> Global Average Pool -> Linear
    """
    
    def __init__(self, input_shape=(3, 64, 64), embed_dim=256):
        super().__init__()
        
        # CNN backbone for spatial feature extraction
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Second conv block 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Global average pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear projection to embedding dimension
        self.projection = nn.Linear(256, embed_dim)
        
    def forward(self, x):
        # Extract CNN features
        features = self.cnn(x)  # (batch, 256, 4, 4)
        
        # Global average pooling
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # (batch, 256)
        
        # Project to embedding dimension
        embedding = self.projection(pooled)  # (batch, embed_dim)
        
        return embedding


class CNNTactileEncoder(nn.Module):
    """
    CNN-based tactile encoder for tactile sensor data
    Treats tactile as multi-channel 2D images (6 channels for force/pressure)
    """
    
    def __init__(self, input_shape=(6, 32, 32), embed_dim=256):
        super().__init__()
        
        # CNN backbone for tactile feature extraction
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear projection to embedding dimension
        self.projection = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        # Extract CNN features
        features = self.cnn(x)  # (batch, 128, 4, 4)
        
        # Global average pooling
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # (batch, 128)
        
        # Project to embedding dimension
        embedding = self.projection(pooled)  # (batch, embed_dim)
        
        return embedding


class MultimodalFusion(nn.Module):
    """
    Fuse vision and tactile embeddings
    Simple but effective: concatenate + MLP
    """
    
    def __init__(self, vision_dim=256, tactile_dim=256, output_dim=512):
        super().__init__()
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + tactile_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        
    def forward(self, vision_embed, tactile_embed):
        # Concatenate embeddings
        combined = torch.cat([vision_embed, tactile_embed], dim=-1)
        
        # Fuse with MLP
        fused = self.fusion(combined)
        
        return fused


class CNNBasedExtractor(BaseFeaturesExtractor):
    """
    CNN-based feature extractor for Stable-Baselines3
    Follows the modern robotic control pipeline:
    RGB -> CNN -> Embedding
    Tactile -> CNN -> Embedding  
    Fuse -> Policy Network
    """
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Check what inputs we have
        self.has_image = 'image' in observation_space.spaces
        self.has_tactile = 'tactile' in observation_space.spaces
        
        embed_dim = features_dim // 2  # Split embedding dimension
        
        # Vision encoder
        if self.has_image:
            image_shape = observation_space.spaces['image'].shape
            # Handle both (H, W, C) and (C, H, W) formats
            if image_shape[0] == 3:
                vision_shape = image_shape  # Already (C, H, W)
            else:
                vision_shape = (image_shape[2], image_shape[0], image_shape[1])  # Convert to (C, H, W)
            
            self.vision_encoder = CNNVisionEncoder(vision_shape, embed_dim)
        
        # Tactile encoder
        if self.has_tactile:
            tactile_shape = observation_space.spaces['tactile'].shape
            self.tactile_encoder = CNNTactileEncoder(tactile_shape, embed_dim)
        
        # Fusion network
        if self.has_image and self.has_tactile:
            self.fusion = MultimodalFusion(embed_dim, embed_dim, features_dim)
        elif self.has_image:
            # Vision only - simple projection
            self.projection = nn.Linear(embed_dim, features_dim)
        elif self.has_tactile:
            # Tactile only - simple projection
            self.projection = nn.Linear(embed_dim, features_dim)
        else:
            raise ValueError("No valid inputs found")
    
    def forward(self, observations: TensorDict) -> torch.Tensor:
        """
        Forward pass through CNN encoders and fusion
        """
        vision_embed = None
        tactile_embed = None
        
        # Process vision
        if self.has_image and 'image' in observations:
            vision_input = observations['image'].float()
            
            # Handle batch dimensions
            if vision_input.dim() == 3:
                vision_input = vision_input.unsqueeze(0)
            
            # Handle channel format (H, W, C) -> (C, H, W)
            if vision_input.shape[-1] == 3:
                vision_input = vision_input.permute(0, 3, 1, 2)
            
            vision_embed = self.vision_encoder(vision_input)
        
        # Process tactile
        if self.has_tactile and 'tactile' in observations:
            tactile_input = observations['tactile'].float()
            
            # Handle batch dimensions
            if tactile_input.dim() == 3:
                tactile_input = tactile_input.unsqueeze(0)
            
            tactile_embed = self.tactile_encoder(tactile_input)
        
        # Fuse embeddings
        if self.has_image and self.has_tactile and vision_embed is not None and tactile_embed is not None:
            # Multimodal fusion
            features = self.fusion(vision_embed, tactile_embed)
        elif vision_embed is not None:
            # Vision only
            features = self.projection(vision_embed)
        elif tactile_embed is not None:
            # Tactile only
            features = self.projection(tactile_embed)
        else:
            raise ValueError("No valid observations found")
        
        return features


class WandbCallback(BaseCallback):
    """Callback for logging to Weights & Biases"""
    
    def __init__(self, verbose=0, video_freq=5000):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.video_freq = video_freq
        self.video_dir = Path("./videos")
        self.video_dir.mkdir(exist_ok=True)
    
    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    wandb.log({
                        'episode/reward': info['episode']['r'],
                        'episode/length': info['episode']['l'],
                        'episode/time': info['episode']['t'],
                        'global_step': self.num_timesteps,
                    })
                
                if 'is_success' in info:
                    self.episode_successes.append(info['is_success'])
                    wandb.log({
                        'episode/success': info['is_success'],
                        'global_step': self.num_timesteps,
                    })
        
        if self.num_timesteps % 1000 == 0:
            if self.episode_rewards:
                wandb.log({
                    'train/mean_reward': np.mean(self.episode_rewards[-10:]),
                    'train/mean_length': np.mean(self.episode_lengths[-10:]),
                    'train/success_rate': np.mean(self.episode_successes[-10:]) if self.episode_successes else 0,
                    'global_step': self.num_timesteps,
                })
        
        # Record video occasionally
        if self.num_timesteps % self.video_freq == 0:
            self.record_video()
        
        return True
    
    def record_video(self):
        """Record a video of the current policy"""
        try:
            print(f"📹 Recording video at timestep {self.num_timesteps}")
            
            # Create a single environment for video recording
            env = self.training_env.envs[0]
            
            frames = []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            for step in range(100):  # Record 100 steps
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                # Get video frame from observation
                if isinstance(obs, dict) and 'image' in obs:
                    frame = obs['image']
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frames.append(frame)
                
                if done or truncated:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
            
            # Save video
            if frames:
                video_path = self.video_dir / f"policy_video_{self.num_timesteps:06d}.mp4"
                imageio.mimsave(str(video_path), frames, fps=20)
                
                # Log to wandb
                wandb.log({
                    "policy_video": wandb.Video(str(video_path), fps=20, format="mp4"),
                    "global_step": self.num_timesteps
                })
                
                print(f"📹 Video saved: {video_path}")
            else:
                print("⚠️  No frames captured for video")
                
        except Exception as e:
            print(f"Warning: Could not record video: {e}")


class CNNBasedTrainer:
    """
    CNN-based trainer for tactile-vision RL
    """
    
    def __init__(self, args):
        self.args = args
        self.display = None
        
        # Setup virtual display if headless
        if args.headless:
            self.setup_virtual_display()
        
        # Setup wandb if enabled
        if args.use_wandb:
            self.setup_wandb()
        
        # Setup environment
        self.setup_environment()
        
        # Setup algorithm
        self.setup_algorithm()
    
    def setup_virtual_display(self):
        """Setup virtual display for headless mode"""
        print("Setting up virtual display...")
        try:
            self.display = Display(visible=0, size=(1024, 768))
            self.display.start()
            print(f"Virtual display started on DISPLAY={os.environ.get('DISPLAY', 'None')}")
        except Exception as e:
            print(f"Warning: Could not start virtual display: {e}")
            print("Setting environment variables for headless rendering...")
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        print("Setting up Weights & Biases...")
        
        config = {
            'state_type': self.args.state_type,
            'im_size': self.args.im_size,
            'tactile_shape': self.args.tactile_shape,
            'no_gripping': self.args.no_gripping,
            'no_rotation': self.args.no_rotation,
            'max_delta': self.args.max_delta,
            'n_envs': self.args.n_envs,
            'algorithm': self.args.algorithm,
            'total_timesteps': self.args.total_timesteps,
            'learning_rate': self.args.learning_rate,
            'features_dim': self.args.features_dim,
            'architecture': 'CNN-based',
            'encoder_type': 'CNN -> Embedding',
        }
        
        if self.args.algorithm == 'PPO':
            config.update({
                'n_steps': self.args.n_steps,
                'batch_size': self.args.batch_size,
                'n_epochs': self.args.n_epochs,
                'gamma': self.args.gamma,
                'gae_lambda': self.args.gae_lambda,
            })
        elif self.args.algorithm == 'SAC':
            config.update({
                'buffer_size': self.args.buffer_size,
                'batch_size': self.args.batch_size,
                'tau': self.args.tau,
                'gamma': self.args.gamma,
            })
        
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            config=config,
            tags=[self.args.algorithm, 'CNN-based', self.args.state_type],
            reinit=True
        )
    
    def setup_environment(self):
        """Setup the tactile insertion environment"""
        print("Setting up tactile insertion environment...")
        
        # Environment creation function
        def make_env():
            return gym.make(
                "tactile_envs/Insertion-v0",
                state_type=self.args.state_type,
                im_size=self.args.im_size,
                no_gripping=self.args.no_gripping,
                no_rotation=self.args.no_rotation,
                tactile_shape=tuple(self.args.tactile_shape),
                max_delta=self.args.max_delta
            )
        
        # Create vectorized environment
        self.env = make_vec_env(make_env, n_envs=self.args.n_envs, vec_env_cls=DummyVecEnv)
        
        print(f"Environment created with {self.args.n_envs} parallel environments")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space: {self.env.observation_space}")
    
    def setup_algorithm(self):
        """Setup the RL algorithm with CNN-based policy"""
        print(f"Setting up {self.args.algorithm} with CNN-based policy...")
        
        policy_kwargs = {
            'features_extractor_class': CNNBasedExtractor,
            'features_extractor_kwargs': {'features_dim': self.args.features_dim},
            'activation_fn': nn.ReLU,  # Use ReLU for robotic tasks
        }
        
        if self.args.algorithm == 'PPO':
            policy_kwargs['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
            self.model = PPO(
                'MultiInputPolicy',
                self.env,
                learning_rate=self.args.learning_rate,
                n_steps=self.args.n_steps,
                batch_size=self.args.batch_size,
                n_epochs=self.args.n_epochs,
                gamma=self.args.gamma,
                gae_lambda=self.args.gae_lambda,
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
        elif self.args.algorithm == 'SAC':
            policy_kwargs['net_arch'] = dict(pi=[256, 256], qf=[256, 256])
            self.model = SAC(
                'MultiInputPolicy',
                self.env,
                learning_rate=self.args.learning_rate,
                buffer_size=self.args.buffer_size,
                batch_size=self.args.batch_size,
                tau=self.args.tau,
                gamma=self.args.gamma,
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
        
        print(f"Algorithm setup complete with CNN-based policy")
    
    def train(self):
        """Main training loop"""
        print(f"Starting CNN-based training for {self.args.total_timesteps} timesteps...")
        
        callbacks = []
        if self.args.use_wandb:
            callbacks.append(WandbCallback())
        
        self.model.learn(
            total_timesteps=self.args.total_timesteps,
            callback=callbacks,
            log_interval=self.args.log_interval,
        )
        
        # Save model
        model_path = Path(self.args.model_save_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        if self.args.use_wandb:
            wandb.finish()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.display:
            self.display.stop()
        if hasattr(self, 'env'):
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description='CNN-based Tactile-Vision RL')
    
    # Environment arguments
    parser.add_argument('--state_type', type=str, default='vision_and_touch',
                       choices=['vision', 'touch', 'vision_and_touch', 'privileged'],
                       help='Type of state observation')
    parser.add_argument('--im_size', type=int, default=64, help='Image size')
    parser.add_argument('--tactile_shape', type=int, nargs=2, default=[32, 32],
                       help='Tactile sensor shape')
    parser.add_argument('--no_gripping', action='store_true', help='Disable gripping')
    parser.add_argument('--no_rotation', action='store_true', help='Disable rotation')
    parser.add_argument('--max_delta', type=float, default=None, help='Max delta for actions')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    
    # Algorithm arguments
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC'],
                       help='RL algorithm')
    parser.add_argument('--total_timesteps', type=int, default=500000, help='Total timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--features_dim', type=int, default=512, help='Feature dimension')
    
    # PPO arguments
    parser.add_argument('--n_steps', type=int, default=2048, help='PPO steps per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    
    # SAC arguments
    parser.add_argument('--buffer_size', type=int, default=50000, help='SAC buffer size')
    parser.add_argument('--tau', type=float, default=0.005, help='SAC tau')
    
    # System arguments
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='tactile-insertion-cnn',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--model_save_path', type=str, default='./models/cnn_model',
                       help='Model save path')
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = None
    try:
        trainer = CNNBasedTrainer(args)
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if trainer:
            trainer.cleanup()


if __name__ == "__main__":
    main()
