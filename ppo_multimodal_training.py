"""
PPO Multimodal Training Script for Tactile Environment
- Inputs: RGB image, tactile force grid, joint angles (extracted from privileged state)
- Encoders: CNN for image/tactile, MLP for joints
- Fusion: Concatenate embeddings, MLP
- Actor/Critic: Separate heads from fused embedding
- Backend: RSL-RL PPO (PyTorch)
"""
import os
# Set up headless rendering for MuJoCo
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gymnasium as gym
import tactile_envs  # This registers the tactile_envs/Insertion-v0 environment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
import imageio
from pathlib import Path

# --- Custom Environment Wrapper ---
class MultimodalWrapper(gym.Wrapper):
    """
    Wrapper that extracts joint angles from privileged state and combines with vision/tactile data.
    Creates a multimodal observation space with image, tactile, and joints.
    """
    def __init__(self, env):
        # Start with the sensory environment
        super().__init__(env)
        
        # Store original state type
        self.original_state_type = env.state_type
        
        # Define new observation space
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32),
            'tactile': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6, 32, 32), dtype=np.float32),
            'joints': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)  # Actual qpos size
        })
    
    def _get_privileged_obs(self):
        """Temporarily switch to privileged mode to get joint angles."""
        # Store current state type
        current_state_type = self.env.state_type
        
        # Switch to privileged mode
        self.env.state_type = 'privileged'
        
        # Get privileged observation by directly accessing the environment's qpos
        # Since we're in the same simulation state, we can directly read joint angles
        joints = self.env.unwrapped.mj_data.qpos.copy()  # Extract all joint angles (19 values)
        
        # Restore original state type
        self.env.state_type = current_state_type
        
        return joints.astype(np.float32)
    
    def reset(self, **kwargs):
        # Reset with vision_and_touch mode
        self.env.state_type = 'vision_and_touch'
        sensory_obs, info = self.env.reset(**kwargs)
        
        # Get joint angles from privileged state
        joints = self._get_privileged_obs()
        
        # Combine observations
        multimodal_obs = {
            'image': sensory_obs['image'].astype(np.float32),
            'tactile': sensory_obs['tactile'].astype(np.float32),
            'joints': joints
        }
        
        return multimodal_obs, info
    
    def step(self, action):
        # Step with vision_and_touch mode
        self.env.state_type = 'vision_and_touch'
        sensory_obs, reward, done, truncated, info = self.env.step(action)
        
        # Get joint angles from privileged state
        joints = self._get_privileged_obs()
        
        # Combine observations
        multimodal_obs = {
            'image': sensory_obs['image'].astype(np.float32),
            'tactile': sensory_obs['tactile'].astype(np.float32),
            'joints': joints
        }
        
        return multimodal_obs, reward, done, truncated, info

class VecMultimodalWrapper(VecEnvWrapper):
    """Vector environment wrapper for multimodal observations."""
    def __init__(self, venv):
        super().__init__(venv)
        # Update observation space
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32),
            'tactile': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6, 32, 32), dtype=np.float32),
            'joints': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)  # Actual qpos size
        })
    
    def reset(self):
        obs = self.venv.reset()
        return self._transform_obs(obs)
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._transform_obs(obs), rewards, dones, infos
    
    def _transform_obs(self, obs):
        # obs is already in multimodal format from individual wrappers
        return obs

# --- Encoder Modules ---
class CNNEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim=256, input_size=64):
        super().__init__()
        # Calculate dimensions after each conv layer
        # Conv1: 64->32 (stride=2), Conv2: 32->16 (stride=2)
        conv1_out_size = input_size // 2  # 32 for 64x64, 16 for 32x32
        conv2_out_size = conv1_out_size // 2  # 16 for 64x64, 8 for 32x32
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),  # Use BatchNorm instead of LayerNorm for spatial data
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * conv2_out_size * conv2_out_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cnn(x)

class JointEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.mlp(x)

class MultimodalFusion(nn.Module):
    def __init__(self, embed_dim=256, fused_dim=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
        )
    def forward(self, img, tactile, joints):
        x = torch.cat([img, tactile, joints], dim=1)
        return self.fusion(x)

# --- Custom Feature Extractor for SB3 ---
class MultimodalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=256):
        super().__init__(observation_space, features_dim=embed_dim)
        img_shape = observation_space['image'].shape
        tactile_shape = observation_space['tactile'].shape  
        joint_dim = observation_space['joints'].shape[0]  # Should be 37
        
        print(f"Initializing MultimodalFeatureExtractor:")
        print(f"  Image shape: {img_shape}")
        print(f"  Tactile shape: {tactile_shape}")
        print(f"  Joint dimension: {joint_dim}")
        
        # Image encoder: 64x64x3 -> embedding
        self.img_encoder = CNNEncoder(img_shape[2], embed_dim, input_size=img_shape[0])
        # Tactile encoder: 6x32x32 -> embedding  
        self.tactile_encoder = CNNEncoder(tactile_shape[0], embed_dim, input_size=tactile_shape[1])
        # Joint encoder: joint_dim -> embedding
        self.joint_encoder = JointEncoder(joint_dim, embed_dim)
        self.fusion = MultimodalFusion(embed_dim, embed_dim)
        
    def forward(self, obs):
        # Image: (batch, height, width, channels) -> (batch, channels, height, width)
        img = obs['image'].float().permute(0, 3, 1, 2) / 255.0  # Normalize to [0,1]
        
        # Tactile: (batch, channels, height, width) - already in correct format
        tactile = obs['tactile'].float()
        
        # Joints: (batch, joint_dim) - 37 joint angles from qpos
        joints = obs['joints'].float()
        # Normalize joints (optional but recommended)
        joints = (joints - joints.mean(dim=1, keepdim=True)) / (joints.std(dim=1, keepdim=True) + 1e-6)
        
        img_emb = self.img_encoder(img)
        tactile_emb = self.tactile_encoder(tactile)
        joint_emb = self.joint_encoder(joints)
        
        fused = self.fusion(img_emb, tactile_emb, joint_emb)
        return fused

class WandbVideoCallback(BaseCallback):
    def __init__(self, video_freq=2000, verbose=0, wandb_enabled=True):
        super().__init__(verbose)
        self.video_freq = video_freq
        self.video_dir = Path("./videos")
        self.video_dir.mkdir(exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wandb_enabled = wandb_enabled
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info and self.wandb_enabled:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                wandb.log({
                    'episode/reward': info['episode']['r'],
                    'episode/length': info['episode']['l'],
                    'global_step': self.num_timesteps,
                })
        if self.wandb_enabled:
            if 'loss' in self.locals:
                wandb.log({'loss': self.locals['loss'], 'global_step': self.num_timesteps})
            if 'policy_loss' in self.locals:
                wandb.log({'policy_loss': self.locals['policy_loss'], 'global_step': self.num_timesteps})
            if 'value_loss' in self.locals:
                wandb.log({'value_loss': self.locals['value_loss'], 'global_step': self.num_timesteps})
            if 'entropy_loss' in self.locals:
                wandb.log({'entropy_loss': self.locals['entropy_loss'], 'global_step': self.num_timesteps})
        if self.num_timesteps % self.video_freq == 0:
            self.record_video()
        return True
    def record_video(self):
        try:
            env = self.training_env.envs[0]
            frames = []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            for _ in range(100):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
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
            if frames and self.wandb_enabled:
                video_path = self.video_dir / f"policy_video_{self.num_timesteps:06d}.mp4"
                imageio.mimsave(str(video_path), frames, fps=20)
                wandb.log({"policy_video": wandb.Video(str(video_path), fps=20, format="mp4"), "global_step": self.num_timesteps})
        except Exception as e:
            print(f"Warning: Could not record video: {e}")

# --- Main Training Script ---
def make_multimodal_env(env_id, rank, seed=0):
    """Create a single multimodal environment."""
    def _init():
        # Create vision+tactile environment
        env = gym.make(env_id, state_type='vision_and_touch')
        env.seed(seed + rank)
        # Wrap with multimodal wrapper
        env = MultimodalWrapper(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="multimodal-ppo", entity="catresearch", name=f"job_{args.job_id}", config=vars(args))

    env_id = 'tactile_envs/Insertion-v0'
    
    # Create vectorized environment with multimodal wrapper
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env_fns = [make_multimodal_env(env_id, i, args.seed) for i in range(8)]
    env = SubprocVecEnv(env_fns)
    
    policy_kwargs = dict(
        features_extractor_class=MultimodalFeatureExtractor,
        features_extractor_kwargs=dict(embed_dim=256),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=nn.ReLU,
    )
    model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, verbose=2, batch_size=256, n_steps=2048, device='auto', seed=args.seed)
    model.learn(total_timesteps=2_000_000, callback=WandbVideoCallback(video_freq=2000, wandb_enabled=args.wandb))
    model.save(f'ppo_multimodal_tactile_job{args.job_id}')
    if args.wandb:
        wandb.save(f'ppo_multimodal_tactile_job{args.job_id}.zip')
        wandb.finish()

if __name__ == '__main__':
    main()
