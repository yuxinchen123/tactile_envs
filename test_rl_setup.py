#!/usr/bin/env python3
"""
Test script for the RL training setup
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import tactile_envs

# Set environment variables for headless mode
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def test_environment():
    """Test that the environment works properly"""
    print("Testing environment...")
    
    env = gym.make('tactile_envs/Insertion-v0', 
                   state_type='vision_and_touch', 
                   im_size=64, 
                   tactile_shape=(32, 32))
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    
    for key, value in obs.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, terminated={terminated}, success={info.get('is_success', False)}")
    
    env.close()
    print("Environment test completed!")

def test_features_extractor():
    """Test the custom features extractor"""
    print("Testing features extractor...")
    
    from train_rl_insertion import TactileVisionFeaturesExtractor
    
    # Create mock observation space
    obs_space = gym.spaces.Dict({
        'image': gym.spaces.Box(0, 1, (64, 64, 3), dtype=np.float64),
        'tactile': gym.spaces.Box(-np.inf, np.inf, (6, 32, 32), dtype=np.float64),
    })
    
    extractor = TactileVisionFeaturesExtractor(obs_space, features_dim=512)
    
    # Test forward pass
    batch_size = 4
    obs = {
        'image': torch.randn(batch_size, 64, 64, 3),
        'tactile': torch.randn(batch_size, 6, 32, 32),
    }
    
    features = extractor(obs)
    print(f"Features shape: {features.shape}")
    print(f"Expected shape: (batch_size={batch_size}, features_dim=512)")
    
    assert features.shape == (batch_size, 512), f"Expected shape (4, 512), got {features.shape}"
    print("Features extractor test passed!")

def test_different_state_types():
    """Test different state types"""
    print("Testing different state types...")
    
    from train_rl_insertion import TactileVisionFeaturesExtractor
    
    state_types = ['vision', 'touch', 'vision_and_touch', 'privileged']
    
    for state_type in state_types:
        print(f"\nTesting {state_type}...")
        
        # Create environment
        env = gym.make('tactile_envs/Insertion-v0', 
                       state_type=state_type, 
                       im_size=64, 
                       tactile_shape=(32, 32))
        
        # Test features extractor
        extractor = TactileVisionFeaturesExtractor(env.observation_space, features_dim=256)
        
        # Get sample observation
        obs, _ = env.reset()
        
        # Convert to torch and add batch dimension
        torch_obs = {}
        for key, value in obs.items():
            torch_obs[key] = torch.from_numpy(value).unsqueeze(0)
        
        # Test forward pass
        features = extractor(torch_obs)
        print(f"  Features shape: {features.shape}")
        
        env.close()
    
    print("All state types test passed!")

if __name__ == "__main__":
    test_environment()
    test_features_extractor()
    test_different_state_types()
    print("\nAll tests passed! 🎉")
