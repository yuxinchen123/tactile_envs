#!/usr/bin/env python3
"""
Simplified 3D Action Training Script
Restart with easier environment configuration
"""

import os
import sys
import subprocess
import time

def start_simple_3d_training():
    """Start training with simplified 3D action space"""
    print("🚀 Starting simplified 3D action training...")
    
    # Kill any existing processes first
    print("🔄 Cleaning up existing processes...")
    os.system("pkill -f 'python.*train_embedding_rl.py'")
    time.sleep(3)
    
    # PPO with 3D actions (translation only)
    ppo_cmd = [
        'python', 'train_embedding_rl.py',
        '--algorithm', 'PPO',
        '--state_type', 'vision_and_touch',
        '--n_envs', '4',
        '--total_timesteps', '300000',
        '--learning_rate', '3e-4',
        '--features_dim', '256',        # Reasonable size
        '--n_steps', '2048',
        '--batch_size', '64',
        '--n_epochs', '10',
        '--gamma', '0.99',
        '--gae_lambda', '0.95',
        '--no_gripping',               # Disable gripping (4D -> 3D)
        '--no_rotation',               # Disable rotation (5D -> 3D)
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-simple',  # New project
        '--wandb_run_name', 'PPO-3D-Vision+Touch-Simple',
        '--log_interval', '10',
        '--model_save_path', './models/ppo_3d_simple'
    ]
    
    print(f"Starting PPO 3D: {' '.join(ppo_cmd)}")
    ppo_process = subprocess.Popen(ppo_cmd)
    print(f"✅ PPO 3D training started with PID: {ppo_process.pid}")
    
    return ppo_process

def start_privileged_3d_training():
    """Start privileged training with 3D actions for comparison"""
    print("🚀 Starting privileged 3D action training...")
    
    time.sleep(5)  # Wait a bit before starting second experiment
    
    # Privileged with 3D actions
    privileged_cmd = [
        'python', 'train_embedding_rl.py',
        '--algorithm', 'PPO',
        '--state_type', 'privileged',   # Ground truth state
        '--n_envs', '4',
        '--total_timesteps', '200000',  # Should learn faster
        '--learning_rate', '3e-4',
        '--features_dim', '128',        # Smaller for privileged
        '--n_steps', '2048',
        '--batch_size', '64',
        '--n_epochs', '10',
        '--gamma', '0.99',
        '--gae_lambda', '0.95',
        '--no_gripping',               # Disable gripping (4D -> 3D)
        '--no_rotation',               # Disable rotation (5D -> 3D)
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-simple',  # Same project
        '--wandb_run_name', 'PPO-3D-Privileged-GroundTruth',
        '--log_interval', '10',
        '--model_save_path', './models/ppo_3d_privileged'
    ]
    
    print(f"Starting Privileged 3D: {' '.join(privileged_cmd)}")
    privileged_process = subprocess.Popen(privileged_cmd)
    print(f"✅ Privileged 3D training started with PID: {privileged_process.pid}")
    
    return privileged_process

def start_vision_only_3d_training():
    """Start vision-only training with 3D actions"""
    print("🚀 Starting vision-only 3D action training...")
    
    time.sleep(5)  # Wait a bit before starting third experiment
    
    # Vision only with 3D actions
    vision_cmd = [
        'python', 'train_embedding_rl.py',
        '--algorithm', 'PPO',
        '--state_type', 'vision',       # Vision only
        '--n_envs', '4',
        '--total_timesteps', '400000',  # More timesteps for vision-only
        '--learning_rate', '1e-4',      # Lower learning rate
        '--features_dim', '256',
        '--n_steps', '1024',           # Smaller steps for more frequent updates
        '--batch_size', '32',          # Smaller batch
        '--n_epochs', '4',             # Fewer epochs
        '--gamma', '0.99',
        '--gae_lambda', '0.95',
        '--no_gripping',               # Disable gripping (4D -> 3D)
        '--no_rotation',               # Disable rotation (5D -> 3D)
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-simple',  # Same project
        '--wandb_run_name', 'PPO-3D-Vision-Only',
        '--log_interval', '10',
        '--model_save_path', './models/ppo_3d_vision'
    ]
    
    print(f"Starting Vision-Only 3D: {' '.join(vision_cmd)}")
    vision_process = subprocess.Popen(vision_cmd)
    print(f"✅ Vision-only 3D training started with PID: {vision_process.pid}")
    
    return vision_process

def create_monitoring_script():
    """Create a simple monitoring script for the new experiments"""
    monitoring_script = '''#!/usr/bin/env python3
"""
Monitor the simplified 3D training experiments
"""

import subprocess
import time
from pathlib import Path

def check_training_status():
    print("🔍 Checking 3D Training Status")
    print("=" * 50)
    
    # Check for training processes
    result = subprocess.run(['pgrep', '-f', 'python.*train_embedding_rl.py'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        pids = result.stdout.strip().split('\\n')
        print(f"✅ Found {len(pids)} training processes running")
        for pid in pids:
            print(f"   PID: {pid}")
    else:
        print("❌ No training processes found")
    
    # Check video files
    video_dir = Path("./videos")
    if video_dir.exists():
        videos = list(video_dir.glob("*.mp4"))
        print(f"\\n📹 Videos: {len(videos)} files")
        for video in sorted(videos)[-3:]:  # Show last 3
            size_kb = video.stat().st_size / 1024
            print(f"   {video.name} ({size_kb:.1f} KB)")
    
    print("\\n📊 Wandb Project: https://wandb.ai/rl_research/tactile-insertion-simple")
    print("💡 Expected: 3D actions should learn faster than 5D!")

if __name__ == "__main__":
    while True:
        check_training_status()
        print("\\n" + "="*50)
        time.sleep(60)  # Check every minute
'''
    
    with open('monitor_3d_training.py', 'w') as f:
        f.write(monitoring_script)
    
    print("📊 Created monitoring script: monitor_3d_training.py")

def main():
    print("🎯 Simplified 3D Action Training Setup")
    print("=" * 60)
    
    print("🔧 Configuration:")
    print("   • Action Space: 3D (x, y, z translation only)")
    print("   • No gripping or rotation")
    print("   • Multiple baselines for comparison")
    print("   • New wandb project: tactile-insertion-simple")
    
    # Start all training experiments
    ppo_process = start_simple_3d_training()
    privileged_process = start_privileged_3d_training()
    vision_process = start_vision_only_3d_training()
    
    # Create monitoring script
    create_monitoring_script()
    
    print("\n" + "=" * 60)
    print("✅ All 3D training experiments started!")
    print("\n🚀 Running experiments:")
    print("   1. PPO 3D Vision+Touch (multimodal)")
    print("   2. PPO 3D Privileged (ground truth)")
    print("   3. PPO 3D Vision-Only")
    print("\n🔍 Monitor progress:")
    print("   • python monitor_3d_training.py")
    print("   • https://wandb.ai/rl_research/tactile-insertion-simple")
    print("\n💡 Expected outcomes:")
    print("   • 3D actions should be much easier to learn")
    print("   • Privileged should learn fastest")
    print("   • Should see positive rewards within 50k timesteps")

if __name__ == "__main__":
    main()
