#!/usr/bin/env python3
"""
Improved Training Script with Better Hyperparameters and Curriculum Learning
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def kill_all_training():
    """Kill all existing training processes"""
    print("🔄 Stopping all training processes...")
    os.system("pkill -f 'python.*train_embedding_rl.py'")
    time.sleep(3)

def start_easier_training():
    """Start with easier configuration first"""
    print("🎯 Starting with easier configuration...")
    
    # Phase 1: Start with vision-only, 3D actions, more parallel envs
    cmd = [
        'python', 'train_embedding_rl.py',
        '--algorithm', 'PPO',
        '--state_type', 'vision',  # Vision only first
        '--n_envs', '8',           # More parallel environments
        '--total_timesteps', '200000',  # More timesteps
        '--learning_rate', '1e-4',      # Lower learning rate
        '--n_steps', '4096',            # Larger rollout buffer
        '--batch_size', '128',          # Larger batch size
        '--n_epochs', '4',              # Fewer epochs per update
        '--gamma', '0.995',             # Higher discount factor
        '--gae_lambda', '0.98',         # Higher GAE lambda
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-experiments',
        '--wandb_run_name', 'PPO-Vision-Only-3D-Easier',
        '--log_interval', '1',
        '--model_save_path', './models/ppo_vision_easier',
        '--no_rotation',               # Disable rotation for easier task
        '--no_gripping'                # Disable gripping for easier task
    ]
    
    print(f"🚀 Starting easier training: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    return process

def start_medium_training():
    """Start medium difficulty after easier one"""
    print("🎯 Starting medium difficulty training...")
    
    # Phase 2: Add tactile but keep 3D actions
    cmd = [
        'python', 'train_embedding_rl.py',
        '--algorithm', 'PPO',
        '--state_type', 'vision_and_touch',
        '--n_envs', '6',
        '--total_timesteps', '300000',
        '--learning_rate', '5e-5',      # Even lower learning rate
        '--n_steps', '4096',
        '--batch_size', '256',
        '--n_epochs', '8',
        '--gamma', '0.99',
        '--gae_lambda', '0.95',
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-experiments',
        '--wandb_run_name', 'PPO-Vision+Touch-3D-Medium',
        '--log_interval', '1',
        '--model_save_path', './models/ppo_multimodal_medium',
        '--no_rotation'                # Still no rotation
    ]
    
    print(f"🚀 Starting medium training: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    return process

def start_sac_training():
    """Start SAC with better hyperparameters"""
    print("🎯 Starting SAC with optimized hyperparameters...")
    
    cmd = [
        'python', 'train_embedding_rl.py',
        '--algorithm', 'SAC',
        '--state_type', 'vision_and_touch',
        '--n_envs', '1',
        '--total_timesteps', '500000',  # More timesteps for SAC
        '--learning_rate', '1e-4',      # Lower learning rate
        '--buffer_size', '100000',      # Larger buffer
        '--batch_size', '512',          # Larger batch size
        '--tau', '0.01',                # Slightly higher tau
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-experiments',
        '--wandb_run_name', 'SAC-Vision+Touch-Optimized',
        '--log_interval', '1',
        '--model_save_path', './models/sac_optimized'
    ]
    
    print(f"🚀 Starting SAC training: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    return process

def create_monitoring_script():
    """Create a script to monitor all training processes"""
    
    script_content = """#!/usr/bin/env python3
import subprocess
import time
import os
from pathlib import Path

def monitor_training():
    while True:
        print("\\n" + "="*60)
        print("🔍 Training Status Check")
        print("="*60)
        
        # Check running processes
        result = subprocess.run(['pgrep', '-f', 'train_embedding_rl.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\\n')
            print(f"🟢 Found {len(pids)} training process(es): {pids}")
        else:
            print("🔴 No training processes found")
        
        # Check video files
        video_dir = Path("./videos")
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            print(f"📹 Videos: {len(videos)} files")
            if videos:
                latest = max(videos, key=lambda x: x.stat().st_mtime)
                print(f"   Latest: {latest.name}")
        
        # Check model files
        models_dir = Path("./models")
        if models_dir.exists():
            models = list(models_dir.glob("*"))
            print(f"🤖 Models: {len(models)} files")
        
        print("\\n⏱️  Waiting 30 seconds...")
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\\n👋 Monitoring stopped")
"""
    
    with open("monitor_all_training.py", "w") as f:
        f.write(script_content)
    
    print("📊 Created monitoring script: monitor_all_training.py")

def main():
    parser = argparse.ArgumentParser(description='Improved Training Launcher')
    parser.add_argument('--phase', choices=['easy', 'medium', 'sac', 'all'], 
                       default='all', help='Training phase to run')
    
    args = parser.parse_args()
    
    print("🚀 Improved Training Launcher")
    print("=" * 60)
    
    # Kill existing training
    kill_all_training()
    
    # Create monitoring script
    create_monitoring_script()
    
    processes = []
    
    if args.phase in ['easy', 'all']:
        processes.append(start_easier_training())
        time.sleep(5)  # Stagger starts
    
    if args.phase in ['medium', 'all']:
        processes.append(start_medium_training())
        time.sleep(5)
    
    if args.phase in ['sac', 'all']:
        processes.append(start_sac_training())
    
    print("\n" + "=" * 60)
    print("✅ Training processes started!")
    print("🔍 Monitor with: python monitor_all_training.py")
    print("📊 Wandb: https://wandb.ai/rl_research/tactile-insertion-experiments")
    print("💡 These configurations should work much better!")

if __name__ == "__main__":
    main()
