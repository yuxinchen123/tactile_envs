#!/usr/bin/env python3
"""
Start CNN-based training with simplified settings
"""

import os
import sys
import subprocess
import signal
import time

def kill_all_training():
    """Kill all existing training processes"""
    print("🔄 Stopping all existing training processes...")
    
    # Kill all python processes with training scripts
    os.system("pkill -f 'python.*train_.*rl.py'")
    time.sleep(3)
    
    print("✅ All training processes stopped")

def start_cnn_training():
    """Start CNN-based training experiments"""
    print("🚀 Starting CNN-based training experiments...")
    
    # Simple 3D action space experiment
    cmd_3d = [
        'python', 'train_cnn_rl.py',
        '--algorithm', 'PPO',
        '--state_type', 'vision_and_touch',
        '--no_gripping',     # Disable gripping (3D actions)
        '--no_rotation',     # Disable rotation (3D actions)
        '--n_envs', '4',
        '--total_timesteps', '300000',
        '--learning_rate', '1e-4',    # Lower learning rate
        '--features_dim', '256',      # Smaller feature dimension
        '--n_steps', '1024',          # Smaller steps for more updates
        '--batch_size', '32',         # Smaller batch size
        '--n_epochs', '4',            # Fewer epochs
        '--gamma', '0.99',
        '--gae_lambda', '0.95',
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-cnn',
        '--wandb_run_name', 'PPO-CNN-3D-VisionTouch-Simple',
        '--log_interval', '1',
        '--model_save_path', './models/cnn_3d_model'
    ]
    
    print("🎯 Starting 3D CNN-based training...")
    print(f"Command: {' '.join(cmd_3d)}")
    
    # Start in background
    process_3d = subprocess.Popen(cmd_3d)
    print(f"✅ 3D CNN training started with PID: {process_3d.pid}")
    
    # Wait a bit before starting next experiment
    time.sleep(10)
    
    # Vision-only experiment for comparison
    cmd_vision = [
        'python', 'train_cnn_rl.py',
        '--algorithm', 'PPO',
        '--state_type', 'vision',     # Vision only
        '--no_gripping',
        '--no_rotation',
        '--n_envs', '4',
        '--total_timesteps', '300000',
        '--learning_rate', '1e-4',
        '--features_dim', '256',
        '--n_steps', '1024',
        '--batch_size', '32',
        '--n_epochs', '4',
        '--gamma', '0.99',
        '--gae_lambda', '0.95',
        '--headless',
        '--use_wandb',
        '--wandb_project', 'tactile-insertion-cnn',
        '--wandb_run_name', 'PPO-CNN-3D-VisionOnly-Simple',
        '--log_interval', '1',
        '--model_save_path', './models/cnn_vision_model'
    ]
    
    print("🎯 Starting Vision-only CNN training...")
    print(f"Command: {' '.join(cmd_vision)}")
    
    # Start in background
    process_vision = subprocess.Popen(cmd_vision)
    print(f"✅ Vision-only CNN training started with PID: {process_vision.pid}")
    
    return process_3d, process_vision

def main():
    print("🛠️  CNN-based Training Launcher")
    print("=" * 60)
    
    # Kill existing training
    kill_all_training()
    
    # Start CNN experiments
    process_3d, process_vision = start_cnn_training()
    
    print("\n" + "=" * 60)
    print("✅ CNN-based experiments started!")
    print("\n📊 Experiments running:")
    print("   • 3D CNN Vision+Touch (most realistic)")
    print("   • 3D CNN Vision-only (baseline)")
    print("\n🏗️  Architecture:")
    print("   • Vision: RGB -> CNN -> Embedding (256D)")
    print("   • Tactile: Force grid -> CNN -> Embedding (256D)")
    print("   • Fusion: Concatenate + MLP -> Policy (256D)")
    print("\n🔍 Monitor progress:")
    print("   • python simple_monitor.py")
    print("   • https://wandb.ai/rl_research/tactile-insertion-cnn")
    print("\n💡 This should work much better!")
    print("   • CNN extractors are proven for robotic vision")
    print("   • 3D actions are much easier to learn")
    print("   • Smaller networks prevent overfitting")

if __name__ == "__main__":
    main()
