#!/bin/bash

# Multi-GPU PPO Training Launcher
# Launches 8 separate training processes, each on a different GPU
# Each process logs separately to WandB

echo "🚀 Starting 8-GPU multimodal PPO training..."
echo "📊 Each GPU will run 4 environments with separate WandB logging"
echo "🎯 Total: 32 parallel environments across 8 GPUs"

# Set common parameters
SEED=42
PROJECT_NAME="multimodal-ppo-8gpu"

# Kill any existing training processes
echo "🧹 Cleaning up existing processes..."
pkill -f ppo_multimodal_training.py

# Start training on each GPU
for gpu_id in {0..7}; do
    echo "🔧 Starting training on GPU ${gpu_id}..."
    
    # Set environment variables for headless rendering
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    
    # Start training process in background
    nohup python ppo_multimodal_training.py \
        --wandb \
        --gpu_id ${gpu_id} \
        --job_id ${gpu_id} \
        --seed $((SEED + gpu_id)) \
        > logs/gpu_${gpu_id}_training.log 2>&1 &
    
    # Store process ID
    echo $! > logs/gpu_${gpu_id}_pid.txt
    
    echo "✅ GPU ${gpu_id}: Started process $(cat logs/gpu_${gpu_id}_pid.txt)"
    
    # Small delay to avoid startup conflicts
    sleep 2
done

echo ""
echo "🎯 All 8 training processes started!"
echo "📂 Logs saved to: logs/gpu_X_training.log"
echo "🔍 Process IDs saved to: logs/gpu_X_pid.txt"
echo "📊 WandB project: ${PROJECT_NAME}"
echo ""
echo "🔧 Monitor training with:"
echo "  tail -f logs/gpu_0_training.log  # View GPU 0 logs"
echo "  nvidia-smi                       # Check GPU usage"
echo "  ps aux | grep ppo_multimodal     # Check running processes"
echo ""
echo "🛑 Stop all training:"
echo "  pkill -f ppo_multimodal_training.py"
