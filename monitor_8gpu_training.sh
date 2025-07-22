#!/bin/bash

# Multi-GPU Training Monitor
# Checks status of all 8 training processes

echo "🔍 Multi-GPU Training Monitor"
echo "=============================="

# Check if any training processes are running
running_processes=$(ps aux | grep "ppo_multimodal_training.py" | grep -v grep | wc -l)
echo "📊 Running training processes: ${running_processes}/8"
echo ""

# Check each GPU
for gpu_id in {0..7}; do
    pid_file="logs/gpu_${gpu_id}_pid.txt"
    log_file="logs/gpu_${gpu_id}_training.log"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            # Process is running
            echo "🟢 GPU ${gpu_id}: Running (PID: ${pid})"
            
            # Show last few lines of log
            if [ -f "$log_file" ]; then
                echo "   📝 Latest: $(tail -n 1 "$log_file" 2>/dev/null | cut -c1-80)..."
            fi
        else
            echo "🔴 GPU ${gpu_id}: Stopped (PID: ${pid} not found)"
        fi
    else
        echo "⚪ GPU ${gpu_id}: No PID file found"
    fi
done

echo ""
echo "💾 GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU %s: %s MB / %s MB (%.1f%%) - %s%% util\n", $1, $3, $4, ($3/$4)*100, $5}'

echo ""
echo "🔧 Commands:"
echo "  ./launch_8gpu_separate_training.sh  # Start all training"
echo "  pkill -f ppo_multimodal_training.py # Stop all training"
echo "  tail -f logs/gpu_X_training.log     # View specific GPU log"
