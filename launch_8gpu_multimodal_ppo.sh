#!/bin/bash
# Launch 8 parallel PPO jobs, one per GPU, each with unique seed and WandB run name (no tmux)
CONDA_ENV=tactile_envs
for i in {0..7}
do
  bash -i -c "conda activate $CONDA_ENV && CUDA_VISIBLE_DEVICES=$i python tactile_envs/ppo_multimodal_training.py --seed $i --job_id $i &"
done
wait
echo "All 8 PPO jobs launched in background. Use 'ps aux | grep ppo_multimodal_training.py' to check."
