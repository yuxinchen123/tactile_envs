#!/bin/bash
# Unset conda-related variables to avoid tmux library conflicts
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PROMPT_MODIFIER
unset _CONDA_EXE
unset _CE_CONDA
unset _CE_M
unset CONDA_SHLVL
unset CONDA_EXE
unset CONDA_PYTHON_EXE
unset LD_LIBRARY_PATH

# Launch 8 parallel PPO jobs in tmux windows, one per GPU, each with unique seed and WandB run name
SESSION=ppo_multimodal_8gpu
CONDA_ENV=tactile_envs

# Start tmux session
if ! tmux has-session -t $SESSION 2>/dev/null; then
  tmux new-session -d -s $SESSION
fi
for i in {0..7}
  do
    tmux new-window -t $SESSION:$i -n gpu_$i
    tmux send-keys -t $SESSION:$i "bash -i -c 'conda activate $CONDA_ENV && CUDA_VISIBLE_DEVICES=$i python ppo_multimodal_training.py --seed $i --job_id $i'" C-m
  done
echo "All 8 PPO jobs launched in tmux session '$SESSION'. Attach with: tmux attach-session -t $SESSION"
