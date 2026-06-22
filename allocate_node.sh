#!/bin/bash

# Default time is 1 hour if no argument is provided
HOURS=${1:-1}

srun --partition=gpu_a100 \
     --nodes=1 \
     --gpus-per-node=1 \
     --time=$(printf "%02d:00:00" "$HOURS") \
     --account=gusr38169 \
     --pty bash -c "
        source ~/.bashrc
        conda activate llmenv
        cd /gpfs/work5/0/gusr0688/fair_stuff/toy-models-of-preference-learning
        exec bash
     "