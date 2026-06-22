#!/bin/bash
# Sweep PT-G over 3 weight decays x 3 model seeds = 9 jobs

WEIGHT_DECAYS=(0.1 0.2 0.5)
MODEL_SEEDS=(1234 5678 9012)

for wd in "${WEIGHT_DECAYS[@]}"; do
    for ms in "${MODEL_SEEDS[@]}"; do
        echo "Submitting: wd=$wd model_seed=$ms"
        sbatch --export=ALL,WANDB_PROJECT=toy-preference-less-than-sweep-seeds \
        run_job.sh pretrain_gated.py \
            --weight_decay "$wd" \
            --model_seed "$ms"
    done
done

echo "Submitted ${#WEIGHT_DECAYS[@]} x ${#MODEL_SEEDS[@]} = $(( ${#WEIGHT_DECAYS[@]} * ${#MODEL_SEEDS[@]} )) jobs"
