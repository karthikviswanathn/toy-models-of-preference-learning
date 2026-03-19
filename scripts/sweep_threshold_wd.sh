#!/bin/bash
# Sweep: unsafe_threshold × weight_decay for PT-G
# Thresholds: 113 × (1/2, 2/3, 3/4, 4/5) ≈ 57, 75, 85, 90
# Weight decays: 0.1, 0.2, 0.3, 0.5

THRESHOLDS=(57 75 85 90)
WEIGHT_DECAYS=(0.1 0.2 0.3 0.5)

for th in "${THRESHOLDS[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
        echo "Submitting: threshold=$th, wd=$wd"
        sbatch --export=ALL,WANDB_PROJECT=toy-preference-threshold-sweep \
               --job-name="ptg-th${th}-wd${wd}" \
               run_job.sh pretrain_gated.py \
               --unsafe_threshold "$th" \
               --weight_decay "$wd"
    done
done
