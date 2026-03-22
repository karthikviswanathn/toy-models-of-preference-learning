#!/bin/bash
# Sweep: weight_decay × batch_size for PT-G
# wd: 0.2, 0.5
# bs: 256, 512, 1024

WEIGHT_DECAYS=(0.2 0.5)
BATCH_SIZES=(256 512 1024)

for wd in "${WEIGHT_DECAYS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        echo "Submitting: wd=$wd, bs=$bs"
        sbatch --export=ALL,WANDB_PROJECT=toy-preference-wd-bs-sweep \
               --job-name="ptg-wd${wd}-bs${bs}" \
               run_job.sh pretrain_gated.py \
               --weight_decay "$wd" \
               --batch_size "$bs"
    done
done
