#!/bin/bash
# Sweep: adam_eps for PT-G
# eps: 1e-4, 1e-5, 1e-6, 1e-7, 1e-8

EPS_VALUES=(1e-4 1e-5 1e-6 1e-7 1e-8)

for eps in "${EPS_VALUES[@]}"; do
    echo "Submitting: adam_eps=$eps"
    sbatch --export=ALL,WANDB_PROJECT=toy-preference-threshold-sweep \
           --job-name="ptg-eps${eps}" \
           run_job.sh pretrain_gated.py \
           --adam_eps "$eps"
done
