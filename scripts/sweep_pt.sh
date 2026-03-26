#!/bin/bash
# Sweep PT: wd × bs × model_seed × data_seed
# 3 × 3 × 3 × 3 = 81 jobs

WEIGHT_DECAYS=(0.15 0.3 0.5)
BATCH_SIZES=(512 1024 -1)
MODEL_SEEDS=(1234 1235 1236)
DATA_SEEDS=(42 43 44)

COUNT=0

for wd in "${WEIGHT_DECAYS[@]}"; do
for bs in "${BATCH_SIZES[@]}"; do
for ms in "${MODEL_SEEDS[@]}"; do
for ds in "${DATA_SEEDS[@]}"; do

    sbatch --export=ALL,WANDB_PROJECT=toy-preference-sweep-pt-p106 \
           --job-name="pt-wd${wd}-bs${bs}-ms${ms}-ds${ds}" \
           run_job.sh trainer/pretrain.py \
           --weight_decay "$wd" \
           --batch_size "$bs" \
           --model_seed "$ms" \
           --data_seed "$ds"

    COUNT=$((COUNT + 1))

done
done
done
done

echo "Submitted $COUNT PT jobs"
