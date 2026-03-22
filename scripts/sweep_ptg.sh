#!/bin/bash
# Sweep PT-G: wd × bs × model_seed × split_seed × shuffle_seed
# 3 × 4 × 3 × 2 × 2 = 144 jobs

WEIGHT_DECAYS=(0.15 0.3 0.5)
BATCH_SIZES=(256 512 1024 -1)
MODEL_SEEDS=(1234 1235 1236)
SPLIT_SEEDS=(41 42)
SHUFFLE_SEEDS=(43 44)

COUNT=0

for wd in "${WEIGHT_DECAYS[@]}"; do
for bs in "${BATCH_SIZES[@]}"; do
for ms in "${MODEL_SEEDS[@]}"; do
for ss in "${SPLIT_SEEDS[@]}"; do
for sh in "${SHUFFLE_SEEDS[@]}"; do

    sbatch --export=ALL,WANDB_PROJECT=toy-preference-sweep-ptg \
           --job-name="ptg-wd${wd}-bs${bs}-ms${ms}-ss${ss}-sh${sh}" \
           run_job.sh pretrain_gated.py \
           --weight_decay "$wd" \
           --batch_size "$bs" \
           --model_seed "$ms" \
           --split_seed "$ss" \
           --shuffle_seed "$sh"

    COUNT=$((COUNT + 1))

done
done
done
done
done

echo "Submitted $COUNT PT-G jobs"
