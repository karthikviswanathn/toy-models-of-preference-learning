#!/bin/bash
# Sweep SFT using matching PT models as base.
# Run this AFTER sweep_full.sh PT jobs have completed.
# 3 × 4 × 3 × 2 × 2 = 144 jobs

WEIGHT_DECAYS=(0.15 0.3 0.5)
BATCH_SIZES=(256 512 1024 -1)
MODEL_SEEDS=(1234 1235 1236)
SPLIT_SEEDS=(41 42)
SHUFFLE_SEEDS=(43 44)

COUNT=0
SKIPPED=0

for wd in "${WEIGHT_DECAYS[@]}"; do
for bs in "${BATCH_SIZES[@]}"; do
for ms in "${MODEL_SEEDS[@]}"; do
for ss in "${SPLIT_SEEDS[@]}"; do
for sh in "${SHUFFLE_SEEDS[@]}"; do

    PT_DIR=$(ls -d outputs/runs/pt/pt_wd${wd}_bs${bs}_ms${ms}_ss${ss}_sh${sh}_*/ 2>/dev/null | head -1)

    if [ -z "$PT_DIR" ] || [ ! -f "${PT_DIR}/model.pt" ]; then
        echo "SKIP: no PT model for wd=${wd} bs=${bs} ms=${ms} ss=${ss} sh=${sh}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    sbatch --export=ALL,WANDB_PROJECT=toy-preference-sweep-sft \
           --job-name="sft-wd${wd}-bs${bs}-ms${ms}-ss${ss}-sh${sh}" \
           run_job.sh sft.py \
           --base_model "${PT_DIR}/model.pt" \
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

echo "Submitted $COUNT SFT jobs, skipped $SKIPPED"
