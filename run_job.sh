#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --account=gusr38169
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=tmpt-lt
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/toy-models-of-preference-learning/outputs/logs/%j.out
#SBATCH --error=/gpfs/work5/0/gusr0688/fair_stuff/toy-models-of-preference-learning/outputs/logs/logs-err/%j.err


SCRIPT="$1"
shift

source ~/.bashrc
conda activate llmenv

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Script: $SCRIPT"
echo "Args: $@"
echo "=========================================="

export PYTHONUNBUFFERED=1
cd /gpfs/work5/0/gusr0688/fair_stuff/toy-models-of-preference-learning
python -u "trainer/$SCRIPT" "$@"

echo "=========================================="
echo "Done"

