#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=project_465002390
#SBATCH --job-name=tmpt-lt
#SBATCH --output=/pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning/outputs/logs/%j.out
#SBATCH --error=/pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning/outputs/logs/logs-err/%j.err

set -e

# Per-job cache dirs to avoid filesystem lock contention across concurrent jobs
export TMPDIR="/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export WANDB_DIR="$TMPDIR"
export HF_HOME="$TMPDIR/hf"
export XDG_CACHE_HOME="$TMPDIR/cache"

source /project/project_465002390/fair_stuff/simplex-research/.claude/activate_env.sh
cd /pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning

SCRIPT="$1"
shift

echo "=== Running: $SCRIPT $@ ==="
echo "[$(date)] Starting Python..."
python -u -c "
import sys; print(f'[STARTUP] Python {sys.version}', flush=True)
print('[STARTUP] Importing torch...', flush=True)
import torch; print(f'[STARTUP] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}', flush=True)
print('[STARTUP] Importing transformer_lens...', flush=True)
import transformer_lens; print('[STARTUP] transformer_lens OK', flush=True)
print('[STARTUP] Importing wandb...', flush=True)
import wandb; print('[STARTUP] wandb OK', flush=True)
print('[STARTUP] All imports OK', flush=True)
"
echo "[$(date)] Imports OK, launching training..."
python -u "trainer/$SCRIPT" "$@"
echo "=== Done ==="
