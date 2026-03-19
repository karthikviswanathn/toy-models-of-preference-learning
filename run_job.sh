#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=project_465002390
#SBATCH --job-name=tmpt
#SBATCH --output=/pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning/outputs/logs/%j.out
#SBATCH --error=/pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning/outputs/logs/logs-err/%j.err

set -e

source /project/project_465002390/fair_stuff/simplex-research/.claude/activate_env.sh
cd /pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning

SCRIPT="$1"
shift

echo "=== Running: $SCRIPT $@ ==="
python -u "trainer/$SCRIPT" "$@"
echo "=== Done ==="
