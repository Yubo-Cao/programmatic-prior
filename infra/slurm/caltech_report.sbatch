#!/bin/bash
#SBATCH --job-name=progattn-report
#SBATCH --account=tensorlab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -euo pipefail

project_root=${PROGRAMMATIC_PRIOR_ROOT:-/resnick/scratch/ycao3/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/ycao3/.pixi/bin/pixi}

cd "$project_root"
export PIXI_CACHE_DIR="$project_root/.cache/pixi"
export TORCHINDUCTOR_CACHE_DIR="$project_root/.cache/torchinductor/$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# The held-out partition is disjoint from the one the training loop reports on,
# so this is the only number the final comparison may be drawn from.
"$pixi_bin" run python scripts/evaluate_runs.py \
  --config configs/pilot_gpt2_small.yaml \
  --runs runs/pilot/101 \
  --partition final_evaluation
"$pixi_bin" run python scripts/make_report.py \
  --runs runs/pilot/101 \
  --output reports/pilot/101
