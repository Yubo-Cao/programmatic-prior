#!/bin/bash
#SBATCH --job-name=progattn-report
#SBATCH --account=ssci-anima
#SBATCH --partition=b300
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -euo pipefail

project_root=${PROGRAMMATIC_PRIOR_ROOT:-/weka/scratch/schmidt/ssci-anima/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/schmidt/ssci-ycao3/.pixi/bin/pixi}

cd "$project_root"
export PIXI_CACHE_DIR="$project_root/.cache/pixi"
export TORCHINDUCTOR_CACHE_DIR="$project_root/.cache/torchinductor/$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$pixi_bin" run python scripts/evaluate_runs.py \
  --config configs/pilot_gpt2_small.yaml \
  --runs runs/pilot/101 \
  --partition final_evaluation
"$pixi_bin" run python scripts/make_report.py \
  --runs runs/pilot/101 \
  --output reports/pilot/101
