#!/bin/bash
#SBATCH --job-name=progattn-report
#SBATCH --account=ssci-anima
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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

# Scoring every arm inside one job on one device is what keeps the comparison
# fair, and it has to be the same accelerator the arms trained on so the held-out
# numbers stay comparable with the training curves.
config=${PROGATTN_CONFIG:-configs/pilot_gpt2_small_v2.yaml}
runs=${PROGATTN_RUNS:-runs/pilot_v2/101}
report_dir=${PROGATTN_REPORT_DIR:-reports/pilot_v2/101}

"$pixi_bin" run python scripts/evaluate_runs.py \
  --config "$config" \
  --runs "$runs" \
  --partition final_evaluation
"$pixi_bin" run python scripts/make_report.py \
  --runs "$runs" \
  --output "$report_dir"
