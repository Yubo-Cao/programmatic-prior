#!/bin/bash
#SBATCH --job-name=progattn-report
#SBATCH --account=tensorlab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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

# Inference only, so the footprint is kept small deliberately: the gpu partition
# runs saturated and a 8-CPU/32G/1.5h job backfills into gaps a larger one misses.
# Scoring all four checkpoints inside one job is also what keeps the comparison
# fair, since every arm is then evaluated on the same device by the same code.
config=${PROGATTN_CONFIG:-configs/pilot_gpt2_small.yaml}
runs=${PROGATTN_RUNS:-runs/pilot/101}
report_dir=${PROGATTN_REPORT_DIR:-reports/pilot/101}

"$pixi_bin" run python scripts/evaluate_runs.py \
  --config "$config" \
  --runs "$runs" \
  --partition final_evaluation
"$pixi_bin" run python scripts/make_report.py \
  --runs "$runs" \
  --output "$report_dir"
