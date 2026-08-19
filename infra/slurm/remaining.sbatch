#!/bin/bash
#SBATCH --job-name=progattn-remaining
#SBATCH --account=ssci-anima
#SBATCH --partition=b300
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail

conditions=(
  flex_noop
  matched_program_prior
  incorrect_program_prior
)
condition=${conditions[$SLURM_ARRAY_TASK_ID]}
project_root=${PROGRAMMATIC_PRIOR_ROOT:-/weka/scratch/schmidt/ssci-anima/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/schmidt/ssci-ycao3/.pixi/bin/pixi}

cd "$project_root"
export PIXI_CACHE_DIR="$project_root/.cache/pixi"
export TORCHINDUCTOR_CACHE_DIR="$project_root/.cache/torchinductor/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

set +e
"$pixi_bin" run python -m progattn.train \
  --config configs/pilot_gpt2_small.yaml \
  --condition "$condition" \
  --output "runs/pilot/101/$condition"
status=$?
set -e

if [ "$status" -eq 99 ]; then
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit "$status"
