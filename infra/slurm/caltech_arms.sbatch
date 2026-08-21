#!/bin/bash
#SBATCH --job-name=progattn-arms
#SBATCH --account=tensorlab
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail

# All four arms run concurrently because the protocol is frozen: the initial
# state, the batch schedule and the selected programs are inputs staged ahead of
# submission, so no arm has to derive anything another arm needs.
conditions=(
  flash_baseline
  flex_noop
  matched_program_prior
  incorrect_program_prior
)
condition=${conditions[$SLURM_ARRAY_TASK_ID]}
project_root=${PROGRAMMATIC_PRIOR_ROOT:-/resnick/scratch/ycao3/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/ycao3/.pixi/bin/pixi}

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
