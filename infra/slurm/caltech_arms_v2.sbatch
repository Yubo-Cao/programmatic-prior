#!/bin/bash
#SBATCH --job-name=progattn-arms-v2
#SBATCH --account=tensorlab
#SBATCH --partition=gpu
#SBATCH --array=0-4
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

# Five arms this time. wide_window_control trains against the first pilot's
# width-monotone program set at the new prior strength, which is what separates
# "the selection criterion was wrong" from "the prior was too weak to matter" -
# the two faults the first pilot could not tell apart.
conditions=(
  flash_baseline
  flex_noop
  matched_program_prior
  incorrect_program_prior
  wide_window_control
)
condition=${conditions[$SLURM_ARRAY_TASK_ID]}
project_root=${PROGRAMMATIC_PRIOR_ROOT:-/resnick/scratch/ycao3/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/ycao3/.pixi/bin/pixi}
config=${PROGATTN_CONFIG:-configs/pilot_gpt2_small_v2.yaml}
runs=${PROGATTN_RUNS:-runs/pilot_v2/101}

cd "$project_root"
export PIXI_CACHE_DIR="$project_root/.cache/pixi"
export TORCHINDUCTOR_CACHE_DIR="$project_root/.cache/torchinductor/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

set +e
"$pixi_bin" run python -m progattn.train \
  --config "$config" \
  --condition "$condition" \
  --output "$runs/$condition"
status=$?
set -e

if [ "$status" -eq 99 ]; then
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit "$status"
