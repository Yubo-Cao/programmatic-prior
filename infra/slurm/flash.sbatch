#!/bin/bash
#SBATCH --job-name=progattn-flash
#SBATCH --account=ssci-anima
#SBATCH --partition=b300
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
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

set +e
srun "$pixi_bin" run python -m progattn.train \
  --config configs/pilot_gpt2_small.yaml \
  --condition flash_baseline \
  --output runs/pilot/101/flash_baseline
status=$?
set -e

if [ "$status" -eq 99 ]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
