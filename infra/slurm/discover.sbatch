#!/bin/bash
#SBATCH --job-name=progattn-discover
#SBATCH --account=ssci-anima
#SBATCH --partition=b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -euo pipefail

project_root=${PROGRAMMATIC_PRIOR_ROOT:-/weka/scratch/schmidt/ssci-anima/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/schmidt/ssci-ycao3/.pixi/bin/pixi}

cd "$project_root"
export PIXI_CACHE_DIR="$project_root/.cache/pixi"
export PYTHONUNBUFFERED=1

srun "$pixi_bin" run python -m progattn.discover \
  --config configs/pilot_gpt2_small.yaml \
  --checkpoint runs/pilot/101/flash_baseline/checkpoints/last.pt \
  --output protocol/selected_programs.json \
  --report protocol/program_discovery_report.json
