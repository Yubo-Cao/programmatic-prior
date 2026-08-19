#!/bin/bash
#SBATCH --job-name=progattn-data
#SBATCH --account=ssci-anima
#SBATCH --partition=b200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -euo pipefail

project_root=${PROGRAMMATIC_PRIOR_ROOT:-/weka/scratch/schmidt/ssci-anima/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/schmidt/ssci-ycao3/.pixi/bin/pixi}

cd "$project_root"
export HF_HOME="$project_root/.cache/huggingface"
export PIXI_CACHE_DIR="$project_root/.cache/pixi"
export PYTHONUNBUFFERED=1

"$pixi_bin" install --locked
"$pixi_bin" run python scripts/prepare_tinystories.py \
  --output data/tinystories_gpt2 \
  --tokenizer-threads "$SLURM_CPUS_PER_TASK"
