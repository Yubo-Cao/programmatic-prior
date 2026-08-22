#!/bin/bash
#SBATCH --job-name=progattn-discover
#SBATCH --account=tensorlab
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -euo pipefail

# Discovery reads one checkpoint and writes two small JSON files, so it fits inside
# the debug QOS wall limit with room to spare. That matters more than it sounds:
# the gpu partition runs saturated under the normal QOS, where this job's estimated
# start was days out, while debug carries priority 10000 and starts within minutes.
project_root=${PROGRAMMATIC_PRIOR_ROOT:-/resnick/scratch/ycao3/programmatic-prior}
pixi_bin=${PIXI_BIN:-/home/ycao3/.pixi/bin/pixi}
config=${PROGATTN_CONFIG:-configs/pilot_gpt2_small_v2.yaml}
checkpoint=${PROGATTN_CHECKPOINT:-runs/pilot/101/flash_baseline/checkpoints/step_003815.pt}
criterion=${PROGATTN_CRITERION:-enrichment}
output=${PROGATTN_PROGRAMS:-protocol/selected_programs_v2.json}
report=${PROGATTN_DISCOVERY_REPORT:-protocol/program_discovery_report_v2.json}

cd "$project_root"
export PIXI_CACHE_DIR="$project_root/.cache/pixi"
export TORCHINDUCTOR_CACHE_DIR="$project_root/.cache/torchinductor/$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$pixi_bin" run python -m progattn.discover \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --report "$report" \
  --criterion "$criterion"
