# Programmatic attention on TinyStories

This repository contains a four-run GPT-2-small pilot on TinyStories. The goal is to test whether a simple positional program, fitted to attention patterns learned by a FlashAttention model, can help a new model train without reducing final language-model quality.

The pilot trains one paired seed for 500 million tokens in each condition:

1. A FlashAttention baseline
2. A FlexAttention no-op control
3. FlexAttention with a Flash-matched program prior
4. FlexAttention with a causally permuted control prior

The Flash run happens first. Its learned query-key attention is measured on a reserved discovery split. A small position-only program is fitted to selected heads, frozen, and then used by the last two conditions. The incorrect control has exactly the same number of preferred edges in every causal row, but those edges point to different token positions.

Each programmed head has a learned nonnegative prior strength. It is parameterized with `softplus` and introduced linearly over the first 10 million training tokens.

## Environment

Pixi is the only supported environment manager.

```bash
pixi install
pixi run test
```

The GPU environment uses PyTorch 2.11 with CUDA 13.0 wheels. The job launcher records the exact Python, PyTorch, CUDA, driver, and GPU versions in every run directory.

## Data

TinyStories is downloaded directly on the machine that performs preprocessing. Dataset binaries and checkpoints are not stored in Git.

```bash
pixi run python scripts/prepare_tinystories.py \
  --output data/tinystories_gpt2
```

The preprocessing step pins the Hugging Face revision, adds the GPT-2 end-of-text token to every story, stores `uint16` token IDs, records story offsets, and writes content hashes.

## Pilot order

Run the Flash reference first:

```bash
pixi run python -m progattn.train \
  --config configs/pilot_gpt2_small.yaml \
  --condition flash_baseline \
  --output runs/pilot/101/flash_baseline
```

Fit and freeze the programs from its final checkpoint:

```bash
pixi run python -m progattn.discover \
  --config configs/pilot_gpt2_small.yaml \
  --checkpoint runs/pilot/101/flash_baseline/checkpoints/last.pt \
  --output protocol/selected_programs.json
```

The remaining three conditions may then run independently. All four conditions load the same saved initialization and batch schedule.

The Slurm launchers in `infra/slurm` use Schmidt scratch for data, checkpoints, and logs. Cluster nodes obtain source code with `git clone` or `git pull`; code is not copied between hosts with SCP or SFTP.

## Scope

This pilot ends after four complete 500-million-token runs and their basic comparison report. It does not claim that a Flash-matched positional program is an objective ground truth. It tests whether that compact approximation is a useful training prior and whether its effect exceeds a density-matched incorrect control.

The longer research plan is preserved in `programmatic_attention_tinystories_experiment.md`.
