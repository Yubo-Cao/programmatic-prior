# Experiment handoff

Status captured at 2026-08-21 00:12 PDT.

Active monitoring by the original operator ended on 2026-08-21 at the user's request. The submitted Slurm jobs were left untouched. Slurm can still start the training jobs and their dependent report automatically when the reserved B300 node becomes available.

## Repository and compute location

- Public repository: <https://github.com/Yubo-Cao/programmatic-prior>
- Handoff commit parent: `14b5a047ed5eb59f72e46e61e565e74f325cabbb`
- Cluster: Schmidt Sciences Skipjack
- Partition and accelerator: `b300`, one NVIDIA B300 per task
- Remote checkout: `/weka/scratch/schmidt/ssci-anima/programmatic-prior`
- Data and run storage: Schmidt scratch under the remote checkout
- Environment manager: Pixi
- Python gates: Ruff formatting, Ruff lint, strict ty, and pytest

Schmidt scratch was selected because it had several terabytes free and comfortably fit the tokenized dataset, four checkpoints, logs, and reports. Caltech was not reachable when the placement decision was made. Source code is synchronized through Git only. No SSH copy, SCP, or SFTP workflow is used for code.

## Pilot question

The pilot asks whether a compact positional program fitted to attention learned by a FlashAttention GPT-2-small model is a useful training prior. FlashAttention is the neural reference. There is no fifth standard-attention arm.

The four paired conditions are:

1. `flash_baseline`: causal FlashAttention baseline and neural reference.
2. `flex_noop`: causal FlexAttention with a no-op score modifier.
3. `matched_program_prior`: causal FlexAttention plus a frozen positional program fitted to query-key attention from the completed Flash model.
4. `incorrect_program_prior`: causal FlexAttention plus an incorrect control program. Its preferred edges are causally rotated, preserving the exact number of preferred zero and one entries in every causal row.

The program prior is added to the attention score. Each selected head has a learned nonnegative strength parameterized with `softplus`. Its contribution is multiplied by a linear warmup scale over the first 10 million training tokens.

## Shared training protocol

- Model: GPT-2 small, 12 layers, 12 heads, 768 hidden dimensions
- Parameter count: 124,046,736
- Context length: 512 tokens
- Vocabulary: GPT-2, 50,257 tokens
- Requested training budget per arm: 500,000,000 tokens
- Implemented budget per arm: 3,815 steps at 131,072 tokens per step
- Actual tokens per completed arm: 500,039,680
- Precision: bfloat16
- Microbatch: 32 sequences
- Gradient accumulation: 8
- Optimizer: AdamW
- Learning-rate schedule: 10 million token warmup, then cosine decay
- Seed: 101

All arms load the same saved initial model state and the same precomputed batch schedule. The dataset and large runtime artifacts are intentionally excluded from Git.

## TinyStories data

The data preparation job completed successfully.

- Hugging Face dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Training stories: 2,119,719
- Training tokens: 473,992,236
- Validation stories: 21,990
- Validation tokens: 4,765,918
- Training binary: 947,984,472 bytes
- Validation binary: 9,531,836 bytes

Training samples from the fixed schedule with replacement, so a 500 million token training budget is valid even though the stored training corpus is slightly smaller. The preprocessing metadata records complete content hashes.

## Program discovery result

Discovery ran against the completed Flash checkpoint and selected eight heads. Every selected candidate was a local window of width 64. Weighted IoU scores ranged from about 0.389 to 0.565.

| Layer | Head | Program |
|---:|---:|---|
| 0 | 4 | `LOCAL_WINDOW(64)` |
| 0 | 6 | `LOCAL_WINDOW(64)` |
| 1 | 4 | `LOCAL_WINDOW(64)` |
| 2 | 2 | `LOCAL_WINDOW(64)` |
| 2 | 3 | `LOCAL_WINDOW(64)` |
| 4 | 0 | `LOCAL_WINDOW(64)` |
| 4 | 3 | `LOCAL_WINDOW(64)` |
| 11 | 3 | `LOCAL_WINDOW(64)` |

The exact per-head scores, source checkpoint hash, and fitted-program metadata are in `protocol/selected_programs.json` on Schmidt.

## Current run status

The authoritative Slurm snapshot at handoff was:

| Work | Job | State | Evidence |
|---|---|---|---|
| TinyStories preparation | `96695` | Complete | Exit `0:0`, 9 minutes 35 seconds |
| First Flash attempt | `96696` | Failed, superseded | One-second launcher metadata conflict |
| Flash baseline | `97355` | Complete | Exit `0:0`, 37 minutes 8 seconds |
| Program discovery | `97356` | Complete | Exit `0:0`, 4 minutes 19 seconds |
| FlexAttention no-op | `97357_0` | Complete | Exit `0:0`, 36 minutes 40 seconds |
| Matched prior first attempt | `97357_1` | Timed out, resumable | Reached step 719; checkpoint at step 500 |
| Incorrect prior first attempt | `97357_2` | Timed out, resumable | Reached step 718; checkpoint at step 500 |
| Prior-arm continuation array | `97935_[1-2]` | Pending | Scheduler reason: B300 node reserved or unavailable |
| Final evaluation and report | `97936` | Pending | Dependency on successful completion of `97935` |

The latest queue query still showed the prior-arm array pending because the sole B300 node was reserved for another user. The observed reservation ended at 2026-08-23 00:00 Eastern, which is 2026-08-22 21:00 Pacific. The scheduler may revise start timing.

Accounting for the listed preparation, training, discovery, and failed attempts had reached about 5.464 GPU-hours at handoff. Pending jobs had consumed no GPU time.

## Completed-run evidence

The Flash baseline finished all 3,815 steps and 500,039,680 tokens. Its recorded training time was 1,821.47 seconds. Training loss fell from about 10.87 to about 1.25, with a final gradient norm near 0.27. Its final checkpoint was about 1.49 GB.

The FlexAttention no-op job also completed with exit code zero. Its output directory should be audited together with the other arms before making a final scientific comparison.

The two prior arms were numerically healthy before timing out. Their last uncheckpointed records were:

| Arm | Last logged step | Tokens logged | Loss | Gradient norm | Learned alpha range |
|---|---:|---:|---:|---:|---:|
| Matched prior | 719 | 94,240,768 | 1.7414 | 0.4089 | 0.1077 to 0.1264 |
| Incorrect prior | 718 | 94,109,696 | 1.7600 | 0.4378 | 0.0838 to 0.0947 |

These rows were produced after the durable step-500 checkpoints. They are not valid resume points.

## Failures and repairs

### Initial Flash launcher failure

Job `96696` failed because `srun` saw conflicting CPU metadata. The Slurm requests were normalized to 15 CPUs per GPU and unnecessary `srun` wrappers were removed. The replacement Flash job completed.

### Validation padding bug

The original periodic validation included repeated padding targets, so those validation NLL values are not comparable and must not be used. Commit `5df66c9` masks padding targets with `-100`. Commit `3e25fb7` adds a separate held-out final evaluation that loads every final checkpoint and writes `final_evaluation.json` before producing the report.

### Prior-arm two-hour timeout

The prior score modifier is much slower than the no-op FlexAttention kernel. The two-hour allocation was too short. The models did not diverge.

Commit `14b5a04` made two resume repairs:

- The continuation allocation is 14 hours, within the partition's three-day maximum.
- On resume, `metrics.jsonl` is atomically truncated to the checkpointed step. This removes steps 501 through 719 or 718 from the first attempts before replaying them. Accumulated training time is restored from the retained metrics.

Both continuation tasks should automatically load `checkpoints/last.pt`, resume at step 500, and train through step 3,815.

## Important paths

All paths below are relative to the remote checkout unless noted otherwise.

- Configuration: `configs/pilot_gpt2_small.yaml`
- Slurm scripts: `infra/slurm/`
- Initial state: `protocol/initial_states/seed_101.pt`
- Batch schedule: `protocol/batch_schedules/seed_101.npy`
- Selected programs: `protocol/selected_programs.json`
- Dataset: `data/tinystories_gpt2/`
- Run roots: `runs/pilot/101/<condition>/`
- Final report destination: `reports/pilot/101/`
- Public source repository: <https://github.com/Yubo-Cao/programmatic-prior>

Each run directory should contain `run.json`, `environment.json`, `metrics.jsonl`, `checkpoints/last.pt`, and eventually `completed.json` and `final_evaluation.json`.

## How to take over

Use the authenticated Remote Hosts MCP for scheduler operations. Do not create a manual SSH polling loop. Query `remote_status` first, then use:

- `slurm_queue` for jobs `97935` and `97936`, grouped by array.
- `slurm_accounting` for exact tasks `97935_1`, `97935_2`, and report job `97936`.
- `slurm_wait` for a native one-hour wait. A timeout with `complete: false` is expected and can be called again.
- `slurm_log` only after a task starts or fails.

Do not treat an empty queue as evidence of completion. Confirm terminal state and exit code through accounting. Do not retry a failed submission until its dispatch state is inspected.

If a code repair is needed:

1. Reproduce or diagnose it locally.
2. Edit with the normal repository workflow.
3. Run `/home/yubo/.agents/skills/resource-guard/scripts/run.sh -- pixi run check`.
4. Make a short atomic Conventional Commit without a co-author trailer.
5. Push to `origin/main`.
6. Pull with `git pull --ff-only origin main` in the Schmidt checkout.
7. Validate the revised sbatch script before resubmission.

Do not transfer source code with SCP or SFTP.

## Completion checklist

The experiment is not complete at handoff. A final owner should verify every item below before making a claim:

- Both `97935_1` and `97935_2` finish with exit code zero.
- All four `completed.json` files report 3,815 steps and 500,039,680 tokens.
- Every `metrics.jsonl` has exactly one training record for each step from 1 through 3,815.
- All four final checkpoints exist and have recorded SHA-256 hashes.
- Every run records the same initial-state path and batch-schedule path.
- Flash and no-op alpha arrays are empty.
- Matched and incorrect prior alpha arrays each contain eight finite, nonnegative values.
- All run environments record the same B300 accelerator and compatible PyTorch and CUDA versions.
- Report job `97936` finishes with exit code zero.
- Every `final_evaluation.json` uses the `final_evaluation` partition, the same token count, and a checkpoint token count of 500,039,680.
- `reports/pilot/101/report.md` and `summary.json` exist and use held-out final metrics, not the invalid early calibration metrics.
- A concise tracked results summary is added to Git and pushed without committing datasets or checkpoints.

Until those checks pass, the pilot supports only an implementation and progress report, not a scientific conclusion.
