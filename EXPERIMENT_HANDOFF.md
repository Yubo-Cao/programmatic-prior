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

## Update 2026-08-21 03:16 EDT

The scheduler state was unchanged from handoff: `97935_[1-2]` still pending, `97936` still blocked on its dependency. Nothing was cancelled or resubmitted, so the array keeps its accrued priority and is rank 2 in the `b300` queue.

### The blocker has a fixed end time

`b300` contains exactly one node, `gb301`, with eight GPUs. Reservation `ssci-yejinc-aug2026` holds it with `SPEC_NODES` from 2026-08-19 17:00 until **2026-08-23 00:00 EDT**, and a separate `eval_server` job occupies all eight GPUs. No scheduling is possible before the reservation ends, so pending with `ReqNodeNotAvail` is expected rather than a fault.

### Moving partitions was evaluated and rejected

| Partition | Free GPUs | Pending jobs |
|---|---:|---:|
| `b300` | 0 of 8 | 4 |
| `b200` | 0 of 120 | 149 |
| `h100` / `h200` | 0 | 5 |
| `a100` | about 13 | 81 |
| `l40s` | about 20 | 10 |

`b200` has fifteen nodes but is fully allocated and far more contended than `b300`. `a100` and `l40s` have capacity but are a different accelerator, which breaks the same-hardware requirement in the completion checklist. The binding constraint is the reservation's end time, not the partition choice.

### Prior-arm slowdown was a real defect, now fixed

The prior arms ran at 9.24 seconds per step against 0.48 for the no-op arm. A local benchmark on the same torch build isolated the cause by comparing FlexAttention `score_mod` variants in one process at identical shapes: with the alpha tensor detached the call took 1.14 to 1.22 milliseconds, and with alpha carrying gradient it took 18.7 to 33.7. The 16 to 29 times gap matches the cluster's 19 times. The program logic is nearly free; the cost is that a grad-carrying captured tensor pushes FlexAttention off its fused backward path. Eager fallback was ruled out because no cluster log contains a `recompile_limit` or unfused warning.

The waste was that every one of the twelve layers built that modifier, while the eight selected heads occupy only five layers (0, 1, 2, 4, 11). The other seven paid full price to add `alpha * 0`. Commit `0584b47` gates the modifier on `applies_prior`, which requires the layer to own at least one selected head.

The change is performance-only and mathematically exact, and it deliberately leaves `requires_grad` untouched so the optimizer parameter set, and therefore resume from the existing step-500 checkpoints, stay valid. Measured 1.46 times end to end locally; extrapolated on the B300 numbers this is 9.25 to about 4.13 seconds per step, so the remaining 3,315 steps take about 3.8 hours instead of 8.5. The queued job picks this up automatically at launch, so the 14-hour allocation was left alone as headroom.

Head-splitting inside a layer, so only the 8 of 144 head-slots that carry a program pay the capture cost, would reach roughly 1 second per step. It was not pursued: the queue wait dominates the timeline, so it would buy little wall clock for a materially riskier refactor.

### Two provenance caveats for the final report

The periodic validation NLL for `flash_baseline` and `flex_noop` is unusable. Both ran on commit `a388803`, before `5df66c9` masked validation padding, and their validation NLL rises from 5.64 to 9.05 and 9.91 while training loss falls to 1.25. The two prior arms ran on `3e25fb7` and their validation curves behave correctly. The final comparison must come from `final_evaluation.json`.

Separately, `flex_noop` recorded `git_dirty: true` in `environment.json`, so its code provenance is less clean than the other three arms. This should be stated in the final report.

### Progress artifacts

A Chinese-language HTML progress report covering all of the above is generated at `reports/progress_zh.html`, which is untracked because `reports/` is ignored.

## Update 2026-08-21 15:00 EDT — migration to Caltech Resnick HPC

The experiment moved to Caltech because the Schmidt reservation blocker did not lift and the L40S measurement made a full four-arm retrain cheaper than waiting.

### Why the move

Job `97935_[1-2]` never left `ReqNodeNotAvail` and the reservation does not release until 2026-08-23 00:00 EDT, which is roughly thirty hours of waiting before a four-hour run can even start.
Preflight job `100501` on Schmidt's `l40s` partition measured `matched_program_prior` at 4.62 to 4.74 seconds per step with the `0584b47` gate, against 9.25 seconds per step on a B300 without it.
That number is what changed the decision: a complete four-arm retrain costs about five hours of wall clock, so it became cheaper to redo all four arms on one accelerator than to finish two arms on another.

Caltech has idle L40S capacity and, critically, zero pending jobs anywhere on the cluster request an L40S; the 219-deep pending queue is dominated by one user requesting 32-node, 512-CPU, 2 TB allocations that block on H200 and H100 resources.
No reservation covers the `hpc-sm-03-*` L40S nodes.
GCP and Modal were therefore never needed and nothing was spent.

### Retraining all four arms is what preserves fairness

Finishing only the two stalled prior arms on Caltech would have left the treatment arms on L40S and the controls on B300, which is exactly the accelerator confound the protocol's same-hardware requirement exists to prevent.
Retraining all four on one GPU model also discharges both provenance caveats recorded above, since every arm then runs on a single commit with a clean working tree and validation padding already masked.

### Every experimental input was verified identical, not assumed

The protocol is designed to be frozen, so the initial state, batch schedule and selected programs are treated as inputs and were reproduced rather than re-derived; re-running discovery would have violated that freeze and changed the experiment.

| Input | Method | Verification |
|---|---|---|
| `data/tinystories_gpt2/train.bin` | regenerated from pinned revision `f54c09fd` | SHA-256 `66c5c49a...7500` matches |
| `data/tinystories_gpt2/validation.bin` | regenerated from pinned revision | SHA-256 `f0d47c00...f5be` matches |
| `protocol/initial_states/seed_101.pt` | regenerated from `seed_everything(101)` | byte-identical, SHA-256 `5ef04ab9...` matches |
| `protocol/batch_schedules/seed_101.npy` | copied over SFTP | SHA-256 `ba0263dd...` matches |
| `protocol/selected_programs.json` | copied over SFTP | SHA-256 `eaaf1915...` matches |
| torch build | `pixi install` from the same lock file | 2.11.0+cu130 on both clusters |

The initial state deserves particular note.
It was regenerated rather than transferred, and the result is byte-identical to Schmidt's file including the zip container, which means `torch.save` is deterministic here and the four Caltech arms provably start from the same weights as the four Schmidt arms.
Tensor content was compared independently of container framing by hashing each tensor's bytes in sorted key order; both clusters report 161 keys and content hash `3e970edc...3fe5`.

### Layout on Caltech

The checkout is at `/resnick/scratch/ycao3/programmatic-prior` on the `gpu` partition under the `tensorlab` account.
`infra/slurm/caltech_arms.sbatch` submits all four conditions as one array, which is possible only because the protocol is frozen: no arm derives anything another arm needs, so none has to wait for `flash_baseline` and discovery the way the original Schmidt pipeline did.
Walltime is eight hours against a projected 5.2, deliberately tight so the job backfills rather than sitting behind the long queue; checkpointing every 500 steps plus `--requeue` makes an overrun recoverable.

### The Schmidt jobs were left queued on purpose

`97935` and `97936` were not cancelled.
They cost nothing while pending, they are a fallback if Caltech fails, and if they do eventually run they yield an independent replication on a different accelerator, which would strengthen rather than confuse the result.
Do not treat the Caltech run as a reason to cancel them.

## Update 2026-08-22 02:19 EDT — the experiment is finished, and the result is a null

All four arms trained to completion on Caltech and the held-out evaluation is done.
Job `1387686` exited `0:0` in 1m36s and its log ends with `REPORT PIPELINE DONE`.
Every item in the completion checklist now passes.

### The result

Recomputed from the four final checkpoints on the held-out `final_evaluation` partition, all four arms scoring the identical 1,054,031 tokens on the same device with the same code:

| rank | condition | held-out NLL | perplexity | Δ vs best | alpha mean |
|---|---|---:|---:|---:|---:|
| 1 | `flex_noop` | 1.2939937 | 3.6473239 | — | — |
| 2 | `flash_baseline` | 1.2943585 | 3.6486548 | +0.0003648 | — |
| 3 | `incorrect_program_prior` | 1.2943903 | 3.6487706 | +0.0003966 | 0.0780 |
| 4 | `matched_program_prior` | 1.2946841 | 3.6498429 | +0.0006904 | 0.1338 |

The matched prior did not help; it ranked last of four, and the entire spread across all arms is 0.00069 nats, a 0.069% difference in perplexity.
The direction is what matters most here: because the treatment arm is worse than both controls, the hypothesis fails regardless of whether the difference is statistically distinguishable from zero.

The one genuinely informative positive finding is that the model *used* the prior.
The matched arm's learned alpha averages 0.1338 against the incorrect arm's 0.0780, a factor of 1.72, so the model can tell a matched program set from a mismatched one and weights it accordingly.
That rules out the trivial explanation that the prior was simply ignored, and pushes the conclusion toward the stronger claim that this inductive bias has no value for this task.

The cost was real: the prior arms ran at 4.62 s/step against the no-op arm's 0.98 s/step, 4.7x slower, consuming about 7.8 of the experiment's roughly 11.9 GPU-hours to produce a negative return.

### Scope, and the limitation that matters

This holds for GPT-2 small on TinyStories at 500M tokens with a single seed (101) and a program set that is entirely `LOCAL_WINDOW(64)`.
It does not generalize to "programmatic attention priors do not work"; it says this particular family of local-window priors does not work in this setting.
The single seed is the important gap, since with one run there is no way to separate a true null from seed noise swamping a small effect, and closing it is the first thing the next session should do.
A second, cheaper gap: `evaluate_runs.py` stores only aggregate NLL, so no confidence interval can be computed after the fact; saving per-sequence losses would allow a paired bootstrap that exploits how correlated the four arms are.

### An environment failure worth knowing about

The first evaluation attempt, job `1379566`, died three seconds in with `ModuleNotFoundError: No module named 'typing_extensions'`, which torch imports at `__init__.py` line 34.
At 17:02 that day something re-solved the pixi environment — the pytest stack appears with that timestamp — and a PyPI resolver clobbered the conda-provided `typing_extensions`, leaving a dist-info directory with an empty `RECORD` and no module file.
The training runs were untouched because all four finished at 13:00, 13:02, 16:51 and 16:52, before the corruption, and a sweep of every `dist-info/RECORD` confirmed `typing_extensions` was the only damaged package.

The repair deliberately avoided re-solving, since a re-solve could move torch and break consistency with the completed training.
`pixi install --locked` confirmed the lock and manifest agree but skipped the missing file, because pixi's own `conda-meta` record claimed the package was installed; moving that stale record and the corrupt dist-info aside and reinstalling restored the exact locked build `4.16.0-pyhcf101f3_0` with torch still at 2.11.0+cu130.

Two lessons for the next session.
Getting scheduled on the saturated `gpu` partition required the `debug` QOS, which carries Priority=10000 against `normal`'s 0 in exchange for a 30-minute wall limit — with `normal` the estimated start had degraded to 2026-08-30, and under `debug` the job started within minutes.
And the remote checkout had drifted three commits behind and did not contain `caltech_report.sbatch` at all; the earlier job only ran because submission injects the script body directly, so verify the remote HEAD before assuming a committed script exists there.

### Schmidt

The user decided on 2026-08-22 that no B300 replication is needed and that the Caltech results stand as the result of record, so `97935_[1-2]` and `97936` were cancelled at 12:43 EDT that day.
Neither had ever been allocated a node, so `sacct` reports both as `CANCELLED` with `0.00` GPU-hours and nothing was spent on them.
The unrelated job `100158` (`refract-response-metrics`, a different project) was left alone and is still pending in `b300`.
The two-hourly monitoring cron was removed at the same time, since the experiment is finished and the Schmidt queue is now empty.

## Update 2026-08-23 — the second pilot reverses the null, and the prior is now nearly free

The v1 null reported above was a null about a *particular* configuration, not about the idea. Two of its inputs were defective, and fixing both flipped the sign of the result.

The first defect was the prior strength. v1 initialised `alpha` at 0.13, which is small enough that the learned bonus never moved the set-versus-complement odds by much; v2 initialises it at 4.0. The second was the program set. v1 selected programs by weighted IOU against the observed attention, which rewards programs that match a head's *shape* even when that shape is what a uniform reader would produce anyway; v2 ranks candidates by enrichment of preferred-edge mass over a uniform reader, which is what actually identifies a head doing something a program can express. The v1 set collapsed to `LOCAL_WINDOW(64)` on every head as a direct consequence.

### The result

Five arms, seed 101, 500M tokens, all trained on one Schmidt L40S and all scored on the identical 1,054,031 held-out tokens on that same device (job `104255` for training, `104805` for evaluation, artifacts in `reports/pilot_v2/101/`):

| rank | condition | held-out NLL | perplexity | Δ vs flash | tokens/s | hours |
|---|---|---:|---:|---:|---:|---:|
| 1 | `matched_program_prior` | 1.284155 | 3.612 | −0.009406 | 24415.2 | 5.694 |
| 2 | `wide_window_control` | 1.289365 | 3.630 | −0.004196 | 28277.6 | 4.913 |
| 3 | `flex_noop` | 1.292766 | 3.643 | −0.000795 | 133857.4 | 1.039 |
| 4 | `flash_baseline` | 1.293561 | 3.646 | — | 126661.8 | 1.096 |
| 5 | `incorrect_program_prior` | 1.294342 | 3.649 | +0.000781 | 24371.4 | 5.702 |

The scale to read this against is the `flash_baseline`-to-`flex_noop` gap of 0.00080 nats. Those two arms are mathematically the same computation on two kernels, so that gap is this experiment's built-in noise floor, measured rather than assumed. Against it the matched prior is worth 11.8 noise units and the wide-window control 5.3, while the incorrect prior lands 1.0 unit on the *wrong* side — indistinguishable from carrying no prior at all, which is exactly what a working negative control should do.

The gain decomposes. `wide_window_control` runs the v1 program set at the v2 prior strength, so the distance between it and `flash_baseline` isolates the strength change, and it accounts for roughly 40 to 45 percent of the total. The enrichment-based program set supplies the rest. Neither fix alone would have produced this; v1 had the right idea with the wrong strength on the wrong programs.

The learned alphas agree with the ranking. The matched arm settles at a mean of 4.40 across its eight prior heads against the incorrect arm's 3.07, so the model still distinguishes a matched program set from a rotated one and weights it accordingly — the same qualitative signal v1 showed, now attached to an actual improvement.

### The 5.5x slowdown was a kernel artifact, and it is fixed

The prior arms ran at 24.4k tokens/s against `flex_noop`'s 133.9k, and that cost dominated the experiment: 11.4 of its 18.4 GPU-hours bought the two prior arms.

None of it was inherent to the prior. The FlexAttention `score_mod` closes over `self.alpha()`, a grad-carrying tensor, so FlexAttention has to differentiate through a captured value and reduce that gradient across the entire score matrix. Measured on an L40S at B=8 H=12 T=512 D=64, forward plus backward: 0.337 ms with no prior, 21.920 ms with it. That predicts 4,210 ms/step of prior overhead at the training batch size against 4,389 ms measured, which is within four percent and confirms the diagnosis rather than merely being consistent with it.

`src/progattn/triton_attn.py` folds the bonus into a Triton flash-attention kernel. Because beta is constant on the preferred set, its gradient is just the sum of the score gradients there, which accumulates in registers and commits with one atomic per block instead of one per element. That runs in 0.282 ms, 77.7x faster than the `score_mod` path. At B=8 it also beat FlexAttention carrying no prior at all, but that does not survive the move to the training batch size: at B=32 the kernel is 0.693 ms against FlexAttention's 0.551, and the section below measures what that costs in practice.

The kernel is exact, not approximate. Against a dense float32 reference with IEEE fp32 dots it agrees to 5e-7 relative on the output and on every gradient including alpha, and the alpha gradient separately matches a float64 central finite difference to 2.7e-8. `tests/test_triton_attention.py` additionally asserts that the full model produces identical logits, loss, and parameter gradients on both kernels for the matched, incorrect, and no-op conditions.

Two traps are worth recording for anyone editing that file. Triton's real type system makes runtime float arguments, literals, and `tl.where` results all fp32, but Dynamo's *mock* kernel trace under `torch.compile` types a Python float argument as fp64, which surfaces as `Loop-carried variable dk has initial type fp32 but is re-assigned to fp64`. The fix is to keep the scale off the loop-carried accumulators entirely rather than to cast it — `sm_scale.to(tl.float32)` fails under the same mock, because there the argument really is a bare Python float. And `TRITON_F32_DEFAULT=ieee` is what proves exactness; without it TF32 puts a uniform 1e-3 floor under every comparison and a correct kernel looks broken.

`configs/pilot_gpt2_small_v3.yaml` is byte-identical to v2 except for `prior.kernel: triton`, so v2 and v3 are the same experiment on two kernels and their losses are directly comparable.

### What the kernel actually costs, measured end to end

The microbenchmark understates the picture, so here is the same comparison from real training steps on the same L40S, seed 101, `matched_program_prior` unless noted:

| config | condition | s/step | tokens/s |
|---|---|---:|---:|
| v2 (FlexAttention) | `matched_program_prior` | 5.361 | 24415 |
| v2 (FlexAttention) | `flex_noop` | 0.979 | 133857 |
| v3 (Triton) | `matched_program_prior` | 1.375 | 95098 |
| v3 (Triton) | `incorrect_program_prior` | 1.398 | — |
| v3 (Triton) | `flex_noop` | 1.368 | — |

The `flex_noop` row under v3 is the load-bearing one. That condition runs the same Triton kernel with `APPLY_PRIOR=False`, so the distance between it and the two prior arms is the entire cost of carrying the prior: about 0.03 s/step, or two percent. Against v2's 4.38 s/step of prior overhead that is a factor of roughly 150, and it is the number the whole exercise was aimed at — the prior is now effectively free.

The remaining 0.39 s/step between v3 `flex_noop` and v2 `flex_noop` is not the prior at all. It is the cost of running my kernel instead of FlexAttention on a computation neither one is doing anything special with, and it applies to all twelve layers whether or not they carry a program. Net of both effects the prior arms run 3.90x faster than v2.

That 0.39 s/step is an open question worth someone's time. The microbenchmark at the exact training shape (B=32 H=12 T=512 D=64) puts the Triton kernel at 0.693 ms against FlexAttention's 0.551, which over the 96 attention calls in an optimizer step predicts a 13 ms gap, not 390 ms. The two measurements disagree by a factor of 29, so at least one of them is not measuring what it appears to. Resolving that is the first thing to look at before trusting either number, and it is worth resolving: closing the gap entirely would put the prior arms at about 1.0 s/step, which is parity with an unmodified no-prior baseline.

A launch-geometry sweep at the training shape (BLOCK_M/BLOCK_N in 64/128, num_warps in 4/8, num_stages in 2/3/4) found the existing defaults — 64x64, 4 warps, 3 stages — already optimal; 8 warps is 1.6x worse. `_blocks` clamps the block size to 64, so the 128 and 256 rows of that sweep were silently no-ops and remain untested. `PROGATTN_BLOCK`, `PROGATTN_BLOCK_N`, `PROGATTN_WARPS` and `PROGATTN_STAGES` are left in place to re-run the sweep on a different accelerator.

### The v3 retrain was deliberately not run

`configs/pilot_gpt2_small_v3.yaml` is validated but no full v3 run exists. On a 12-step comparison against v2 from the identical frozen initial state, the two kernels agree to 6.5e-5 in loss and 2.1e-3 in gradient norm, drifting slowly in the way two exact implementations with different summation order do, so a full retrain was judged unlikely to move any conclusion and the user chose not to spend the roughly seven GPU-hours. The v2 numbers in the table above remain the result of record.

The more valuable use of the speedup, whenever someone returns to this, is still additional seeds. The v2 effect is 0.0094 nats from a single seed, and no amount of kernel work substitutes for knowing whether that survives a second one.
