# Programmatic Attention on TinyStories
## Claude Code / Cloud Coding 可执行实验方案

### 文档目的

在一个从零训练的 GPT-2-small 上，测试以下三个相互独立的问题：

1. **优化问题**：给部分 attention heads 加入由可读程序定义的 soft prior，是否能减少达到同一 validation loss 所需的训练 tokens？
2. **系统问题**：将 prior 融入 FlexAttention/FlashAttention 风格的 GPU kernel 后，是否能改善真正的 wall-clock time-to-quality，而不是只减少 optimizer steps？
3. **部署问题**：训练结束后，能否把这些 heads 的 neural QK attention 精确替换成程序，而只造成很小的 validation NLL 增长？

本项目必须先做低成本的 correctness 和 pilot；只有 pilot 显示出可重复信号，才启动多 seed 的 GPT-2-small 主实验。第一版不要手写 Triton backward kernel。先用 PyTorch FlexAttention 验证科学假设；只有 soft-prior 实验成功后，再进入 hard program head 和 Triton/CuTeDSL 阶段。

---

## 1. 最终研究问题

正式研究问题写成：

> Can high-fidelity, deployable programmatic attention patterns extracted from trained GPT-2 heads serve as useful priors when training a fresh GPT-2-small model on TinyStories, improving wall-clock time-to-quality while preserving final language-model fit and enabling exact hard replacement?

核心 intervention 为：

```latex
S_{\theta,l,h,i,j}
=
\frac{Q_{l,h,i}K_{l,h,j}^{\top}}{\sqrt{d_h}},
```

```latex
A_{\theta,\pi}
=
\operatorname{softmax}
\left(
S_\theta
+
M_{\mathrm{causal}}
+
\beta_{l,h}\,
\mathbf 1[(i,j)\in E_{\pi_{l,h}}]
\right).
```

其中：

- \(E_{\pi_{l,h}}\) 是程序允许或偏好的 attention edges；
- \(\beta_{l,h}=0\) 表示普通 head；
- \(\beta_{l,h}>0\) 表示 soft programmatic prior；
- QK 路径仍然可微；
- prior 必须由 GPU kernel 内的整数 index comparison 生成，不能 materialize 一个完整的 \(T\times T\) bias tensor。

训练后的 hard replacement 为：

```latex
A_{l,h}
\leftarrow
P_{\pi_{l,h}},
\qquad
O_{l,h}=P_{\pi_{l,h}}V_{l,h}.
```

---

## 2. 项目范围与非目标

### 第一阶段必须完成

- 从零训练 GPT-2-style causal LM；
- TinyStories V1 + GPT-2 tokenizer；
- standard FlashAttention baseline；
- FlexAttention no-op implementation control；
- teacher/public-library matched program prior；
- matched shuffled-program control；
- paired seeds；
- validation NLL、perplexity、tokens-to-target、wall-clock-to-target；
- attention/program alignment；
- exact hard-replacement evaluation；
- Modal 与 generic Slurm launcher；
- 可恢复 checkpoint；
- 一键生成分析报告。

### 第一阶段明确不做

- 不训练 1B+ 模型；
- 不做多机 DDP；
- 不把 8 张 B200/B300 用来并行训练一个 124M 模型；
- 不首先写 Triton backward kernel；
- 不把 LLM-as-judge 作为主要指标；
- 不重新运行完整的 LLM program synthesis agent；
- 不在看到主实验结果后继续调整 beta、head 数量、target loss 或 fake control。

---

## 3. 数据集

使用：

```text
Hugging Face dataset: roneneldan/TinyStories
Variant: original V1
Tokenizer: GPT-2 BPE, vocabulary size 50,257
Context length: 512
Story separator: GPT-2 <|endoftext|>, token id 50256
```

不要下载整个包含 V1、V2 和 archive 的仓库。只使用原始 V1 train/validation，或者使用 Hugging Face `datasets` 的 row-based V1 split。

### 3.1 预处理产物

```text
data/tinystories_gpt2/
├── train.bin                 # uint16, contiguous token IDs
├── validation.bin            # uint16
├── train_story_offsets.npy   # int64
├── val_story_offsets.npy     # int64
├── val_partitions.json
└── metadata.json
```

`metadata.json` 至少记录：

```json
{
  "dataset_id": "roneneldan/TinyStories",
  "dataset_revision": "<pinned revision>",
  "tokenizer": "gpt2",
  "eot_token_id": 50256,
  "train_tokens": 0,
  "validation_tokens": 0,
  "train_stories": 0,
  "validation_stories": 0,
  "sha256_train_bin": "",
  "sha256_validation_bin": "",
  "created_at_utc": ""
}
```

### 3.2 Validation 划分

按 story ID 的稳定 hash 划分，而不是按 token 连续切片：

- 25%：`program_discovery`
- 25%：`protocol_calibration`
- 50%：`final_evaluation`

任何 program/head 选择不得查看 `final_evaluation`。

### 3.3 预处理命令

```bash
uv sync
python scripts/prepare_tinystories.py \
  --dataset roneneldan/TinyStories \
  --variant v1 \
  --tokenizer gpt2 \
  --output data/tinystories_gpt2 \
  --workers 16
```

预处理程序必须：

- 在每个 story 结尾添加 EOT；
- 使用 `np.uint16`；
- 保存 story offsets；
- 输出精确 token 数；
- 固定 dataset revision；
- 支持 resume；
- 最后重新读取 memmap 并验证 token 范围在 `[0, 50256]`。

---

## 4. 模型

### 4.1 GPT-2-small 主模型

```yaml
model:
  vocab_size: 50257
  block_size: 512
  n_layer: 12
  n_head: 12
  n_embd: 768
  mlp_ratio: 4
  bias: true
  dropout: 0.0
  tie_embeddings: true
  position_embedding: learned_absolute
  norm: pre_layernorm
```

预计约 124M parameters。不要加载 GPT-2 pretrained weights；所有主训练 run 从随机初始化开始。

### 4.2 GPT-2-mini pilot

```yaml
model:
  vocab_size: 50257
  block_size: 512
  n_layer: 6
  n_head: 6
  n_embd: 384
  mlp_ratio: 4
  dropout: 0.0
```

该模型只用于：

- 验证训练稳定性；
- 选择 beta；
- 检查 FlexAttention overhead；
- 估算硬件吞吐量；
- 不用于最终科学结论。

---

## 5. Program DSL

第一版只支持能直接编译为 GPU integer comparisons 的 position-only programs。

```python
from enum import IntEnum


class ProgramType(IntEnum):
    NONE = 0
    FIRST_TOKEN = 1
    SELF = 2
    PREVIOUS_K = 3
    LOCAL_WINDOW = 4
```

语义：

```python
def preferred_edge(
    program_type: ProgramType,
    parameter: int,
    query_index: int,
    key_index: int,
) -> bool:
    if program_type == ProgramType.NONE:
        return False

    if program_type == ProgramType.FIRST_TOKEN:
        return key_index == 0

    if program_type == ProgramType.SELF:
        return key_index == query_index

    if program_type == ProgramType.PREVIOUS_K:
        source = max(0, query_index - parameter)
        return key_index == source

    if program_type == ProgramType.LOCAL_WINDOW:
        left = max(0, query_index - parameter + 1)
        return left <= key_index <= query_index

    raise ValueError(program_type)
```

Hard program matrix 必须：

- causal；
- 每行和为 1；
- `FIRST_TOKEN`、`SELF`、`PREVIOUS_K` 为 one-hot；
- `LOCAL_WINDOW` 在允许窗口内 uniform；
- 对句首不足 \(k\) 的位置使用 token 0 fallback。

---

## 6. Program 来源与选择

第一版不直接把公开 GPT-2 checkpoint 的 `(layer, head)` 编号照搬到新随机初始化模型。**同一层内的 attention heads 在随机初始化时近似是可置换的；teacher 的 head 7 并不天然对应 student 的 head 7。**

因此采用“公开程序库提供候选规则，TinyStories teacher 决定规则和层，student 使用冻结的确定性 slots”的流程。

### 6.1 导入公开程序库，构造候选 DSL

克隆并只读使用：

```text
AmiriHayes/explaining_attention_heads
```

重点文件：

```text
data/gpt2_programs.py
data/iou_scores_gpt2.csv
results/best_fits_gpt2.csv
results/gpt2_program_categories.json
code/fixed_attention_gpt2.py
```

运行：

```bash
python scripts/import_public_programs.py \
  --source third_party/explaining_attention_heads \
  --output artifacts/public_programs
```

该步骤只负责：

1. 收集公开程序中的 position-only 规则；
2. 翻译成受限 DSL；
3. 在 256 个例子上验证原 Python program 与 DSL program 的 weighted IoU ≥ 0.99；
4. 生成候选集合，不采用公开 checkpoint 的 head 编号作为主实验 intervention slots。

候选至少包含：

```text
FIRST_TOKEN
SELF
PREVIOUS_K(k), k in {1, 2, 4, 8, 16, 32}
LOCAL_WINDOW(w), w in {2, 4, 8, 16, 32, 64}
```

### 6.2 训练 TinyStories teacher

先训练一个普通 GPT-2-small：

```text
condition: flash_baseline
seed: 999
train tokens: 500M
```

该 teacher 同时承担两件事：

- 定义 calibration target NLL；
- 在本项目的数据分布上发现哪些 layer 形成了可程序化 attention。

Program discovery 只能使用 `program_discovery` validation partition，不得查看 `final_evaluation`。

### 6.3 在 teacher 上匹配程序

对 teacher 的全部 144 heads，以 streaming 方式计算候选 DSL 的：

- weighted IoU；
- Jensen–Shannon divergence；
- preferred-edge attention mass；
- single-head hard-replacement \(\Delta\)NLL。

不保存整个 `[examples, layers, heads, T, T]` attention tensor。逐 batch 累积统计量。

选择规则：

1. source weighted IoU ≥ 0.65；
2. single-head hard-replacement \(\Delta\)NLL ≤ 0.005 nats/token；
3. 每层最多选择 2 个 teacher heads；
4. 总计选择 6–12 个 layer/program pairs；
5. 若同一 layer 中多个 heads 匹配同一 program，只保留 replacement cost 最低者；
6. 在 `protocol_calibration` split 上测 joint replacement；
7. 冻结后不得根据 main-run 结果调整 program set。

### 6.4 将 teacher programs 分配给 fresh students

对每个被选中的 teacher `(layer, head, program)`：

- 保留 teacher layer；
- 不假定 teacher head ID 有跨 seed 含义；
- 在该 layer 内按 program 的稳定排序，分配到 student 的最低可用 head IDs；
- 四个主实验条件和所有 paired seeds 使用完全相同的 student slots。

例如：

```json
{
  "teacher": {"layer": 3, "head": 9},
  "student": {"layer": 3, "head": 0},
  "program": {"type": "PREVIOUS_K", "parameter": 1},
  "teacher_iou": 0.83,
  "teacher_single_head_delta_nll": 0.0017
}
```

这样研究的是：

> teacher 发现的“某一层需要怎样的信息路由”能否作为 fresh model 的 inductive bias，

而不是错误地假设不同随机初始化之间的 head 编号天然对齐。

### 6.5 Program 不足时的停止规则

若 teacher 上少于 6 个 candidates 同时通过 IoU 和 replacement gate：

- 不悄悄降低阈值；
- 输出 negative calibration report；
- 允许只做一个 hypothesis-driven pilot（例如 `PREVIOUS_K(1)`）；
- 不启动昂贵的 4-condition × 4-seed main matrix。

最终写入：

```text
protocol/selected_programs.json
protocol/program_discovery_report.json
```

---

## 7. 四个主实验条件

### A. `flash_baseline`

普通 causal attention：

```python
torch.nn.functional.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)
```

主 GPU run 应强制 FlashAttention backend；若硬件或 dtype 不支持，应 fail loudly，而不是静默退回 math backend。

### B. `flex_noop`

使用 FlexAttention 和相同 causal block mask，但不加 program bias：

```python
def noop_score_mod(score, batch, head, q_idx, kv_idx):
    return score
```

该条件测量：

- FlexAttention 本身相对 production-style SDPA/Flash baseline 的 overhead；
- 不同 fused backend 的数值差异。

### C. `matched_program_prior`

在固定 slots 上加 teacher/public-library matched prior：

```python
score = score + beta_by_head[head] * preferred.to(score.dtype)
```

### D. `shuffled_program_prior`

这是 structure-matched false-program control。保持以下量完全相同：

- programmed head 数量；
- student layer/head slots；
- program family 与 program multiset；
- 每行 nonzero 数量；
- beta；
- FlexAttention code path；
- global batch；
- optimizer；
- seed。

优先使用固定 seed `1729` 将 programs 在 selected slots 之间 permutation，从而破坏 teacher 的 layer–program pairing。若 program multiset 太单一、permutation 没有实际变化，则使用固定的错误 offset（例如 `PREVIOUS_K(k)` 变为 `PREVIOUS_K(k+2)`，local window 保持宽度但整体向更早位置平移）。

该条件测试：

> 训练收益来自 teacher-matched routing semantics，还是任何具有相同 sparsity/entropy 的结构化 bias？

可选第五条件 `generic_local_prior` 只在主四条件出现积极信号后增加。

---

## 8. FlexAttention 实现要求

每一层 closure 捕获：

```python
program_type_by_head: torch.Tensor  # [n_head], int32
program_param_by_head: torch.Tensor # [n_head], int32
beta_by_head: torch.Tensor          # [n_head], float32
```

概念代码：

```python
def make_program_score_mod(
    program_type_by_head,
    program_param_by_head,
    beta_by_head,
):
    def score_mod(score, batch, head, q_idx, kv_idx):
        ptype = program_type_by_head[head]
        param = program_param_by_head[head]

        first = (ptype == FIRST_TOKEN) & (kv_idx == 0)
        self_edge = (ptype == SELF) & (kv_idx == q_idx)

        previous_source = torch.maximum(
            q_idx - param,
            torch.zeros_like(q_idx),
        )
        previous = (
            (ptype == PREVIOUS_K)
            & (kv_idx == previous_source)
        )

        local = (
            (ptype == LOCAL_WINDOW)
            & (kv_idx <= q_idx)
            & (kv_idx >= torch.maximum(
                q_idx - param + 1,
                torch.zeros_like(q_idx),
            ))
        )

        preferred = first | self_edge | previous | local
        return score + beta_by_head[head] * preferred.to(score.dtype)

    return score_mod
```

这是 design sketch；Claude Code 应根据当前 PyTorch 2.13 FlexAttention API 写成可编译版本。

硬要求：

- causal legality 用 reusable `BlockMask`；
- 不创建 `[B,H,T,T]` 或 `[T,T]` program bias tensor；
- 不在 training forward 中调用 CPU Python NLP library；
- `beta=0` 时必须与 `flex_noop` 数值一致；
- forward 和 backward 都必须通过 dense reference test；
- 记录实际 backend、PyTorch、CUDA、driver、GPU 型号；
- 编译前用 dummy batch warm up，然后重新载入相同 initial state 和 RNG，再开始计时。

---

## 9. Training protocol

```yaml
training:
  precision: bf16
  train_tokens: 500000000
  global_tokens_per_step: 131072
  optimizer: adamw
  learning_rate: 0.0005
  min_learning_rate: 0.00005
  warmup_tokens: 10000000
  schedule: cosine
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0
  eval_every_steps: 100
  eval_tokens_intermediate: 1000000
  checkpoint_every_steps: 500
  seeds: [101, 202, 303, 404]
```

500M tokens 大约对应：

```latex
\frac{500,000,000}{131,072}
\approx 3815
```

optimizer steps。

### 9.1 Microbatch 与 accumulation

自动探测最大稳定 microbatch，但保持 global tokens/step 相同：

| GPU memory | 建议起始 microbatch（sequences） | accumulation |
|---|---:|---:|
| 16–24 GB | 8 | 32 |
| 40 GB | 16 | 16 |
| 80–94 GB | 32 | 8 |
| 141–288 GB | 64 | 4 |

每条 sequence 为 512 tokens。若 OOM，将 microbatch 减半并将 accumulation 加倍。

### 9.2 Paired seeds

对于每个 seed：

- 相同 initial model state；
- 相同 batch offsets/order；
- 相同 optimizer hyperparameters；
- 相同 LR schedule；
- 相同 global batch；
- 相同 evaluation samples；
- 四个条件分别独立训练。

需要保存：

```text
initial_states/seed_101.pt
batch_schedules/seed_101.npy
```

不要依赖“重新 seed 应该产生相同结果”；直接保存 initialization 和 batch schedule。

---

## 10. Beta pilot

在 GPT-2-mini 上运行：

```yaml
beta_candidates: [0.5, 1.0, 2.0, 4.0]
pilot_tokens: 50000000
pilot_seeds: [17]
```

选择规则必须在主实验前冻结：

1. 不发生 NaN/Inf；
2. final NLL 不比 `flex_noop` 差超过 0.05 nats/token；
3. program-edge mass 明显高于 no-op；
4. 在 matched condition 中 tokens-to-pilot-target 最少；
5. 若多个 beta 相近，选择更小的 beta。

将选择结果写入：

```text
protocol/frozen_protocol.json
```

主实验不再 sweep beta。

---

## 11. Calibration run 与 target loss

在任何 program main run 之前，运行一个独立 seed `999` 的 GPT-2-small `flash_baseline`，训练 500M tokens。

定义：

```text
target_nll = calibration baseline 在 400M tokens 附近三个 evaluation points 的中位数
```

写入 `frozen_protocol.json`。主实验不得重新选择 target。

这样 primary endpoint 是：

```latex
T_{\mathrm{target}}
=
\min\{t:\mathcal L_{\mathrm{val}}(t)\le \mathcal L_{\mathrm{target}}\}.
```

同时计算：

```latex
D_{\mathrm{target}}
=
\min\{D:\mathcal L_{\mathrm{val}}(D)\le \mathcal L_{\mathrm{target}}\}.
```

若某 run 未达到 target，标记为 censored，并同时使用 loss-vs-token AUC 与 final NLL，不得把该 run 删除。

---

## 12. Metrics

### 12.1 Primary efficiency endpoint

```text
steady-state GPU wall-clock time to preregistered target NLL
```

分解为：

```text
tokens_to_target
tokens_per_second
```

关系：

```latex
\mathrm{time\ to\ target}
=
\frac{\mathrm{tokens\ to\ target}}
{\mathrm{tokens/second}}.
```

### 12.2 Primary quality guardrail

对 paired seeds 计算：

```latex
\Delta\mathcal L_s
=
\mathcal L_{\mathrm{matched},s}
-
\mathcal L_{\mathrm{flash},s}.
```

Non-inferiority margin：

```latex
\epsilon=0.02\ \mathrm{nats/token}.
```

该 margin 大约对应 2% relative perplexity：

```latex
\frac{\mathrm{PPL}_{\mathrm{matched}}}
{\mathrm{PPL}_{\mathrm{flash}}}
=
e^{\Delta\mathcal L}
\le e^{0.02}
\approx 1.0202.
```

### 12.3 Secondary model metrics

- final validation NLL；
- perplexity；
- loss-vs-token AUC；
- loss-vs-wall-clock AUC；
- NLL by context-position bins：
  - 0–127
  - 128–255
  - 256–383
  - 384–511
- common-token vs rare-token NLL；
- fixed-prompt story samples，非 primary。

### 12.4 Program-specific metrics

- weighted IoU；
- Jensen–Shannon divergence；
- program-preferred-edge attention mass；
- exact hard-replacement \(\Delta\)NLL；
- exact hard-replacement PPL ratio；
- output-distribution KL on fixed validation examples；
- number/fraction of heads that pass hardening gate。

### 12.5 Systems metrics

- forward latency；
- forward+backward latency；
- optimizer-step latency；
- tokens/s；
- peak allocated/reserved GPU memory；
- compile time；
- kernel count and profiler trace；
- GPU utilization/power；
- checkpoint/eval overhead；
- launch-to-target wall clock，作为 secondary systems metric。

---

## 13. Timing protocol

Primary steady-state timer：

- 包含 data transfer、forward、backward、optimizer step；
- 不包含 evaluation；
- 不包含 checkpoint writes；
- 不包含 first-time compile；
- 每个 timed region 前后正确同步 CUDA；
- 同时记录普通 `perf_counter` wall time和 CUDA events。

另行报告：

```text
compile_seconds
evaluation_seconds
checkpoint_seconds
total_launch_to_target_seconds
```

任何 microbenchmark 必须：

- warmup 至少 50 iterations；
- measurement 至少 200 iterations；
- forward 和 forward+backward 分开；
- sweep sequence length `[128, 256, 512, 1024]`；
- sweep batch；
- 对每个 GPU 单独报告，不跨 GPU 比较绝对 latency 后宣称算法加速。

---

## 14. Hard replacement evaluation

训练完成后，不重新训练，运行：

```bash
python -m progattn.evaluate_hard_replacement \
  --run runs/main/seed_101/matched_program_prior \
  --programs protocol/selected_programs.json \
  --split final_evaluation
```

对 selected heads：

```python
neural_probs = softmax(scores, dim=-1)
program_probs = build_program_matrix(...)

neural_probs[:, selected_heads] = program_probs
output = neural_probs @ value
```

第一阶段只需要 correctness implementation，不要求这个 hard replacement 更快。

分别测：

- 一次替换一个 head；
- 按 source IoU 从高到低累积替换；
- 一次替换全部 selected heads；
- matched model；
- shuffled-control model；
- ordinary flash baseline。

若 matched model 的 hard replacement cost 远低于 baseline，说明训练 prior 让 program 真正成为可部署组件，而不只是一般 regularizer。

---

## 15. Statistical analysis

主实验至少 4 paired seeds。不要只报告平均曲线。

输出每个 seed 的：

```text
tokens_to_target
time_to_target
final_nll
hard_replacement_delta_nll
throughput
```

Primary comparisons：

1. `matched_program_prior` vs `shuffled_program_prior`：program semantics；
2. `matched_program_prior` vs `flex_noop`：prior 在同一 implementation 下的净作用；
3. `flex_noop` vs `flash_baseline`：Flex backend overhead；
4. `matched_program_prior` vs `flash_baseline`：最终真实 wall-clock 结果。

使用：

- paired differences；
- individual seed plot；
- paired mean/median；
- bootstrap over seeds 仅作描述；
- validation uncertainty 以 story 为 resampling unit，而不是单个 token；
- 4 seeds 不做夸张的显著性 claim。

若初步 effect positive，再扩展到 8 seeds。

---

## 16. 成功、部分成功和失败的预注册解释

### Strong positive

同时满足：

- matched vs shuffled 的 mean tokens-to-target 改善 ≥10%；
- 至少 3/4 seeds 同方向；
- matched vs flash 的 steady wall-clock-to-target 改善 ≥5%；
- final NLL paired degradation 的 upper confidence bound <0.02 nats/token；
- all-head hard replacement \(\Delta\)NLL <0.02 nats/token。

解释：

> teacher-matched programs既是有用的优化先验，也接近可部署的模型组件。

### Optimization-only positive

- tokens-to-target 改善；
- wall-clock 没改善；
- final quality 合格。

解释：

> program prior 有学习/课程价值，但当前 Flex/kernel implementation overhead 抵消收益。

下一步是 systems optimization。

### Curriculum-only positive

- 训练更快；
- hard replacement 失败。

解释：

> program 是 training scaffold，而不是最终可删除的 QK component。

### Generic-structure result

- matched 与 shuffled 都同样改善。

解释：

> 收益来自 generic locality/sparsity/regularization，而不是 program semantics。

### Negative result

- matched 不改善 tokens-to-target；
- 或 final NLL 明显变差；
- 或程序 alignment 没增加。

解释：

> 该 program set、beta 或 soft-logit intervention 不支持核心 hypothesis；不要直接进入 Triton hard-kernel 阶段。

---

## 17. Repository structure

```text
programmatic-attention/
├── README.md
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
├── configs/
│   ├── smoke_prev4.yaml
│   ├── pilot_gpt2_mini.yaml
│   ├── calibration_gpt2_small.yaml
│   └── main_gpt2_small.yaml
├── protocol/
│   ├── selected_programs.json
│   └── frozen_protocol.json
├── scripts/
│   ├── prepare_tinystories.py
│   ├── import_public_programs.py
│   ├── run_smoke.sh
│   ├── run_pilot.sh
│   ├── run_calibration.sh
│   ├── run_main_local.sh
│   └── make_report.sh
├── src/progattn/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── attention_backends.py
│   ├── programs.py
│   ├── public_program_import.py
│   ├── discovery.py
│   ├── hard_replacement.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── analysis.py
│   ├── checkpoint.py
│   └── reproducibility.py
├── infra/
│   ├── modal_app.py
│   ├── Dockerfile
│   ├── slurm/
│   │   ├── calibration.sbatch
│   │   ├── main_array.sbatch
│   │   └── pack_8_gpus.sh
│   └── cloud/
│       └── README.md
├── tests/
│   ├── test_programs.py
│   ├── test_dense_attention.py
│   ├── test_flex_attention.py
│   ├── test_gradients.py
│   ├── test_hard_replacement.py
│   ├── test_pairing.py
│   ├── test_checkpoint_resume.py
│   └── test_data.py
└── runs/
```

---

## 18. Environment

建议：

```toml
requires-python = ">=3.11,<3.13"
```

依赖：

```text
torch 2.13.x
numpy
tiktoken
datasets
huggingface_hub
pyyaml
pydantic
pandas
scipy
matplotlib
tensorboard
pytest
```

`wandb` optional；核心结果必须能在无外部 tracking service 时写到本地 JSONL/CSV。

每个 run 保存 `environment.json`：

```json
{
  "git_commit": "",
  "python": "",
  "torch": "",
  "cuda_runtime": "",
  "cuda_driver": "",
  "cudnn": "",
  "gpu_name": "",
  "gpu_compute_capability": "",
  "hostname": "",
  "slurm_job_id": null,
  "modal_function_call_id": null
}
```

PyTorch/CUDA 跨机器不保证 bitwise reproducibility，所以不要把不同 GPU 型号的 runs 混进同一个 paired comparison。每一组 paired conditions 应在同一种 GPU 上完成。

---

## 19. Modal launcher

`infra/modal_app.py` 应做到：

- 一个 Modal Volume 存 data、protocol、checkpoints 和 metrics；
- 一个 run spec 对应一个单 GPU function；
- main matrix 通过 `.map()` 并行；
- 每个 function 最长 24h；
- checkpoint 后显式 commit volume；
- 支持 retry 和 resume；
- 默认 H100；A100/B200/B300 通过一个明确配置切换。

概念代码：

```python
import json
import subprocess
from pathlib import Path

import modal


app = modal.App("programmatic-attention")
volume = modal.Volume.from_name(
    "programmatic-attention-data",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.13.*",
        "numpy",
        "tiktoken",
        "datasets",
        "huggingface_hub",
        "pyyaml",
        "pydantic",
        "pandas",
        "scipy",
        "matplotlib",
    )
    .add_local_dir(".", remote_path="/workspace")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
    cpu=8,
    memory=32768,
)
def run_one(spec_json: str) -> None:
    spec = json.loads(spec_json)
    cmd = [
        "python",
        "-m",
        "progattn.train",
        "--config",
        spec["config"],
        "--condition",
        spec["condition"],
        "--seed",
        str(spec["seed"]),
        "--output",
        spec["output"],
    ]
    subprocess.run(cmd, cwd="/workspace", check=True)
    volume.commit()


@app.local_entrypoint()
def main(stage: str = "main") -> None:
    specs = build_specs(stage)
    list(run_one.map(json.dumps(spec) for spec in specs))
```

Claude Code 应根据当前 Modal API 修正细节，并提供：

```bash
modal run infra/modal_app.py --stage pilot
modal run infra/modal_app.py --stage calibration
modal run infra/modal_app.py --stage main
```

---

## 20. Slurm launcher

主实验 4 conditions × 4 seeds = 16 jobs：

```bash
#!/bin/bash
#SBATCH --job-name=progattn-main
#SBATCH --array=0-15%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --signal=B:USR1@120

set -euo pipefail

CONDITIONS=(
  flash_baseline
  flex_noop
  matched_program_prior
  shuffled_program_prior
)
SEEDS=(101 202 303 404)

condition_index=$((SLURM_ARRAY_TASK_ID % 4))
seed_index=$((SLURM_ARRAY_TASK_ID / 4))

condition=${CONDITIONS[$condition_index]}
seed=${SEEDS[$seed_index]}

srun python -m progattn.train \
  --config configs/main_gpt2_small.yaml \
  --condition "$condition" \
  --seed "$seed" \
  --output "runs/main/${seed}/${condition}"
```

如果 Schmidt allocation 一次给完整 8-GPU node，而不能申请单 GPU jobs，则：

- 一个进程占一张 GPU；
- 每个进程跑不同 condition/seed；
- 不使用 DDP；
- 使用 `CUDA_VISIBLE_DEVICES` pin GPU；
- 8 个独立 run 分两批完成 16-job matrix。

---

## 21. 推荐执行顺序与计算闸门

### Milestone 0：CPU/GPU correctness

```bash
pytest -q
python -m progattn.benchmark --mode correctness
```

通过条件：

- dense baseline tests；
- Flex no-op forward/backward；
- beta=0 equivalence；
- hard program rows sum to 1；
- causal legality；
- checkpoint resume reproduces next 10 losses；
- no program bias tensor materialization。

### Milestone 1：synthetic `previous-4` smoke

构造一个小型周期/延迟复制任务：

- true prior：PREVIOUS_K(4)；
- fake prior：PREVIOUS_K(3)；
- 2 layers、4 heads、small vocabulary；
- 目标仅为验证 intervention 能产生预期优化差异。

不把结果当自然语言证据。

### Milestone 2：GPT-2-mini beta pilot

- 50M tokens；
- one seed；
- beta sweep；
- 记录 throughput、NLL、program mass；
- 冻结 beta。

预计使用一张 GPU。

### Milestone 3：public program import + calibration

- 导入/验证 deployable public programs；
- 若不足，运行 DSL discovery fallback；
- 运行 GPT-2-small calibration seed 999；
- 冻结 target NLL；
- 生成 `frozen_protocol.json`。

### Milestone 4：GPT-2-small main matrix

- 4 conditions；
- 4 paired seeds；
- 500M tokens/run；
- 同一 GPU family；
- independent single-GPU jobs。

### Milestone 5：hard replacement + report

```bash
python scripts/make_report.py \
  --runs runs/main \
  --protocol protocol/frozen_protocol.json \
  --output reports/main
```

### Milestone 6：只有 positive signal 后才做 hard kernels

第一版 hard kernel 只实现：

- FIRST_TOKEN broadcast；
- PREVIOUS_K gather/shift；
- LOCAL_WINDOW fixed-width reduction。

真正的模型修改：

```python
q_neural = q_proj_neural(x)
k_neural = k_proj_neural(x)
v_all = v_proj_all(x)

out_neural = flash_attention(q_neural, k_neural, v_neural)
out_program = program_kernel(v_program, program_specs)

out = output_projection(pack_heads(out_neural, out_program))
```

要避免“已经算了 program heads 的 Q/K，最后只是忽略它们”的伪加速。

---

## 22. 资源预算

### 最小开发环境

```text
GPU: 16 GB 可运行；24 GB 更舒服
CPU: 8 cores
RAM: 32 GB
Disk: 30 GB minimum
```

### 主实验推荐

```text
单 run：1 × A100 40/80GB、H100、H200、B200 或 B300
不需要 model parallel
不需要 DDP
```

多个 GPU 的用途是同时跑不同 seed/condition。

### 数据与 checkpoint 存储

- tokenized train 数据约 1 GB；
- validation 约 10 MB；
- 一个完整 training checkpoint（含 Adam states）约 2–3 GB；
- 每个 run 只保留 `best`、`last` 和必要 milestone；
- 16 main runs 建议预留 100–150 GB；
- profiler traces 单独限额，避免数百 GB。

### 预算层级

```text
Smoke + GPT-2-mini pilot:
  约 1–5 GPU-hours

四条件、单 seed、100M tokens/condition 的 preflight:
  约 1–4 GPU-hours

GPT-2-small 500M-token main:
  16 runs + 1 calibration = 8.5B processed tokens
  大约 8–47 GPU-hours，取决于 GPU 与实际 Flex throughput
```

所有 estimate 都必须由目标硬件上的 10 分钟 calibration 替换。不要根据理论 peak FLOPs 直接购买大量 cloud time。

---

## 23. Claude Code 验收标准

Claude Code 不应在“代码能 import”时宣布完成。至少满足：

- [ ] TinyStories 可复现预处理，metadata 和 hashes 完整；
- [ ] GPT-2-mini 能在单 GPU 正常训练；
- [ ] GPT-2-small 参数量与 config 报告正确；
- [ ] Flash baseline 明确确认使用 Flash backend；
- [ ] Flex no-op 与 dense reference 通过 forward/backward tolerance；
- [ ] program prior 不 materialize \(T\times T\) bias；
- [ ] public program import 有 DSL equivalence report；
- [ ] matched/shuffled assignments 可复现；
- [ ] paired initialization 和 batch schedule 由文件保证；
- [ ] checkpoint/resume 保留 optimizer、scheduler、scaler、RNG、data position；
- [ ] run manifest 记录完整 environment；
- [ ] calibration protocol 会冻结 beta、program set、target NLL；
- [ ] main launcher 能生成恰好 16 个 unique run specs；
- [ ] analysis report 包含 individual seeds，不只平均；
- [ ] hard replacement 支持 single、cumulative、all selected heads；
- [ ] Modal launcher 和 Slurm array 至少各通过一个 smoke run；
- [ ] README 包含从空目录到 smoke/main/report 的命令；
- [ ] 主实验不会在 protocol 未冻结时启动；
- [ ] 第一阶段没有无必要的 Triton/CUDA 自定义 kernel。

---

## 24. 最终报告必须回答的问题

1. matched programs 是否比 shuffled programs 更少 tokens 达到 target？
2. FlexAttention 的 overhead 是多少？
3. tokens savings 是否足以抵消 throughput loss？
4. matched model 的最终 NLL 是否 non-inferior？
5. program-edge attention mass 是否真的增加？
6. hard replacement 是否在 matched model 中更安全？
7. 效果是否跨 seeds 一致？
8. 效果是否集中在某些 layer/program type？
9. B200/B300 上的结果是否只是 hardware throughput，还是算法 wall-clock gain？
10. 下一步应该：
   - 写 hard kernel；
   - 增加 content-dependent programs；
   - 换现代 1B model；
   - 还是停止该方向？

---

## 25. 第一次实际运行建议

不要先申请 8 张 B300。按下面顺序：

```bash
# 1. 本地或一张便宜 GPU
pytest -q
python -m progattn.train --config configs/smoke_prev4.yaml

# 2. Modal A100/H100 或 Caltech 单卡
python -m progattn.launch --stage pilot

# 3. 只跑四条件 × 单 seed × 100M tokens preflight
python -m progattn.launch --stage preflight

# 4. 查看 preflight report；只有实现和信号合理才继续
python scripts/make_report.py --runs runs/preflight

# 5. Schmidt/Caltech job array 跑完整 paired seeds
sbatch infra/slurm/main_array.sbatch
```

这个顺序将“代码 bug”“Flex backend overhead”“program prior 没有科学信号”和“需要大规模多 seed”分开，避免一上来浪费昂贵 GPU。
