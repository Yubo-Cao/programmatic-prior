from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .programs import (
    ProgramSpec,
    dense_program_mask,
    inverse_softplus,
    make_score_mod,
    program_tensors,
)

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


def causal_mask(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    del batch, head
    return q_idx >= kv_idx


def noop_score_mod(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    del batch, head, q_idx, kv_idx
    return score


class ScaledLinear(nn.Linear):
    pass


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        layer: int,
        condition: str,
        programs: list[ProgramSpec],
        initial_alpha: float,
        control_seed: int,
        strict_flash: bool,
    ) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.layer = layer
        self.condition = condition
        self.control_seed = control_seed
        self.strict_flash = strict_flash

        self.qkv: nn.Linear = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        self.proj: ScaledLinear = ScaledLinear(
            config.n_embd, config.n_embd, bias=config.bias
        )
        self.attn_dropout = config.dropout
        self.resid_dropout: nn.Dropout = nn.Dropout(config.dropout)

        types, params = program_tensors(programs, layer=layer, n_head=config.n_head)
        self.has_programs = bool(types.ne(0).any().item())
        self.program_types: torch.Tensor
        self.program_params: torch.Tensor
        self.prior_warmup_scale: torch.Tensor
        self.register_buffer("program_types", types, persistent=False)
        self.register_buffer("program_params", params, persistent=False)
        self.register_buffer("prior_warmup_scale", torch.tensor(0.0), persistent=False)
        raw_alpha = torch.full((config.n_head,), inverse_softplus(initial_alpha))
        self.raw_alpha: nn.Parameter = nn.Parameter(raw_alpha)
        self.raw_alpha.requires_grad_(
            condition in {"matched_program_prior", "incorrect_program_prior"}
        )
        self._block_mask: BlockMask | None = None

    @property
    def uses_prior(self) -> bool:
        return self.condition in {"matched_program_prior", "incorrect_program_prior"}

    @property
    def applies_prior(self) -> bool:
        """Whether the prior can change any score in this layer.

        A layer that owns no selected head would only ever add ``alpha * 0``. Building
        the score modifier anyway makes FlexAttention differentiate through a captured
        grad-carrying tensor, which costs an order of magnitude per step for nothing.
        """
        return self.uses_prior and self.has_programs

    def set_prior_warmup_scale(self, value: float) -> None:
        self.prior_warmup_scale.fill_(value)

    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) * self.prior_warmup_scale

    def prepare_flex(self, device: torch.device) -> None:
        if not self.condition.startswith("flex") and not self.uses_prior:
            return
        from torch.nn.attention.flex_attention import create_block_mask

        self._block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=self.block_size,
            KV_LEN=self.block_size,
            device=device,
        )

    def _flash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.device.type != "cuda":
            if self.strict_flash:
                raise RuntimeError("strict FlashAttention requires a CUDA tensor")
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )

    def _flex(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from torch.nn.attention.flex_attention import flex_attention

        if self._block_mask is None:
            raise RuntimeError("call prepare_attention before using FlexAttention")
        score_mod = noop_score_mod
        if self.applies_prior:
            score_mod = make_score_mod(
                self.program_types,
                self.program_params,
                self.alpha(),
                layer=self.layer,
                incorrect=self.condition == "incorrect_program_prior",
                control_seed=self.control_seed,
            )
        result = flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=self._block_mask,
        )
        if not isinstance(result, torch.Tensor):
            raise TypeError("FlexAttention returned unexpected auxiliary output")
        return result

    def _dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        apply_prior: bool,
        hard_overrides: Mapping[int, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = q.size(-2)
        scores = q.float() @ k.float().transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=q.device,
        ).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        if apply_prior and self.applies_prior:
            mask = dense_program_mask(
                self.program_types,
                self.program_params,
                sequence_length,
                layer=self.layer,
                incorrect=self.condition == "incorrect_program_prior",
                control_seed=self.control_seed,
            )
            scores = scores + self.alpha()[None, :, None, None] * mask[None]
        probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
        if hard_overrides:
            probabilities = probabilities.clone()
            for head, matrix in hard_overrides.items():
                probabilities[:, head] = matrix.to(probabilities.dtype)
        return probabilities @ v, probabilities

    def forward(
        self,
        x: torch.Tensor,
        *,
        force_dense: bool = False,
        return_attention: bool = False,
        hard_overrides: Mapping[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, channels = x.shape
        qkv = cast(torch.Tensor, self.qkv(x))
        q, k, v = qkv.split(channels, dim=-1)
        q = q.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(
            1, 2
        )

        probabilities = None
        if force_dense or return_attention or hard_overrides:
            y, probabilities = self._dense(
                q,
                k,
                v,
                apply_prior=self.uses_prior,
                hard_overrides=hard_overrides,
            )
        elif self.condition == "flash_baseline":
            y = self._flash(q, k, v)
        else:
            y = self._flex(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        output = cast(torch.Tensor, self.resid_dropout(self.proj(y)))
        return output, probabilities


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.fc: nn.Linear = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.proj: ScaledLinear = ScaledLinear(hidden, config.n_embd, bias=config.bias)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))
        return cast(torch.Tensor, output)


class Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        layer: int,
        condition: str,
        programs: list[ProgramSpec],
        initial_alpha: float,
        control_seed: int,
        strict_flash: bool,
    ) -> None:
        super().__init__()
        self.ln_1: nn.LayerNorm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn: CausalSelfAttention = CausalSelfAttention(
            config,
            layer=layer,
            condition=condition,
            programs=programs,
            initial_alpha=initial_alpha,
            control_seed=control_seed,
            strict_flash=strict_flash,
        )
        self.ln_2: nn.LayerNorm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp: MLP = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        force_dense: bool = False,
        return_attention: bool = False,
        hard_overrides: Mapping[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = cast(torch.Tensor, self.ln_1(x))
        attn_out, probabilities = cast(
            tuple[torch.Tensor, torch.Tensor | None],
            self.attn(
                normalized,
                force_dense=force_dense,
                return_attention=return_attention,
                hard_overrides=hard_overrides,
            ),
        )
        x = x + attn_out
        normalized = cast(torch.Tensor, self.ln_2(x))
        x = x + cast(torch.Tensor, self.mlp(normalized))
        return x, probabilities


class GPT(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        condition: str,
        programs: list[ProgramSpec] | None = None,
        initial_alpha: float = 0.1,
        control_seed: int = 1729,
        strict_flash: bool = True,
    ) -> None:
        super().__init__()
        programs = programs or []
        self.config = config
        self.condition = condition
        self.token_embedding: nn.Embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )
        self.position_embedding: nn.Embedding = nn.Embedding(
            config.block_size, config.n_embd
        )
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.blocks: nn.ModuleList = nn.ModuleList(
            Block(
                config,
                layer=layer,
                condition=condition,
                programs=programs,
                initial_alpha=initial_alpha,
                control_seed=control_seed,
                strict_flash=strict_flash,
            )
            for layer in range(config.n_layer)
        )
        self.ln_f: nn.LayerNorm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head: nn.Linear = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if isinstance(module, ScaledLinear):
                std /= math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def prepare_attention(self, device: torch.device) -> None:
        for block in self.blocks:
            assert isinstance(block, Block)
            block.attn.prepare_flex(device)

    def set_prior_progress(self, tokens_seen: int, warmup_tokens: int) -> None:
        scale = min(1.0, tokens_seen / max(1, warmup_tokens))
        for block in self.blocks:
            assert isinstance(block, Block)
            block.attn.set_prior_warmup_scale(scale)

    def alpha_summary(self) -> dict[str, float | list[float]]:
        values: list[float] = []
        for block in self.blocks:
            assert isinstance(block, Block)
            selected = block.attn.program_types.ne(0)
            if bool(selected.any().item()):
                values.extend(
                    cast(
                        list[float],
                        F.softplus(block.attn.raw_alpha[selected])
                        .detach()
                        .cpu()
                        .tolist(),
                    )
                )
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "values": []}
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        force_dense: bool = False,
        return_attentions: bool = False,
        hard_overrides: Mapping[int, Mapping[int, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        _, sequence_length = tokens.shape
        if sequence_length > self.config.block_size:
            raise ValueError(f"sequence length {sequence_length} exceeds block size")
        positions = torch.arange(sequence_length, device=tokens.device)
        x = cast(
            torch.Tensor,
            self.dropout(
                self.token_embedding(tokens) + self.position_embedding(positions)
            ),
        )
        attentions: list[torch.Tensor] | None = [] if return_attentions else None
        for layer, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x, probabilities = block(
                x,
                force_dense=force_dense,
                return_attention=return_attentions,
                hard_overrides=None
                if hard_overrides is None
                else hard_overrides.get(layer),
            )
            if attentions is not None:
                assert probabilities is not None
                attentions.append(probabilities)
        logits = cast(torch.Tensor, self.lm_head(self.ln_f(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, attentions


def autocast_context(
    device: torch.device, precision: str
) -> AbstractContextManager[None]:
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return cast(
        AbstractContextManager[None],
        torch.autocast(device_type="cuda", dtype=dtype),
    )


def configure_optimizer(
    model: GPT,
    *,
    learning_rate: float,
    betas: tuple[float, float],
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2 and not name.endswith("raw_alpha"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=betas,
        fused=device.type == "cuda",
    )
