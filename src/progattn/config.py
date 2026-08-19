from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONDITIONS = (
    "flash_baseline",
    "flex_noop",
    "matched_program_prior",
    "incorrect_program_prior",
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_ratio: int = 4
    bias: bool = True
    dropout: float = 0.0
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")


@dataclass(frozen=True)
class DataConfig:
    root: str
    schedule_dir: str
    initial_state_dir: str


@dataclass(frozen=True)
class TrainingConfig:
    train_tokens: int
    global_tokens_per_step: int
    micro_batch_size: int
    precision: str
    learning_rate: float
    min_learning_rate: float
    warmup_tokens: int
    betas: tuple[float, float]
    weight_decay: float
    grad_clip: float
    eval_every_steps: int
    eval_tokens: int
    checkpoint_every_steps: int
    seed: int
    compile: bool
    strict_flash: bool


@dataclass(frozen=True)
class PriorConfig:
    selected_programs: str
    initial_alpha: float
    warmup_tokens: int
    control_seed: int
    selection_count: int
    max_per_layer: int
    discovery_stories: int
    discovery_batch_size: int


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    prior: PriorConfig
    source_path: Path

    @property
    def sequences_per_step(self) -> int:
        tokens = self.training.global_tokens_per_step
        if tokens % self.model.block_size:
            raise ValueError("global_tokens_per_step must be divisible by block_size")
        return tokens // self.model.block_size

    @property
    def accumulation_steps(self) -> int:
        sequences = self.sequences_per_step
        micro = self.training.micro_batch_size
        if sequences % micro:
            raise ValueError(
                "global sequence batch must be divisible by micro_batch_size"
            )
        return sequences // micro

    @property
    def total_steps(self) -> int:
        tokens = self.training.train_tokens
        per_step = self.training.global_tokens_per_step
        return (tokens + per_step - 1) // per_step


def _strict_dataclass(cls: type[Any], values: dict[str, Any]) -> Any:
    expected = set(cls.__dataclass_fields__)
    extra = set(values) - expected
    if extra:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(extra)}")
    if cls is TrainingConfig and isinstance(values.get("betas"), list):
        values = {**values, "betas": tuple(values["betas"])}
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    expected = {"model", "data", "training", "prior"}
    if set(raw) != expected:
        raise ValueError(f"config sections must be exactly {sorted(expected)}")
    config = ExperimentConfig(
        model=_strict_dataclass(ModelConfig, raw["model"]),
        data=_strict_dataclass(DataConfig, raw["data"]),
        training=_strict_dataclass(TrainingConfig, raw["training"]),
        prior=_strict_dataclass(PriorConfig, raw["prior"]),
        source_path=source,
    )
    config.model.validate()
    _ = config.accumulation_steps
    return config
