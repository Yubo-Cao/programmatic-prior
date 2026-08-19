from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import json
import math
from pathlib import Path
from typing import Iterable

import torch


class ProgramType(IntEnum):
    NONE = 0
    FIRST_TOKEN = 1
    SELF = 2
    PREVIOUS_K = 3
    LOCAL_WINDOW = 4


@dataclass(frozen=True)
class ProgramSpec:
    layer: int
    head: int
    program_type: ProgramType
    parameter: int = 0
    source_layer: int | None = None
    source_head: int | None = None
    weighted_iou: float | None = None
    js_divergence: float | None = None
    preferred_edge_mass: float | None = None

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["program_type"] = self.program_type.name
        return result

    @classmethod
    def from_json(cls, value: dict[str, object]) -> "ProgramSpec":
        return cls(
            layer=int(value["layer"]),
            head=int(value["head"]),
            program_type=ProgramType[str(value["program_type"])],
            parameter=int(value.get("parameter", 0)),
            source_layer=_optional_int(value.get("source_layer")),
            source_head=_optional_int(value.get("source_head")),
            weighted_iou=_optional_float(value.get("weighted_iou")),
            js_divergence=_optional_float(value.get("js_divergence")),
            preferred_edge_mass=_optional_float(value.get("preferred_edge_mass")),
        )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def candidate_programs() -> tuple[tuple[ProgramType, int], ...]:
    return (
        (ProgramType.FIRST_TOKEN, 0),
        (ProgramType.SELF, 0),
        *((ProgramType.PREVIOUS_K, k) for k in (1, 2, 4, 8, 16, 32)),
        *((ProgramType.LOCAL_WINDOW, w) for w in (2, 4, 8, 16, 32, 64)),
    )


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


def load_programs(path: str | Path) -> list[ProgramSpec]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    entries = raw["programs"] if isinstance(raw, dict) else raw
    return [ProgramSpec.from_json(entry) for entry in entries]


def save_programs(
    path: str | Path,
    programs: Iterable[ProgramSpec],
    metadata: dict[str, object],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "programs": [item.to_json() for item in programs]}
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def program_tensors(
    programs: Iterable[ProgramSpec],
    *,
    layer: int,
    n_head: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    types = torch.zeros(n_head, dtype=torch.int32, device=device)
    params = torch.zeros(n_head, dtype=torch.int32, device=device)
    for spec in programs:
        if spec.layer != layer:
            continue
        if not 0 <= spec.head < n_head:
            raise ValueError(f"head {spec.head} is invalid for n_head={n_head}")
        if types[spec.head].item() != ProgramType.NONE:
            raise ValueError(f"multiple programs assigned to layer {layer}, head {spec.head}")
        types[spec.head] = int(spec.program_type)
        params[spec.head] = spec.parameter
    return types, params


def _logical_key(
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    head: torch.Tensor,
    *,
    layer: int,
    control_seed: int,
) -> torch.Tensor:
    """Causally rotate each row; this is a bijection over keys 0..q."""
    span = q_idx + 1
    positive_q = torch.clamp(q_idx, min=1)
    raw = control_seed + 131 * head + 17 * layer
    shift = torch.where(q_idx > 0, 1 + torch.remainder(raw, positive_q), 0)
    return torch.remainder(kv_idx + shift, span)


def preferred_edges(
    program_type: torch.Tensor,
    parameter: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros_like(q_idx)
    first = (program_type == ProgramType.FIRST_TOKEN) & (kv_idx == 0)
    self_edge = (program_type == ProgramType.SELF) & (kv_idx == q_idx)
    source = torch.maximum(q_idx - parameter, zero)
    previous = (program_type == ProgramType.PREVIOUS_K) & (kv_idx == source)
    left = torch.maximum(q_idx - parameter + 1, zero)
    local = (
        (program_type == ProgramType.LOCAL_WINDOW)
        & (kv_idx >= left)
        & (kv_idx <= q_idx)
    )
    return first | self_edge | previous | local


def make_score_mod(
    program_type_by_head: torch.Tensor,
    program_param_by_head: torch.Tensor,
    beta_by_head: torch.Tensor,
    *,
    layer: int,
    incorrect: bool,
    control_seed: int,
):
    def score_mod(score, batch, head, q_idx, kv_idx):
        del batch
        logical_k = kv_idx
        if incorrect:
            logical_k = _logical_key(
                q_idx,
                kv_idx,
                head,
                layer=layer,
                control_seed=control_seed,
            )
        preferred = preferred_edges(
            program_type_by_head[head],
            program_param_by_head[head],
            q_idx,
            logical_k,
        )
        return score + beta_by_head[head].to(score.dtype) * preferred.to(score.dtype)

    return score_mod


def dense_program_mask(
    program_type_by_head: torch.Tensor,
    program_param_by_head: torch.Tensor,
    sequence_length: int,
    *,
    layer: int,
    incorrect: bool = False,
    control_seed: int = 1729,
) -> torch.Tensor:
    device = program_type_by_head.device
    heads = torch.arange(len(program_type_by_head), device=device)[:, None, None]
    q_idx = torch.arange(sequence_length, device=device)[None, :, None]
    kv_idx = torch.arange(sequence_length, device=device)[None, None, :]
    logical_k = kv_idx
    if incorrect:
        logical_k = _logical_key(
            q_idx,
            kv_idx,
            heads,
            layer=layer,
            control_seed=control_seed,
        )
    result = preferred_edges(
        program_type_by_head[:, None, None],
        program_param_by_head[:, None, None],
        q_idx,
        logical_k,
    )
    causal = kv_idx <= q_idx
    return result & causal
