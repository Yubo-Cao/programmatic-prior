from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import cast

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
    uniform_edge_mass: float | None = None
    selection_score: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "head": self.head,
            "program_type": self.program_type.name,
            "parameter": self.parameter,
            "source_layer": self.source_layer,
            "source_head": self.source_head,
            "weighted_iou": self.weighted_iou,
            "js_divergence": self.js_divergence,
            "preferred_edge_mass": self.preferred_edge_mass,
            "uniform_edge_mass": self.uniform_edge_mass,
            "selection_score": self.selection_score,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> ProgramSpec:
        return cls(
            layer=_required_int(value.get("layer"), "layer"),
            head=_required_int(value.get("head"), "head"),
            program_type=ProgramType[_required_str(value.get("program_type"))],
            parameter=_required_int(value.get("parameter", 0), "parameter"),
            source_layer=_optional_int(value.get("source_layer")),
            source_head=_optional_int(value.get("source_head")),
            weighted_iou=_optional_float(value.get("weighted_iou")),
            js_divergence=_optional_float(value.get("js_divergence")),
            preferred_edge_mass=_optional_float(value.get("preferred_edge_mass")),
            uniform_edge_mass=_optional_float(value.get("uniform_edge_mass")),
            selection_score=_optional_float(value.get("selection_score")),
        )


def _required_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _required_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("program_type must be a string")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _required_int(value, "optional integer")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError("optional float must be numeric")
    return float(value)


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
        raw: object = json.load(handle)
    entries: object = raw.get("programs") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise TypeError("program file must contain a list")
    result: list[ProgramSpec] = []
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(key, str) for key in entry
        ):
            raise TypeError("each program must be an object with string keys")
        result.append(ProgramSpec.from_json(cast(dict[str, object], entry)))
    return result


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
            raise ValueError(
                f"multiple programs assigned to layer {layer}, head {spec.head}"
            )
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
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    def score_mod(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
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
