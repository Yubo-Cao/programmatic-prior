from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .config import ExperimentConfig, load_config
from .data import TokenStore
from .model import GPT, autocast_context
from .programs import (
    ProgramSpec,
    candidate_programs,
    dense_program_mask,
    save_programs,
)
from .utils import atomic_json, json_object, sha256_file


@dataclass
class MetricTotal:
    weighted_iou: float = 0.0
    js_divergence: float = 0.0
    preferred_edge_mass: float = 0.0
    uniform_edge_mass: float = 0.0
    query_count: int = 0

    def add(
        self,
        *,
        weighted_iou: torch.Tensor,
        js_divergence: torch.Tensor,
        preferred_edge_mass: torch.Tensor,
        uniform_edge_mass: torch.Tensor,
        valid_queries: torch.Tensor,
    ) -> None:
        weights = valid_queries.float()
        self.weighted_iou += float((weighted_iou * weights).sum().item())
        self.js_divergence += float((js_divergence * weights).sum().item())
        self.preferred_edge_mass += float((preferred_edge_mass * weights).sum().item())
        self.uniform_edge_mass += float((uniform_edge_mass * weights).sum().item())
        self.query_count += int(weights.sum().item())

    def means(self) -> tuple[float, float, float, float]:
        denominator = max(1, self.query_count)
        return (
            self.weighted_iou / denominator,
            self.js_divergence / denominator,
            self.preferred_edge_mass / denominator,
            self.uniform_edge_mass / denominator,
        )


def selection_score(preferred_mass: float, uniform_mass: float) -> float:
    """Bits of evidence that a head follows a program rather than reading uniformly.

    ``weighted_iou`` rises monotonically with the width of the preferred set, so on
    its own it always crowns the widest candidate in the DSL and says nothing about
    whether the head is actually structured. Dividing the observed preferred mass by
    the mass a uniform causal reader would place on the same set removes that width
    advantage, and weighting the log ratio by the mass keeps a razor-sharp program
    that explains almost nothing from outranking a slightly wider one that explains
    most of the head.
    """
    if preferred_mass <= 0.0 or uniform_mass <= 0.0:
        return 0.0
    return preferred_mass * math.log2(preferred_mass / uniform_mass)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("protocol/discovery.json"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--criterion",
        choices=("enrichment", "weighted_iou"),
        default="enrichment",
        help=(
            "how to rank layer-head pairs. 'weighted_iou' reproduces the pilot "
            "selection, which is monotone in program width; 'enrichment' scores a "
            "head against a uniform causal reader instead."
        ),
    )
    return parser.parse_args()


def program_distributions(
    config: ExperimentConfig, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    sequence_length = config.model.block_size
    positions = torch.arange(1, sequence_length + 1, device=device, dtype=torch.float32)
    result: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for program_type, parameter in candidate_programs():
        types = torch.tensor([int(program_type)], dtype=torch.int32, device=device)
        params = torch.tensor([parameter], dtype=torch.int32, device=device)
        mask = dense_program_mask(types, params, sequence_length, layer=0)[0]
        probability = mask.float() / mask.sum(dim=-1, keepdim=True).clamp_min(1)
        uniform = mask.sum(dim=-1).float() / positions
        result.append((mask, probability, uniform))
    return result


def compare_distribution(
    attention: torch.Tensor,
    mask: torch.Tensor,
    program_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = 1e-8
    attention = attention.float().clamp_min(epsilon)
    program = program_probability.clamp_min(epsilon)
    midpoint = 0.5 * (attention + program[None, None])
    js = 0.5 * (
        (attention * (attention / midpoint).log()).sum(dim=-1)
        + (program[None, None] * (program[None, None] / midpoint).log()).sum(dim=-1)
    )
    intersection = torch.minimum(attention, program[None, None]).sum(dim=-1)
    weighted_iou = intersection / (2.0 - intersection).clamp_min(epsilon)
    preferred_mass = (attention * mask[None, None]).sum(dim=-1)
    return weighted_iou, js, preferred_mass


def _rank_key(criterion: str) -> Callable[[ProgramSpec], tuple[float, float]]:
    if criterion == "weighted_iou":
        return lambda spec: (spec.weighted_iou or 0.0, -(spec.js_divergence or 0.0))
    return lambda spec: (spec.selection_score or 0.0, -(spec.js_divergence or 0.0))


def head_metrics(
    totals: list[list[list[MetricTotal]]], config: ExperimentConfig
) -> list[ProgramSpec]:
    """Every candidate program scored on every layer-head pair."""
    candidates = candidate_programs()
    entries: list[ProgramSpec] = []
    for layer in range(config.model.n_layer):
        for head in range(config.model.n_head):
            for candidate_index, (program_type, parameter) in enumerate(candidates):
                iou, js, mass, uniform = totals[layer][head][candidate_index].means()
                entries.append(
                    ProgramSpec(
                        layer=layer,
                        head=head,
                        program_type=program_type,
                        parameter=parameter,
                        source_layer=layer,
                        source_head=head,
                        weighted_iou=iou,
                        js_divergence=js,
                        preferred_edge_mass=mass,
                        uniform_edge_mass=uniform,
                        selection_score=selection_score(mass, uniform),
                    )
                )
    return entries


def choose_programs(
    entries: list[ProgramSpec], config: ExperimentConfig, criterion: str
) -> list[ProgramSpec]:
    """Best program per head, then the top pairs under ``criterion``.

    The per-layer cap and the total count come from the config and are shared by
    both criteria, so the two selections differ only in how a head is ranked.
    """
    key = _rank_key(criterion)
    best_by_head: dict[tuple[int, int], ProgramSpec] = {}
    for entry in entries:
        pair = (entry.layer, entry.head)
        current = best_by_head.get(pair)
        if current is None or key(entry) > key(current):
            best_by_head[pair] = entry
    ranked = sorted(best_by_head.values(), key=key, reverse=True)
    selected: list[ProgramSpec] = []
    per_layer: dict[int, int] = {}
    for spec in ranked:
        if per_layer.get(spec.layer, 0) >= config.prior.max_per_layer:
            continue
        selected.append(spec)
        per_layer[spec.layer] = per_layer.get(spec.layer, 0) + 1
        if len(selected) == config.prior.selection_count:
            break
    if len(selected) != config.prior.selection_count:
        raise RuntimeError(
            "not enough layer-head pairs to freeze the pilot program set"
        )
    return selected


@torch.no_grad()
def run_discovery(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device(args.device)
    data_root = (args.data_root or Path(config.data.root)).resolve()
    model = GPT(config.model, condition="flash_baseline", strict_flash=False)
    raw: Any = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint = json_object(raw)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    store = TokenStore(data_root)
    distributions = program_distributions(config, device)
    candidate_count = len(distributions)
    totals = [
        [
            [MetricTotal() for _ in range(candidate_count)]
            for _ in range(config.model.n_head)
        ]
        for _ in range(config.model.n_layer)
    ]
    processed_stories = 0
    for x, _ in store.discovery_batches(
        block_size=config.model.block_size,
        batch_size=config.prior.discovery_batch_size,
        max_stories=config.prior.discovery_stories,
        pin_memory=device.type == "cuda",
    ):
        valid_queries = x.ne(50256).to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        with autocast_context(device, config.training.precision):
            _, _, attentions = cast(
                tuple[torch.Tensor, object, list[torch.Tensor] | None],
                model(x, return_attentions=True, force_dense=True),
            )
        if attentions is None:
            raise RuntimeError("dense discovery did not return attention probabilities")
        processed_stories += x.size(0)
        for layer, attention in enumerate(attentions):
            for candidate_index, (mask, probability, uniform) in enumerate(
                distributions
            ):
                iou, js, mass = compare_distribution(attention, mask, probability)
                for head in range(config.model.n_head):
                    totals[layer][head][candidate_index].add(
                        weighted_iou=iou[:, head],
                        js_divergence=js[:, head],
                        preferred_edge_mass=mass[:, head],
                        uniform_edge_mass=uniform[None, :],
                        valid_queries=valid_queries,
                    )

    entries = head_metrics(totals, config)
    selected = choose_programs(entries, config, args.criterion)
    legacy = choose_programs(entries, config, "weighted_iou")
    metadata: dict[str, object] = {
        "source_condition": "flash_baseline",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "selection_split": "program_discovery",
        "selection_criterion": args.criterion,
        "selection_mode": f"top_k_flash_matched_by_{args.criterion}",
        "selected_count": len(selected),
        "max_per_layer": config.prior.max_per_layer,
        "processed_stories": processed_stories,
        "student_assignment": "same_layer_and_head_for_paired_seed",
    }
    save_programs(args.output, selected, metadata)
    atomic_json(
        args.report,
        {
            "metadata": metadata,
            "selected_programs": [spec.to_json() for spec in selected],
            "weighted_iou_selection": [spec.to_json() for spec in legacy],
            "all_candidate_metrics": [spec.to_json() for spec in entries],
        },
    )
    print(json.dumps([spec.to_json() for spec in selected], indent=2))


def main() -> None:
    run_discovery(parse_args())


if __name__ == "__main__":
    main()
