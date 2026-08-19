from __future__ import annotations

import argparse
import json
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
    query_count: int = 0

    def add(
        self,
        *,
        weighted_iou: torch.Tensor,
        js_divergence: torch.Tensor,
        preferred_edge_mass: torch.Tensor,
        valid_queries: torch.Tensor,
    ) -> None:
        weights = valid_queries[:, None, :].float()
        self.weighted_iou += float((weighted_iou * weights).sum().item())
        self.js_divergence += float((js_divergence * weights).sum().item())
        self.preferred_edge_mass += float((preferred_edge_mass * weights).sum().item())
        self.query_count += int(weights.sum().item())

    def means(self) -> tuple[float, float, float]:
        denominator = max(1, self.query_count)
        return (
            self.weighted_iou / denominator,
            self.js_divergence / denominator,
            self.preferred_edge_mass / denominator,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("protocol/discovery.json"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def program_distributions(
    config: ExperimentConfig, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sequence_length = config.model.block_size
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for program_type, parameter in candidate_programs():
        types = torch.tensor([int(program_type)], dtype=torch.int32, device=device)
        params = torch.tensor([parameter], dtype=torch.int32, device=device)
        mask = dense_program_mask(types, params, sequence_length, layer=0)[0]
        probability = mask.float() / mask.sum(dim=-1, keepdim=True).clamp_min(1)
        result.append((mask, probability))
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


def select_programs(
    totals: list[list[list[MetricTotal]]], config: ExperimentConfig
) -> tuple[list[ProgramSpec], list[dict[str, object]]]:
    candidates = candidate_programs()
    all_best: list[ProgramSpec] = []
    report: list[dict[str, object]] = []
    for layer in range(config.model.n_layer):
        for head in range(config.model.n_head):
            best: ProgramSpec | None = None
            for candidate_index, (program_type, parameter) in enumerate(candidates):
                iou, js, mass = totals[layer][head][candidate_index].means()
                entry = ProgramSpec(
                    layer=layer,
                    head=head,
                    program_type=program_type,
                    parameter=parameter,
                    source_layer=layer,
                    source_head=head,
                    weighted_iou=iou,
                    js_divergence=js,
                    preferred_edge_mass=mass,
                )
                report.append(entry.to_json())
                if best is None or (iou, -js) > (
                    best.weighted_iou or 0.0,
                    -(best.js_divergence or 0.0),
                ):
                    best = entry
            assert best is not None
            all_best.append(best)

    all_best.sort(
        key=lambda spec: (
            spec.weighted_iou or 0.0,
            -(spec.js_divergence or 0.0),
        ),
        reverse=True,
    )
    selected: list[ProgramSpec] = []
    per_layer: dict[int, int] = {}
    for spec in all_best:
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
    return selected, report


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
            for candidate_index, (mask, probability) in enumerate(distributions):
                iou, js, mass = compare_distribution(attention, mask, probability)
                for head in range(config.model.n_head):
                    totals[layer][head][candidate_index].add(
                        weighted_iou=iou[:, head],
                        js_divergence=js[:, head],
                        preferred_edge_mass=mass[:, head],
                        valid_queries=valid_queries,
                    )

    selected, report = select_programs(totals, config)
    metadata: dict[str, object] = {
        "source_condition": "flash_baseline",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "selection_split": "program_discovery",
        "selection_mode": "top_k_flash_matched_pilot",
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
            "all_candidate_metrics": report,
        },
    )
    print(json.dumps([spec.to_json() for spec in selected], indent=2))


def main() -> None:
    run_discovery(parse_args())


if __name__ == "__main__":
    main()
