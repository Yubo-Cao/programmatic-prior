from collections.abc import Callable

import torch

from progattn.config import ExperimentConfig, load_config
from progattn.discover import (
    MetricTotal,
    choose_programs,
    compare_distribution,
    head_metrics,
    program_distributions,
    selection_score,
)
from progattn.programs import ProgramType, candidate_programs


def test_perfect_program_match_metrics() -> None:
    mask = torch.eye(8, dtype=torch.bool)
    program = mask.float()
    attention = program[None, None]

    iou, js, mass = compare_distribution(attention, mask, program)

    torch.testing.assert_close(iou, torch.ones_like(iou))
    torch.testing.assert_close(js, torch.zeros_like(js), atol=1e-6, rtol=0)
    torch.testing.assert_close(mass, torch.ones_like(mass))


def test_metric_total_weights_each_batch_once() -> None:
    total = MetricTotal()
    metric = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    valid = torch.tensor([[True, False], [True, True]])

    total.add(
        weighted_iou=metric,
        js_divergence=metric,
        preferred_edge_mass=metric,
        uniform_edge_mass=metric,
        valid_queries=valid,
    )

    assert total.query_count == 3
    assert total.weighted_iou == 8.0
    assert total.uniform_edge_mass == 8.0


def test_selection_score_is_zero_when_the_head_reads_uniformly() -> None:
    assert selection_score(0.25, 0.25) == 0.0
    assert selection_score(0.5, 0.25) > 0.0
    assert selection_score(0.1, 0.25) < 0.0


def test_uniform_mass_matches_the_share_of_the_causal_row() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")
    distributions = program_distributions(config, torch.device("cpu"))
    index = candidate_programs().index((ProgramType.LOCAL_WINDOW, 8))
    _, _, uniform = distributions[index]

    # Before the window fills, the program covers the whole causal prefix.
    torch.testing.assert_close(uniform[:8], torch.ones(8))
    # After it fills, it covers exactly eight of the q + 1 visible keys.
    torch.testing.assert_close(uniform[63], torch.tensor(8.0 / 64.0))


Assign = Callable[[int, int, tuple[ProgramType, int]], tuple[float, float, float]]


def _totals(config: ExperimentConfig, assign: Assign) -> list[list[list[MetricTotal]]]:
    candidates = candidate_programs()
    totals = [
        [[MetricTotal() for _ in candidates] for _ in range(config.model.n_head)]
        for _ in range(config.model.n_layer)
    ]
    for layer in range(config.model.n_layer):
        for head in range(config.model.n_head):
            for index, candidate in enumerate(candidates):
                mass, uniform, iou = assign(layer, head, candidate)
                total = totals[layer][head][index]
                total.query_count = 1
                total.preferred_edge_mass = mass
                total.uniform_edge_mass = uniform
                total.weighted_iou = iou
    return totals


def test_enrichment_prefers_a_sharp_head_that_weighted_iou_ranks_last() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")

    def assign(
        layer: int, head: int, candidate: tuple[ProgramType, int]
    ) -> tuple[float, float, float]:
        # (preferred mass, uniform mass, weighted_iou)
        program_type, parameter = candidate
        wide = program_type is ProgramType.LOCAL_WINDOW and parameter == 64
        if (layer, head) == (0, 0):
            if program_type is ProgramType.PREVIOUS_K and parameter == 1:
                # A previous-token head. Nearly all of its mass sits inside the wide
                # window too, but concentrated on one key, and weighted_iou caps the
                # overlap at the program's own uniform height - so the sharper the
                # head, the worse it scores against a wide window.
                return (0.60, 0.02, 0.43)
            return (0.98, 0.40, 0.06) if wide else (0.05, 0.05, 0.02)
        # Every other head reads diffusely, which is what a wide window rewards.
        return (0.95, 0.40, 0.55) if wide else (0.10, 0.05, 0.05)

    totals = _totals(config, assign)
    entries = head_metrics(totals, config)

    by_enrichment = choose_programs(entries, config, "enrichment")
    by_iou = choose_programs(entries, config, "weighted_iou")

    top = by_enrichment[0]
    assert (top.layer, top.head) == (0, 0)
    assert top.program_type is ProgramType.PREVIOUS_K
    assert top.parameter == 1
    assert all(spec.program_type is ProgramType.LOCAL_WINDOW for spec in by_iou)
    assert (0, 0) not in {(spec.layer, spec.head) for spec in by_iou}


def test_both_criteria_respect_the_per_layer_cap() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")

    def assign(
        layer: int, head: int, candidate: tuple[ProgramType, int]
    ) -> tuple[float, float, float]:
        return (0.9 - 0.001 * layer, 0.3, 0.9 - 0.001 * layer)

    totals = _totals(config, assign)
    entries = head_metrics(totals, config)

    for criterion in ("enrichment", "weighted_iou"):
        selected = choose_programs(entries, config, criterion)
        assert len(selected) == config.prior.selection_count
        counts: dict[int, int] = {}
        for spec in selected:
            counts[spec.layer] = counts.get(spec.layer, 0) + 1
        assert max(counts.values()) <= config.prior.max_per_layer
