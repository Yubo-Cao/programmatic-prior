import torch

from progattn.discover import MetricTotal, compare_distribution


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
        valid_queries=valid,
    )

    assert total.query_count == 3
    assert total.weighted_iou == 8.0
