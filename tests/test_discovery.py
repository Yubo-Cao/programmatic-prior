import torch

from progattn.discover import compare_distribution


def test_perfect_program_match_metrics() -> None:
    mask = torch.eye(8, dtype=torch.bool)
    program = mask.float()
    attention = program[None, None]

    iou, js, mass = compare_distribution(attention, mask, program)

    torch.testing.assert_close(iou, torch.ones_like(iou))
    torch.testing.assert_close(js, torch.zeros_like(js), atol=1e-6, rtol=0)
    torch.testing.assert_close(mass, torch.ones_like(mass))
