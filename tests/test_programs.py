import torch

from progattn.programs import (
    ProgramSpec,
    ProgramType,
    dense_program_mask,
    inverse_softplus,
    program_tensors,
)


def test_softplus_initialization() -> None:
    raw = torch.tensor(inverse_softplus(0.1))

    torch.testing.assert_close(torch.nn.functional.softplus(raw), torch.tensor(0.1))


def test_incorrect_control_preserves_every_row_count() -> None:
    programs = [
        ProgramSpec(0, 0, ProgramType.PREVIOUS_K, 4),
        ProgramSpec(0, 1, ProgramType.LOCAL_WINDOW, 8),
        ProgramSpec(0, 2, ProgramType.FIRST_TOKEN),
        ProgramSpec(0, 3, ProgramType.SELF),
    ]
    types, params = program_tensors(programs, layer=0, n_head=4)
    matched = dense_program_mask(types, params, 32, layer=0)
    incorrect = dense_program_mask(types, params, 32, layer=0, incorrect=True)

    assert torch.equal(matched.sum(dim=-1), incorrect.sum(dim=-1))
    assert not torch.equal(matched[:, 1:], incorrect[:, 1:])
    assert not matched.triu(diagonal=1).any()
    assert not incorrect.triu(diagonal=1).any()
