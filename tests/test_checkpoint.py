from pathlib import Path

import torch

from progattn.checkpoint import load_checkpoint, save_checkpoint
from progattn.config import ModelConfig
from progattn.model import GPT, configure_optimizer


def checkpoint_model() -> GPT:
    config = ModelConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
        bias=True,
        dropout=0.0,
        tie_embeddings=True,
    )
    return GPT(config, condition="flash_baseline", strict_flash=False)


def test_checkpoint_restores_model_optimizer_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(101)
    model = checkpoint_model()
    optimizer = configure_optimizer(
        model,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        device=torch.device("cpu"),
    )
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    path = save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        next_step=3,
        tokens_seen=384,
        best_validation_nll=2.5,
        mark_best=True,
    )
    expected_random = torch.rand(4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)

    checkpoint = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        restore_rng=True,
    )

    assert checkpoint["next_step"] == 3
    assert (tmp_path / "last.pt").resolve() == path.resolve()
    assert (tmp_path / "best.pt").resolve() == path.resolve()
    torch.testing.assert_close(torch.rand(4), expected_random)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
