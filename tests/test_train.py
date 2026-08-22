import json
from pathlib import Path

import torch
import torch.nn.functional as F

from progattn.config import load_config
from progattn.programs import inverse_softplus
from progattn.train import (
    apply_initial_alpha,
    learning_rate,
    prepare_metrics_for_resume,
)


def test_learning_rate_schedule() -> None:
    config = load_config("configs/pilot_gpt2_small.yaml")

    assert learning_rate(config, 0) == 0.0
    assert learning_rate(config, config.training.warmup_tokens) == 0.0005
    assert learning_rate(config, config.training.train_tokens) == 0.00005


def test_prepare_metrics_for_resume_discards_uncheckpointed_steps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    records = [
        {"event": "train", "step": 1, "training_seconds": 1.25},
        {"event": "evaluation", "step": 1},
        {"event": "train", "step": 2, "training_seconds": 2.75},
        {"event": "train", "step": 3, "training_seconds": 4.0},
    ]
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )

    elapsed = prepare_metrics_for_resume(path, completed_steps=2)

    retained = [json.loads(line) for line in path.read_text().splitlines()]
    assert retained == records[:3]
    assert elapsed == 2.75


def test_initial_state_cannot_pin_the_prior_strength() -> None:
    """A state frozen at one alpha must not override the configured one.

    The frozen initial state exists to make every arm start from identical weights,
    and it carries raw_alpha along with them. The first pilot froze it at alpha=0.1,
    so any later run reusing it would silently train at 0.1 no matter what the config
    asked for - which would make the prior strength unadjustable without discarding
    the shared initialization the paired protocol depends on.
    """
    config = load_config("configs/pilot_gpt2_small_v2.yaml")
    stale = inverse_softplus(0.1)
    state = {
        "blocks.0.attn.raw_alpha": torch.full((12,), stale),
        "blocks.1.attn.raw_alpha": torch.full((12,), stale),
        "blocks.0.attn.qkv.weight": torch.zeros(3, 3),
    }

    apply_initial_alpha(state, config)

    for name, tensor in state.items():
        if name.endswith("raw_alpha"):
            torch.testing.assert_close(
                F.softplus(tensor),
                torch.full_like(tensor, config.prior.initial_alpha),
            )
    # Weights that are not the prior strength stay untouched.
    torch.testing.assert_close(state["blocks.0.attn.qkv.weight"], torch.zeros(3, 3))
