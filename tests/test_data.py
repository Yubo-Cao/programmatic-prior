from pathlib import Path

import numpy as np

from progattn.data import ensure_batch_schedule


def test_batch_schedule_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "schedule.npy"
    first = ensure_batch_schedule(
        path,
        total_steps=4,
        sequences_per_step=8,
        train_token_count=10_000,
        block_size=32,
        seed=101,
    )
    second = ensure_batch_schedule(
        path,
        total_steps=4,
        sequences_per_step=8,
        train_token_count=10_000,
        block_size=32,
        seed=999,
    )

    assert np.array_equal(first, second)
    assert first.shape == (4, 8)
