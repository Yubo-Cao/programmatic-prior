import json
from pathlib import Path

import numpy as np

from progattn.data import TokenStore, ensure_batch_schedule


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


def test_validation_stream_ignores_padding_targets(tmp_path: Path) -> None:
    root = tmp_path / "tokens"
    root.mkdir()
    np.asarray([1, 2], dtype=np.uint16).tofile(root / "train.bin")
    np.asarray([10, 11, 50256, 12, 13, 14, 50256], dtype=np.uint16).tofile(
        root / "validation.bin"
    )
    np.save(root / "validation_story_offsets.npy", np.asarray([0, 3, 7]))
    partitions = {
        "program_discovery": [],
        "protocol_calibration": [0, 1],
        "final_evaluation": [],
    }
    (root / "val_partitions.json").write_text(json.dumps(partitions))
    store = TokenStore(root)

    x, y = next(
        store.validation_stream(
            block_size=4,
            batch_size=2,
            max_tokens=8,
            pin_memory=False,
        )
    )

    assert x.tolist() == [[10, 11, 50256, 50256], [12, 13, 14, 50256]]
    assert y.tolist() == [[11, 50256, -100, -100], [13, 14, 50256, -100]]
