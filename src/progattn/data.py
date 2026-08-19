from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch


class TokenStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.train = np.memmap(self.root / "train.bin", dtype=np.uint16, mode="r")
        self.validation = np.memmap(
            self.root / "validation.bin", dtype=np.uint16, mode="r"
        )
        self.validation_offsets = np.load(
            self.root / "validation_story_offsets.npy", mmap_mode="r"
        )
        with (self.root / "val_partitions.json").open("r", encoding="utf-8") as handle:
            self.validation_partitions: dict[str, list[int]] = json.load(handle)

    def training_batch(
        self,
        offsets: np.ndarray,
        block_size: int,
        *,
        pin_memory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        columns = np.arange(block_size + 1, dtype=np.int64)
        tokens = np.asarray(
            self.train[offsets[:, None] + columns[None, :]], dtype=np.int64
        )
        x = torch.from_numpy(tokens[:, :-1])
        y = torch.from_numpy(tokens[:, 1:])
        if pin_memory:
            x = x.pin_memory()
            y = y.pin_memory()
        return x, y

    def validation_stream(
        self,
        *,
        block_size: int,
        batch_size: int,
        max_tokens: int,
        partition: str = "protocol_calibration",
        pin_memory: bool,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        story_ids = self.validation_partitions[partition]
        sequences: list[np.ndarray] = []
        yielded_tokens = 0
        for story_id in story_ids:
            start = int(self.validation_offsets[story_id])
            end = int(self.validation_offsets[story_id + 1])
            story = np.asarray(self.validation[start:end], dtype=np.int64)
            for position in range(0, max(1, len(story) - 1), block_size):
                chunk = story[position : position + block_size + 1]
                if len(chunk) < 2:
                    continue
                if len(chunk) < block_size + 1:
                    padded = np.full(block_size + 1, 50256, dtype=np.int64)
                    padded[: len(chunk)] = chunk
                    chunk = padded
                sequences.append(chunk)
                if len(sequences) == batch_size:
                    yield _sequence_batch(sequences, pin_memory)
                    yielded_tokens += batch_size * block_size
                    sequences = []
                    if yielded_tokens >= max_tokens:
                        return
        if sequences and yielded_tokens < max_tokens:
            yield _sequence_batch(sequences, pin_memory)

    def discovery_batches(
        self,
        *,
        block_size: int,
        batch_size: int,
        max_stories: int,
        pin_memory: bool,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        story_ids = self.validation_partitions["program_discovery"][:max_stories]
        sequences: list[np.ndarray] = []
        for story_id in story_ids:
            start = int(self.validation_offsets[story_id])
            end = int(self.validation_offsets[story_id + 1])
            story = np.asarray(self.validation[start:end], dtype=np.int64)
            chunk = story[: block_size + 1]
            if len(chunk) < 2:
                continue
            if len(chunk) < block_size + 1:
                padded = np.full(block_size + 1, 50256, dtype=np.int64)
                padded[: len(chunk)] = chunk
                chunk = padded
            sequences.append(chunk)
            if len(sequences) == batch_size:
                yield _sequence_batch(sequences, pin_memory)
                sequences = []
        if sequences:
            yield _sequence_batch(sequences, pin_memory)


def _sequence_batch(
    sequences: Sequence[np.ndarray], pin_memory: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.from_numpy(np.stack(sequences))
    x, y = batch[:, :-1], batch[:, 1:]
    if pin_memory:
        x = x.pin_memory()
        y = y.pin_memory()
    return x, y


def ensure_batch_schedule(
    path: str | Path,
    *,
    total_steps: int,
    sequences_per_step: int,
    train_token_count: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    target = Path(path)
    expected_shape = (total_steps, sequences_per_step)
    if target.exists():
        schedule = np.load(target, mmap_mode="r")
        if schedule.shape != expected_shape:
            raise ValueError(
                f"batch schedule has shape {schedule.shape}, expected {expected_shape}"
            )
        return schedule
    target.parent.mkdir(parents=True, exist_ok=True)
    maximum = train_token_count - block_size - 1
    if maximum <= 0:
        raise ValueError("training token file is smaller than one sequence")
    generator = np.random.default_rng(seed)
    schedule = generator.integers(
        0,
        maximum,
        size=expected_shape,
        dtype=np.int64,
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, schedule)
    temporary.replace(target)
    return np.load(target, mmap_mode="r")
