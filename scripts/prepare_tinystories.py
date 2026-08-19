from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

EOT_TOKEN_ID = 50256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--tokenizer-threads", type=int, default=16)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(
    dataset: Dataset,
    *,
    output: Path,
    split_name: str,
    batch_size: int,
    tokenizer_threads: int,
) -> tuple[int, int]:
    token_path = output / f"{split_name}.bin"
    offset_path = output / f"{split_name}_story_offsets.npy"
    token_partial = token_path.with_suffix(".bin.partial")
    offset_partial = offset_path.with_suffix(".npy.partial")
    progress_path = output / f".{split_name}_progress.json"

    if token_path.exists() and offset_path.exists():
        offsets = np.load(offset_path, mmap_mode="r")
        return len(offsets) - 1, token_path.stat().st_size // 2

    completed_rows = 0
    token_count = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed_rows = int(progress["completed_rows"])
        token_count = int(progress["token_count"])
    else:
        token_partial.write_bytes(b"")
        np.asarray([0], dtype=np.int64).tofile(offset_partial)
        atomic_json(progress_path, {"completed_rows": 0, "token_count": 0})

    with token_partial.open("r+b") as token_file:
        token_file.truncate(token_count * np.dtype(np.uint16).itemsize)
    with offset_partial.open("r+b") as offset_file:
        offset_file.truncate((completed_rows + 1) * np.dtype(np.int64).itemsize)

    encoding = tiktoken.get_encoding("gpt2")
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
    with (
        token_partial.open("ab") as token_file,
        offset_partial.open("ab") as offset_file,
    ):
        for start in range(completed_rows, len(dataset), batch_size):
            end = min(len(dataset), start + batch_size)
            texts = dataset[start:end][text_column]
            encoded = encoding.encode_ordinary_batch(
                texts, num_threads=tokenizer_threads
            )
            flat: list[int] = []
            ending_offsets: list[int] = []
            for tokens in encoded:
                flat.extend(tokens)
                flat.append(EOT_TOKEN_ID)
                token_count += len(tokens) + 1
                ending_offsets.append(token_count)
            values = np.asarray(flat, dtype=np.uint16)
            values.tofile(token_file)
            np.asarray(ending_offsets, dtype=np.int64).tofile(offset_file)
            token_file.flush()
            offset_file.flush()
            os.fsync(token_file.fileno())
            os.fsync(offset_file.fileno())
            completed_rows = end
            atomic_json(
                progress_path,
                {"completed_rows": completed_rows, "token_count": token_count},
            )
            print(
                f"{split_name}: {completed_rows:,}/{len(dataset):,} stories, "
                f"{token_count:,} tokens",
                flush=True,
            )

    with offset_partial.open("rb") as handle:
        offsets = np.fromfile(handle, dtype=np.int64)
    temporary_npy = offset_path.with_suffix(".npy.tmp")
    with temporary_npy.open("wb") as handle:
        np.save(handle, offsets)
    temporary_npy.replace(offset_path)
    token_partial.replace(token_path)
    progress_path.unlink(missing_ok=True)
    offset_partial.unlink(missing_ok=True)
    return completed_rows, token_count


def make_partitions(story_count: int) -> dict[str, list[int]]:
    partitions: dict[str, list[int]] = {
        "program_discovery": [],
        "protocol_calibration": [],
        "final_evaluation": [],
    }
    for story_id in range(story_count):
        bucket = (
            int.from_bytes(
                hashlib.blake2b(str(story_id).encode(), digest_size=8).digest(),
                "little",
            )
            % 4
        )
        if bucket == 0:
            partitions["program_discovery"].append(story_id)
        elif bucket == 1:
            partitions["protocol_calibration"].append(story_id)
        else:
            partitions["final_evaluation"].append(story_id)
    return partitions


def validate_tokens(path: Path) -> None:
    values = np.memmap(path, dtype=np.uint16, mode="r")
    if len(values) == 0:
        raise ValueError(f"{path} is empty")
    minimum = int(np.min(values))
    maximum = int(np.max(values))
    if minimum < 0 or maximum > EOT_TOKEN_ID:
        raise ValueError(f"token range [{minimum}, {maximum}] is invalid")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    dataset = load_dataset(
        args.dataset, revision=revision, cache_dir=args.output / "cache"
    )
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError(f"dataset splits are {list(dataset)}")

    train_stories, train_tokens = prepare_split(
        dataset["train"],
        output=args.output,
        split_name="train",
        batch_size=args.batch_size,
        tokenizer_threads=args.tokenizer_threads,
    )
    validation_stories, validation_tokens = prepare_split(
        dataset["validation"],
        output=args.output,
        split_name="validation",
        batch_size=args.batch_size,
        tokenizer_threads=args.tokenizer_threads,
    )
    atomic_json(
        args.output / "val_partitions.json",
        make_partitions(validation_stories),
    )
    validate_tokens(args.output / "train.bin")
    validate_tokens(args.output / "validation.bin")
    metadata = {
        "dataset_id": args.dataset,
        "dataset_revision": revision,
        "tokenizer": "gpt2",
        "eot_token_id": EOT_TOKEN_ID,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_stories": train_stories,
        "validation_stories": validation_stories,
        "sha256_train_bin": sha256_file(args.output / "train.bin"),
        "sha256_validation_bin": sha256_file(args.output / "validation.bin"),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(args.output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
