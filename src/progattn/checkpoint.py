from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import GPT
from .utils import json_object


def rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    directory: str | Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    tokens_seen: int,
    best_validation_nll: float,
    mark_best: bool,
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{next_step:06d}.pt"
    temporary = target.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_step": next_step,
            "tokens_seen": tokens_seen,
            "best_validation_nll": best_validation_nll,
            "rng": rng_state(),
        },
        temporary,
    )
    temporary.replace(target)
    _replace_symlink(root / "last.pt", target.name)
    if mark_best:
        _replace_symlink(root / "best.pt", target.name)
    return target


def _replace_symlink(path: Path, target_name: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target_name)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    raw: Any = torch.load(Path(path), map_location="cpu", weights_only=False)
    checkpoint = json_object(raw)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        restore_rng_state(json_object(checkpoint["rng"]))
    return checkpoint
