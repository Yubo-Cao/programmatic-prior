from __future__ import annotations

import argparse
import math
import random
import shutil
import signal
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any, cast

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import CONDITIONS, ExperimentConfig, load_config
from .data import TokenStore, ensure_batch_schedule
from .model import GPT, autocast_context, configure_optimizer
from .programs import load_programs
from .utils import append_jsonl, atomic_json, environment_manifest

_STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(config: ExperimentConfig, tokens_seen: int) -> float:
    training = config.training
    if tokens_seen < training.warmup_tokens:
        return training.learning_rate * tokens_seen / max(1, training.warmup_tokens)
    progress = (tokens_seen - training.warmup_tokens) / max(
        1, training.train_tokens - training.warmup_tokens
    )
    progress = min(1.0, max(0.0, progress))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return training.min_learning_rate + coefficient * (
        training.learning_rate - training.min_learning_rate
    )


def initial_state_path(config: ExperimentConfig) -> Path:
    return Path(config.data.initial_state_dir) / f"seed_{config.training.seed}.pt"


def schedule_path(config: ExperimentConfig) -> Path:
    return Path(config.data.schedule_dir) / f"seed_{config.training.seed}.npy"


def ensure_initial_state(model: GPT, config: ExperimentConfig) -> Path:
    path = initial_state_path(config)
    if path.exists():
        raw: Any = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or "model" not in raw:
            raise TypeError(f"invalid initial state: {path}")
        model.load_state_dict(raw["model"])
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "seed": config.training.seed}, temporary)
    temporary.replace(path)
    return path


def warmup_compile(
    forward: Callable[..., Any],
    model: GPT,
    store: TokenStore,
    schedule: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    offsets = np.asarray(schedule[0, : config.training.micro_batch_size])
    x, y = store.training_batch(
        offsets,
        config.model.block_size,
        pin_memory=device.type == "cuda",
    )
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    model.set_prior_progress(0, config.prior.warmup_tokens)
    started = time.perf_counter()
    with autocast_context(device, config.training.precision):
        result = cast(tuple[torch.Tensor, torch.Tensor | None, object], forward(x, y))
        loss = result[1]
    if loss is None:
        raise RuntimeError("compile warmup did not produce a loss")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    return elapsed


@torch.no_grad()
def evaluate(
    forward: Callable[..., Any],
    model: GPT,
    store: TokenStore,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    for x, y in store.validation_stream(
        block_size=config.model.block_size,
        batch_size=config.training.micro_batch_size,
        max_tokens=config.training.eval_tokens,
        pin_memory=device.type == "cuda",
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, config.training.precision):
            result = cast(
                tuple[torch.Tensor, torch.Tensor | None, object], forward(x, y)
            )
            loss = result[1]
        if loss is None:
            raise RuntimeError("evaluation did not produce a loss")
        tokens = y.numel()
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.train()
    return total_loss / total_tokens, time.perf_counter() - started


def _handle_stop(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _resume_path(output: Path, resume: str) -> Path | None:
    if resume == "never":
        return None
    if resume == "auto":
        candidate = output / "checkpoints" / "last.pt"
        return candidate if candidate.exists() else None
    candidate = Path(resume)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def run() -> int:
    args = parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = (args.data_root or Path(config.data.root)).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    programs = []
    if args.condition in {"matched_program_prior", "incorrect_program_prior"}:
        programs = load_programs(config.prior.selected_programs)
        if not programs:
            raise RuntimeError("the prior condition requires selected programs")

    seed_everything(config.training.seed)
    model = GPT(
        config.model,
        condition=args.condition,
        programs=programs,
        initial_alpha=config.prior.initial_alpha,
        control_seed=config.prior.control_seed,
        strict_flash=config.training.strict_flash,
    )
    initial_path = ensure_initial_state(model, config)
    raw_initial: Any = torch.load(initial_path, map_location="cpu", weights_only=True)
    model.load_state_dict(raw_initial["model"])
    model.to(device)
    model.prepare_attention(device)

    store = TokenStore(data_root)
    schedule = ensure_batch_schedule(
        schedule_path(config),
        total_steps=config.total_steps,
        sequences_per_step=config.sequences_per_step,
        train_token_count=len(store.train),
        block_size=config.model.block_size,
        seed=config.training.seed,
    )
    forward: Callable[..., Any] = model
    if config.training.compile:
        forward = torch.compile(model, dynamic=False)
    compile_seconds = warmup_compile(forward, model, store, schedule, config, device)
    model.load_state_dict(raw_initial["model"])
    seed_everything(config.training.seed)

    optimizer = configure_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
        device=device,
    )
    start_step = 0
    tokens_seen = 0
    best_validation_nll = math.inf
    resume_path = _resume_path(output, args.resume)
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            restore_rng=True,
        )
        start_step = int(checkpoint["next_step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        best_validation_nll = float(checkpoint["best_validation_nll"])

    shutil.copy2(config.source_path, output / "config.yaml")
    atomic_json(output / "environment.json", environment_manifest(device))
    atomic_json(
        output / "run.json",
        {
            "condition": args.condition,
            "seed": config.training.seed,
            "train_tokens_requested": config.training.train_tokens,
            "global_tokens_per_step": config.training.global_tokens_per_step,
            "total_steps": config.total_steps,
            "parameter_count": model.parameter_count(),
            "initial_state": str(initial_path.resolve()),
            "batch_schedule": str(schedule_path(config).resolve()),
            "programs": str(Path(config.prior.selected_programs).resolve())
            if programs
            else None,
            "compile_seconds": compile_seconds,
        },
    )

    signal.signal(signal.SIGUSR1, _handle_stop)
    total_steps = config.total_steps
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    model.train()
    training_seconds = 0.0
    latest_validation_nll = math.inf
    for step in range(start_step, total_steps):
        current_lr = learning_rate(config, tokens_seen)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        model.set_prior_progress(tokens_seen, config.prior.warmup_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        offsets = np.asarray(schedule[step])
        x_all, y_all = store.training_batch(
            offsets,
            config.model.block_size,
            pin_memory=device.type == "cuda",
        )
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        micro = config.training.micro_batch_size
        for accumulation in range(config.accumulation_steps):
            start = accumulation * micro
            end = start + micro
            x = x_all[start:end].to(device, non_blocking=True)
            y = y_all[start:end].to(device, non_blocking=True)
            with autocast_context(device, config.training.precision):
                result = cast(
                    tuple[torch.Tensor, torch.Tensor | None, object], forward(x, y)
                )
                loss = result[1]
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite loss at step {step}")
            (loss / config.accumulation_steps).backward()
            step_loss += float(loss.detach().item()) / config.accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.grad_clip
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - started
        training_seconds += step_seconds
        tokens_seen = (step + 1) * config.training.global_tokens_per_step
        record: dict[str, object] = {
            "event": "train",
            "step": step + 1,
            "tokens_seen": tokens_seen,
            "loss": step_loss,
            "learning_rate": current_lr,
            "grad_norm": float(grad_norm.item()),
            "step_seconds": step_seconds,
            "training_seconds": training_seconds,
            "tokens_per_second": config.training.global_tokens_per_step / step_seconds,
            "alpha": model.alpha_summary(),
        }
        if device.type == "cuda":
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        append_jsonl(output / "metrics.jsonl", record)

        evaluated = (step + 1) % config.training.eval_every_steps == 0
        if evaluated or step + 1 == total_steps:
            latest_validation_nll, evaluation_seconds = evaluate(
                forward, model, store, config, device
            )
            append_jsonl(
                output / "metrics.jsonl",
                {
                    "event": "evaluation",
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "validation_nll": latest_validation_nll,
                    "perplexity": math.exp(min(20.0, latest_validation_nll)),
                    "evaluation_seconds": evaluation_seconds,
                },
            )

        mark_best = latest_validation_nll < best_validation_nll
        if mark_best:
            best_validation_nll = latest_validation_nll
        checkpoint_due = (
            (step + 1) % config.training.checkpoint_every_steps == 0
            or step + 1 == total_steps
            or _STOP_REQUESTED
        )
        if checkpoint_due:
            save_checkpoint(
                output / "checkpoints",
                model=model,
                optimizer=optimizer,
                next_step=step + 1,
                tokens_seen=tokens_seen,
                best_validation_nll=best_validation_nll,
                mark_best=mark_best,
            )
        if _STOP_REQUESTED:
            return 99

    atomic_json(
        output / "completed.json",
        {
            "condition": args.condition,
            "steps": total_steps,
            "tokens_seen": tokens_seen,
            "training_seconds": training_seconds,
            "final_validation_nll": latest_validation_nll,
            "best_validation_nll": best_validation_nll,
            "alpha": model.alpha_summary(),
        },
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
