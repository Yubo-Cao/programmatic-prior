from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from progattn.config import (
    CONDITIONS,
    PRIOR_CONDITIONS,
    ExperimentConfig,
    load_config,
    prior_programs_path,
)
from progattn.data import TokenStore
from progattn.model import GPT
from progattn.programs import ProgramSpec, load_programs
from progattn.train import evaluate
from progattn.utils import atomic_json, json_object, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--partition", default="final_evaluation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def programs_for_condition(
    condition: str, config: ExperimentConfig
) -> list[ProgramSpec]:
    if condition in PRIOR_CONDITIONS:
        return load_programs(prior_programs_path(condition, config))
    return []


def evaluate_condition(
    *,
    condition: str,
    runs: Path,
    store: TokenStore,
    config: ExperimentConfig,
    device: torch.device,
    partition: str,
) -> dict[str, object]:
    run = runs / condition
    checkpoint_path = run / "checkpoints" / "last.pt"
    raw: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = json_object(raw)
    model = GPT(
        config.model,
        condition=condition,
        programs=programs_for_condition(condition, config),
        initial_alpha=config.prior.initial_alpha,
        control_seed=config.prior.control_seed,
        strict_flash=config.training.strict_flash,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.prepare_attention(device)
    model.set_prior_progress(int(checkpoint["tokens_seen"]), config.prior.warmup_tokens)
    forward: Callable[..., Any] = model
    if config.training.compile:
        forward = torch.compile(model, dynamic=False)
    nll, seconds, token_count = evaluate(
        forward,
        model,
        store,
        config,
        device,
        partition=partition,
    )
    payload: dict[str, object] = {
        "condition": condition,
        "partition": partition,
        "token_count": token_count,
        "validation_nll": nll,
        "perplexity": math.exp(min(20.0, nll)),
        "evaluation_seconds": seconds,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_tokens_seen": int(checkpoint["tokens_seen"]),
    }
    atomic_json(run / "final_evaluation.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = (args.data_root or Path(config.data.root)).resolve()
    store = TokenStore(data_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    summaries: list[dict[str, object]] = []
    for condition in CONDITIONS:
        summaries.append(
            evaluate_condition(
                condition=condition,
                runs=args.runs,
                store=store,
                config=config,
                device=device,
                partition=args.partition,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
