import json
from pathlib import Path

from progattn.config import load_config
from progattn.train import learning_rate, prepare_metrics_for_resume


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
