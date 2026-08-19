import json
from pathlib import Path

from progattn.report import summarize_run


def test_summary_uses_held_out_evaluation(tmp_path: Path) -> None:
    (tmp_path / "completed.json").write_text(
        json.dumps(
            {
                "condition": "flash_baseline",
                "steps": 3815,
                "tokens_seen": 500_039_680,
                "training_seconds": 1800.0,
                "final_validation_nll": 9.0,
                "best_validation_nll": 5.0,
                "alpha": {"values": []},
            }
        )
    )
    (tmp_path / "final_evaluation.json").write_text(
        json.dumps(
            {
                "condition": "flash_baseline",
                "partition": "final_evaluation",
                "token_count": 1_000_000,
                "validation_nll": 1.25,
                "perplexity": 3.49,
            }
        )
    )
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(
            {
                "event": "train",
                "tokens_per_second": 250_000.0,
            }
        )
        + "\n"
    )

    summary = summarize_run(tmp_path)

    assert summary["final_validation_nll"] == 1.25
    assert summary["final_perplexity"] == 3.49
    assert summary["evaluation_partition"] == "final_evaluation"
