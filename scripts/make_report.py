from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, TypedDict

from progattn.config import CONDITIONS
from progattn.utils import atomic_json, json_object


class RunSummary(TypedDict):
    condition: str
    steps: int
    tokens_seen: int
    final_validation_nll: float
    best_validation_nll: float
    training_seconds: float
    median_tokens_per_second: float
    evaluation_points: int
    alpha: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json_object(json.load(handle))


def read_metrics(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            result.append(json_object(json.loads(line)))
    return result


def summarize_run(path: Path) -> RunSummary:
    completed = read_json(path / "completed.json")
    metrics = read_metrics(path / "metrics.jsonl")
    train = [row for row in metrics if row.get("event") == "train"]
    evaluations = [row for row in metrics if row.get("event") == "evaluation"]
    throughputs = [float(row["tokens_per_second"]) for row in train]
    return {
        "condition": str(completed["condition"]),
        "steps": int(completed["steps"]),
        "tokens_seen": int(completed["tokens_seen"]),
        "final_validation_nll": float(completed["final_validation_nll"]),
        "best_validation_nll": float(completed["best_validation_nll"]),
        "training_seconds": float(completed["training_seconds"]),
        "median_tokens_per_second": statistics.median(throughputs),
        "evaluation_points": len(evaluations),
        "alpha": completed["alpha"],
    }


def markdown_table(summaries: list[RunSummary]) -> str:
    lines = [
        "# Four-arm pilot report",
        "",
        (
            "All conditions use the same initialization, batch schedule, "
            "tokenizer, and 500-million-token target."
        ),
        "",
        (
            "| Condition | Tokens | Final NLL | Best NLL | "
            "Median tokens/s | Training hours |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {condition} | {tokens_seen} | {final:.6f} | {best:.6f} | "
            "{throughput:.1f} | {hours:.3f} |".format(
                condition=summary["condition"],
                tokens_seen=summary["tokens_seen"],
                final=float(summary["final_validation_nll"]),
                best=float(summary["best_validation_nll"]),
                throughput=float(summary["median_tokens_per_second"]),
                hours=float(summary["training_seconds"]) / 3600,
            )
        )
    lines.extend(
        [
            "",
            (
                "The matched arm uses programs fitted only on the reserved "
                "discovery partition. The incorrect arm preserves the exact "
                "number of preferred edges in every causal row."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    missing = [
        condition for condition in CONDITIONS if not (args.runs / condition).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing run directories: {missing}")
    summaries = [summarize_run(args.runs / condition) for condition in CONDITIONS]
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "summary.json", {"runs": summaries})
    (args.output / "report.md").write_text(markdown_table(summaries), encoding="utf-8")


if __name__ == "__main__":
    main()
