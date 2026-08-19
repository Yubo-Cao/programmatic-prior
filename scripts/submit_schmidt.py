from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def submit(script: Path, *, cwd: Path, dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script))
    return run(command, cwd=cwd).split(";")[0]


def validate(script: Path, *, cwd: Path) -> None:
    output = run(["sbatch", "--test-only", str(script)], cwd=cwd)
    print(f"{script.name}: {output}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if run(["git", "status", "--porcelain"], cwd=root):
        raise RuntimeError("refusing to submit from a dirty Git checkout")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    slurm = root / "infra" / "slurm"
    scripts = [
        slurm / "prepare_data.sbatch",
        slurm / "flash.sbatch",
        slurm / "discover.sbatch",
        slurm / "remaining.sbatch",
    ]
    if args.test_only:
        for script in scripts:
            validate(script, cwd=root)
        return

    data_job: str | None = None
    if not args.skip_data:
        data_job = submit(scripts[0], cwd=root)
        print(f"data={data_job}")
    flash_job = submit(scripts[1], cwd=root, dependency=data_job)
    print(f"flash={flash_job}")
    discovery_job = submit(scripts[2], cwd=root, dependency=flash_job)
    print(f"discovery={discovery_job}")
    remaining_job = submit(scripts[3], cwd=root, dependency=discovery_job)
    print(f"remaining={remaining_job}")


if __name__ == "__main__":
    main()
