from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def append_jsonl(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def environment_manifest(device: torch.device) -> dict[str, object]:
    commit, dirty = git_revision()
    gpu_name: str | None = None
    compute_capability: str | None = None
    peak_driver: str | None = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = f"{major}.{minor}"
        try:
            peak_driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    "--id=0",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            peak_driver = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": peak_driver,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_compute_capability": compute_capability,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("expected an object with string keys")
    return value
