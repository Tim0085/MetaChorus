"""CUDA/CPU environment and per-update resource measurements."""

from __future__ import annotations

import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        if sys.platform == "win32":
            return 0
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * 1024


def _nvidia_query(field: str) -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    selector = visible if visible and visible != "-1" else "0"
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
            "-i",
            selector,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or not lines:
        return None
    return lines[0].strip()


def gpu_environment_manifest(
    *,
    source_commit: str,
    source_tree: str,
    storage_root: str | Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("RTX 4090 is required but CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    if "4090" not in name:
        raise RuntimeError(f"target GPU must be RTX 4090, observed {name}")
    return {
        "schema_version": "metachorus.v53.m0c.gpu-environment.v1",
        "gpu_name": name,
        "gpu_uuid": _nvidia_query("uuid"),
        "vram_bytes": int(properties.total_memory),
        "driver_version": _nvidia_query("driver_version"),
        "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__,
        "pyg": _module_version("torch_geometric"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _system_ram_bytes(),
        "storage_root": str(Path(storage_root).resolve()),
        "storage_free_bytes": _storage_free_bytes(storage_root),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "visible_cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count_contract": 1,
        "status": (
            "PASS"
            if torch.cuda.device_count() == 1 and "4090" in name
            else "FAIL"
        ),
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _system_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return 0


def _storage_free_bytes(root: str | Path) -> int:
    return int(
        __import__("shutil").disk_usage(Path(root).resolve()).free
    )


def current_resources(device: torch.device | None = None) -> dict[str, int]:
    selected = device if device is not None else torch.device("cuda:0")
    return {
        "gpu_allocated_bytes": (
            int(torch.cuda.memory_allocated(selected))
            if torch.cuda.is_available()
            else 0
        ),
        "gpu_reserved_bytes": (
            int(torch.cuda.memory_reserved(selected))
            if torch.cuda.is_available()
            else 0
        ),
        "cpu_rss_bytes": _rss_bytes(),
    }


def percentile_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    warmup_updates: int = 20,
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("no resource rows available")
    latest_update = max(int(row["update"]) for row in materialized)
    effective_warmup_updates = min(
        int(warmup_updates),
        max(0, latest_update - 1),
    )
    values = [
        row
        for row in materialized
        if int(row["update"]) > effective_warmup_updates
    ]
    if not values:
        raise ValueError("no steady-state rows after warm-up")

    def describe(field: str) -> dict[str, float]:
        array = np.asarray([float(row[field]) for row in values])
        return {
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
        }

    return {
        "warmup_updates_excluded": effective_warmup_updates,
        "steady_state_update_count": len(values),
        "step_wall_time_seconds": describe("step_wall_time_seconds"),
        "gpu_allocated_bytes": describe("gpu_allocated_bytes"),
        "gpu_reserved_bytes": describe("gpu_reserved_bytes"),
        "cpu_rss_bytes": describe("cpu_rss_bytes"),
        "throughput_updates_per_second_p50": 1.0
        / describe("step_wall_time_seconds")["p50"],
    }
