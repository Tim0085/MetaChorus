"""Four-RTX4090 hardware attestation for M8-F1."""

from __future__ import annotations

import csv
import io
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

from .common import canonical_json_hash, envelope
from .constants import EXPECTED_GPU_COUNT, EXPECTED_GPU_MODEL


def validate_gpu_rows(
    rows: Sequence[Mapping[str, Any]], *, torch_device_count: int
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        raw_memory = row.get("memory_total_mib", row.get("memory.total", ""))
        tokens = str(raw_memory).split()
        if not tokens:
            raise ValueError("M8-F1 GPU memory total is missing")
        normalized.append(
            {
                "index": int(row["index"]),
                "uuid": str(row["uuid"]).strip(),
                "name": str(row["name"]).strip(),
                "pci_bus_id": str(row.get("pci_bus_id", row.get("pci.bus_id", ""))).strip(),
                "memory_total_mib": int(tokens[0]),
                "driver_version": str(row["driver_version"]).strip(),
            }
        )
    if (
        len(normalized) != EXPECTED_GPU_COUNT
        or int(torch_device_count) != EXPECTED_GPU_COUNT
        or [row["index"] for row in normalized] != list(range(EXPECTED_GPU_COUNT))
        or len({row["uuid"] for row in normalized}) != EXPECTED_GPU_COUNT
        or len({row["pci_bus_id"] for row in normalized}) != EXPECTED_GPU_COUNT
        or {row["name"] for row in normalized} != {EXPECTED_GPU_MODEL}
        or len({row["memory_total_mib"] for row in normalized}) != 1
        or len({row["driver_version"] for row in normalized}) != 1
        or any(not row["uuid"].startswith("GPU-") for row in normalized)
    ):
        raise ValueError("M8-F1 requires four identical unique RTX 4090 devices")
    return normalized


def probe_hardware() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for values in csv.reader(io.StringIO(output)):
        if len(values) != 6:
            raise ValueError("unexpected nvidia-smi M8-F1 row")
        rows.append(
            {
                "index": values[0].strip(),
                "uuid": values[1].strip(),
                "name": values[2].strip(),
                "pci_bus_id": values[3].strip(),
                "memory_total_mib": values[4].strip().split()[0],
                "driver_version": values[5].strip(),
            }
        )
    normalized = validate_gpu_rows(rows, torch_device_count=torch.cuda.device_count())
    basis = {
        "gpu_count": EXPECTED_GPU_COUNT,
        "gpus": normalized,
        "gpu_model": EXPECTED_GPU_MODEL,
        "equal_memory_total": True,
        "unique_gpu_uuids": True,
        "unique_gpu_pci_bus_ids": True,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "one_process_per_gpu": True,
        "dynamic_four_worker_queue": True,
        "run_level_parallel_only": True,
        "worker_local_device_count_required": 1,
        "worker_local_device_required": 0,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
    }
    return envelope("metachorus.v53.m8-f1.four-gpu-attestation.v1", "PASS", basis), normalized


def validate_attestation(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    basis = value.get("basis", {})
    if (
        value.get("schema_version") != "metachorus.v53.m8-f1.four-gpu-attestation.v1"
        or value.get("status") != "PASS"
        or value.get("identity") != canonical_json_hash(basis)
        or basis.get("one_process_per_gpu") is not True
        or basis.get("dynamic_four_worker_queue") is not True
        or basis.get("run_level_parallel_only") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
    ):
        raise ValueError("M8-F1 hardware attestation envelope is invalid")
    return validate_gpu_rows(
        basis.get("gpus", []),
        torch_device_count=int(basis.get("torch_cuda_device_count", -1)),
    )


__all__ = ["validate_gpu_rows", "probe_hardware", "validate_attestation"]
