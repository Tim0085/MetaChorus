"""Complete, atomic O0 checkpoint and RNG persistence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch

from metachorus_v53.audit.common import sha256_file, utc_now


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
        "cudnn_deterministic": bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        ),
        "cudnn_benchmark": bool(
            getattr(torch.backends.cudnn, "benchmark", False)
        ),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = list(state.get("torch_cuda_all", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in cuda_states]
        )
    torch.backends.cudnn.deterministic = bool(
        state["cudnn_deterministic"]
    )
    torch.backends.cudnn.benchmark = bool(state["cudnn_benchmark"])
    torch.use_deterministic_algorithms(
        bool(state["deterministic_algorithms"])
    )


def rng_state_sha256(state: Mapping[str, Any] | None = None) -> str:
    buffer = io.BytesIO()
    torch.save(dict(state or capture_rng_state()), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def nested_state_sha256(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _atomic_torch_save(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


@dataclass(frozen=True)
class CheckpointReceipt:
    path: Path
    update: int
    size_bytes: int
    sha256: str
    save_wall_time_seconds: float
    hash_wall_time_seconds: float


def save_checkpoint(
    destination: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    exact_update: int,
    metadata: Mapping[str, Any],
    scheduler: Any | None = None,
) -> CheckpointReceipt:
    path = Path(destination)
    rng_state = capture_rng_state()
    payload = {
        "schema_version": "metachorus.v53.o0-complete-checkpoint.v1",
        "created_at_utc": utc_now(),
        "exact_update": int(exact_update),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "rng_state": rng_state,
        "rng_state_sha256": rng_state_sha256(rng_state),
        "metadata": dict(metadata),
        "contains_evaluation_information": False,
        "contains_labels": False,
        "contains_class_count": False,
    }
    started = time.perf_counter()
    _atomic_torch_save(path, payload)
    save_elapsed = time.perf_counter() - started
    hash_started = time.perf_counter()
    digest = sha256_file(path)
    hash_elapsed = time.perf_counter() - hash_started
    return CheckpointReceipt(
        path=path,
        update=int(exact_update),
        size_bytes=path.stat().st_size,
        sha256=digest,
        save_wall_time_seconds=save_elapsed,
        hash_wall_time_seconds=hash_elapsed,
    )


def load_checkpoint(
    source: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = torch.load(
        Path(source),
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("schema_version") != (
        "metachorus.v53.o0-complete-checkpoint.v1"
    ):
        raise ValueError("unsupported checkpoint schema")
    if payload.get("contains_evaluation_information") is not False:
        raise ValueError("checkpoint contains evaluation information")
    if payload.get("contains_labels") is not False:
        raise ValueError("checkpoint contains labels")
    for key, expected in dict(expected_metadata or {}).items():
        if payload["metadata"].get(key) != expected:
            raise ValueError(
                f"checkpoint metadata mismatch for {key}: "
                f"{payload['metadata'].get(key)!r} != {expected!r}"
            )
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _optimizer_to_device(optimizer, device)
    if scheduler is None:
        if payload["scheduler_state"] is not None:
            raise ValueError("checkpoint has a scheduler but runner does not")
    else:
        scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"])
    observed_rng_hash = rng_state_sha256()
    if observed_rng_hash != payload["rng_state_sha256"]:
        raise RuntimeError("RNG state did not round-trip")
    return {
        "payload": payload,
        "load_wall_time_seconds": time.perf_counter() - started,
        "rng_state_roundtrip": True,
    }
