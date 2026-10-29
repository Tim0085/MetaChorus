"""Four-worker dynamic queue over the frozen M6F/M7F O0 executor."""

from __future__ import annotations

import json
import os
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from metachorus_v53.audit.common import canonical_json_hash
from metachorus_v53.m6f_m7f_fast.training import (
    build_run_config as build_frozen_run_config,
    run_plan_row as run_frozen_plan_row,
)
from metachorus_v53.training.o0_codefaithful_runner import O0TrainingSession

from .common import envelope, read_csv, read_json, sha256_file, write_json_once
from .constants import (
    COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
    EXPECTED_GPU_COUNT,
    MAX_ATTEMPTS_PER_IDENTITY,
)
from .planning import STAGE_STATUS, canonical_plan_row, validate_plan


ENGINE_STAGE = {
    "CANARY": "CANARY",
    "R1": "SEARCH",
    "R2": "SEARCH",
    "R3": "SEARCH",
    "R4": "SEARCH",
    "PRELIMINARY": "PRELIMINARY",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def h3_marker_name(logical_stage: str) -> str:
    if logical_stage not in ENGINE_STAGE:
        raise ValueError(f"unsupported M7F-H3 stage: {logical_stage}")
    return f"DBLP_H3_{logical_stage}_COMPLETE.json"


def _branch(logical_stage: str) -> str:
    if logical_stage == "CANARY":
        return "canary_runs"
    if logical_stage == "PRELIMINARY":
        return "preliminary_runs"
    return "search_runs"


def attempt_root(result_root: str | Path, row: Mapping[str, Any], logical_stage: str, attempt: int) -> Path:
    if int(attempt) not in range(MAX_ATTEMPTS_PER_IDENTITY):
        raise ValueError("M7F-H3 permits only attempt 0 and one engineering retry")
    return (
        Path(result_root).resolve()
        / _branch(logical_stage)
        / str(row["physical_run_id"])
        / f"attempt_{int(attempt):02d}"
    )


def completed_marker(
    result_root: str | Path, row: Mapping[str, Any], logical_stage: str
) -> tuple[Path, dict[str, Any]] | None:
    parent = Path(result_root).resolve() / _branch(logical_stage) / str(row["physical_run_id"])
    matches = [
        attempt / h3_marker_name(logical_stage)
        for attempt in sorted(parent.glob("attempt_*"))
        if (attempt / h3_marker_name(logical_stage)).is_file()
        and not (attempt / h3_marker_name(logical_stage)).is_symlink()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple M7F-H3 completion markers: {row['physical_run_id']} {logical_stage}")
    if not matches:
        return None
    marker = read_json(matches[0])
    basis = marker.get("basis", {})
    try:
        marker_attempt = int(matches[0].parent.name.removeprefix("attempt_"))
    except ValueError as exc:
        raise ValueError("M7F-H3 completion marker attempt directory is malformed") from exc
    if (
        marker.get("schema_version") != "metachorus.v53.m7f-h3.run-complete.v1"
        or marker.get("status") != "PASS"
        or marker.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or basis.get("plan_seal_identity") != row.get("plan_seal_identity")
        or int(basis.get("attempt_index", -1)) != marker_attempt
        or marker_attempt not in range(MAX_ATTEMPTS_PER_IDENTITY)
        or int(basis.get("worker_index", -1)) not in range(EXPECTED_GPU_COUNT)
        or int(basis.get("continuation_from_update", -1))
        != int(row.get("continuation_from_update", -2))
        or int(basis.get("endpoint_update", -1)) != int(row.get("endpoint_update", -2))
        or int(basis.get("parent_endpoint", -1))
        != int(row.get("continuation_from_update", -2))
        or int(basis.get("child_endpoint", -1)) != int(row.get("endpoint_update", -2))
        or int(basis.get("incremental_optimizer_steps", -1))
        != int(row.get("endpoint_update", -2)) - int(row.get("continuation_from_update", -2))
        or not str(basis.get("engine_marker_identity", ""))
        or not str(basis.get("gpu_uuid", "")).startswith("GPU-")
        or not str(basis.get("gpu_pci_bus_id", ""))
        or not str(basis.get("runtime_domain", ""))
        or basis.get("labels_mounted_during_training") is not False
        or basis.get("one_process_per_gpu") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
    ):
        raise ValueError("M7F-H3 completion marker identity is invalid")
    return matches[0], marker


def _gpu_runtime(*, enforce_target_gpu: bool) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid = os.environ.get("DBLP_H3_GPU_UUID", "")
    pci_bus_id = os.environ.get("DBLP_H3_GPU_PCI_BUS_ID", "")
    runtime_domain = os.environ.get("DBLP_H3_RUNTIME_DOMAIN", socket.gethostname())
    if enforce_target_gpu:
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError("M7F-H3 worker must see exactly one local CUDA device 0")
        name = torch.cuda.get_device_name(0)
        if not uuid or not pci_bus_id or not visible:
            raise RuntimeError("M7F-H3 worker lacks frozen GPU UUID/PCI/visibility metadata")
    else:
        name = os.environ.get("DBLP_H3_GPU_NAME", "CPU_TEST_DOUBLE")
        uuid = uuid or "GPU-CPU-TEST-DOUBLE"
        pci_bus_id = pci_bus_id or "0000:00:00.0"
        visible = visible or "CPU_TEST_DOUBLE"
    return {
        "gpu_uuid": uuid,
        "gpu_pci_bus_id": pci_bus_id,
        "gpu_name": name,
        "cuda_visible_devices": visible,
        "runtime_domain": runtime_domain,
        "torch_cuda_device_count_inside_worker": int(torch.cuda.device_count()) if enforce_target_gpu else 1,
        "torch_current_local_device": int(torch.cuda.current_device()) if enforce_target_gpu else 0,
    }


def validate_engine_marker(
    marker: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    *,
    logical_stage: str,
    expected_attempt_index: int | None = None,
) -> dict[str, Any]:
    """Bind a frozen-engine marker to an H3 row, including fresh R2 seeds."""

    row = canonical_plan_row(plan_row)
    basis = marker.get("basis", {})
    attempt = int(basis.get("attempt_index", -1))
    if (
        marker.get("schema_version") != "metachorus.v53.m6f-m7f.fast-run-complete.v1"
        or marker.get("status") != "PASS"
        or marker.get("identity") != canonical_json_hash(basis)
        or basis.get("stage") != ENGINE_STAGE[logical_stage]
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or basis.get("dataset") != row.get("dataset")
        or basis.get("candidate_set_id") != row.get("candidate_set_id")
        or basis.get("candidate_set_hash") != row.get("candidate_set_hash")
        or int(basis.get("candidate_set_ordinal", -1)) != int(row.get("candidate_set_ordinal", -2))
        or list(basis.get("path_ids", [])) != list(row.get("path_ids", []))
        or int(basis.get("model_seed", -1)) != int(row.get("model_seed", -2))
        or int(basis.get("continuation_from_update", -1)) != int(row.get("continuation_from_update", -2))
        or int(basis.get("endpoint_update", -1)) != int(row.get("endpoint_update", -2))
        or int(basis.get("optimizer_steps", -1)) != int(row.get("endpoint_update", -2))
        or attempt not in range(MAX_ATTEMPTS_PER_IDENTITY)
        or (expected_attempt_index is not None and attempt != int(expected_attempt_index))
        or basis.get("labels_mounted") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
    ):
        raise ValueError("frozen engine marker does not match the M7F-H3 plan identity")
    return dict(basis)


def _run_fresh_r2_seed(
    *,
    row: Mapping[str, Any],
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool,
) -> dict[str, Any]:
    """Run an H3-only fresh 0→1000 search identity with the frozen O0 config."""

    attempt = int(row.get("attempt_index", 0))
    root = attempt_root(result_root, row, "R2", attempt)
    marker_path = root / "M6F_M7F_SEARCH_001000_COMPLETE.json"
    if marker_path.is_file() and not marker_path.is_symlink():
        marker = read_json(marker_path)
        validate_engine_marker(marker, row, logical_stage="R2", expected_attempt_index=attempt)
        return marker
    if root.exists():
        raise FileExistsError(f"fresh M7F-H3 R2 attempt is not empty: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    config_row = {
        **dict(row),
        "stage": "R1_ALL_SETS_0000_TO_0300",
        "endpoint_update": 300,
        "continuation_from_update": 0,
    }
    config = build_frozen_run_config(
        plan_row=config_row,
        stage="SEARCH",
        cache_root=cache_root,
        output_root=root,
        source_commit=source_commit,
        source_tree=source_tree,
    )
    session = O0TrainingSession(config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu)
    run = session.run(stop_at_update=1000)
    if (
        run.get("labels_mounted") is not False
        or int(run.get("downstream_evaluations", -1)) != 0
        or int(run.get("optimizer_steps", -1)) != 1000
    ):
        raise RuntimeError("fresh M7F-H3 R2 run violated labels/evaluation/update boundaries")
    write_json_once(root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    result = Path(result_root).resolve()
    checkpoint = root / "checkpoints" / "update_001000.pt"
    representation = root / "representations" / "update_001000.npy"
    if checkpoint.is_symlink() or representation.is_symlink() or not checkpoint.is_file() or not representation.is_file():
        raise RuntimeError("fresh M7F-H3 R2 endpoint capture is incomplete")
    basis = {
        "stage": "SEARCH",
        "physical_run_id": row["physical_run_id"],
        "dataset": row["dataset"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "path_ids": list(row["path_ids"]),
        "model_seed": int(row["model_seed"]),
        "attempt_index": attempt,
        "continuation_from_update": 0,
        "endpoint_update": 1000,
        "optimizer_steps": int(run["optimizer_steps"]),
        "checkpoint_relative_path": checkpoint.relative_to(result).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "representation_relative_path": representation.relative_to(result).as_posix(),
        "representation_sha256": sha256_file(representation),
        "model_total_seconds": float(run.get("model_total_seconds", 0.0)),
        "peak_gpu_allocated_bytes": int(run.get("peak_gpu_allocated_bytes", 0)),
        "peak_gpu_reserved_bytes": int(run.get("peak_gpu_reserved_bytes", 0)),
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "search_checkpoint_reused": False,
        "preliminary_started_from_zero": False,
        "run_identity": config.identity(),
    }
    marker = envelope("metachorus.v53.m6f-m7f.fast-run-complete.v1", "PASS", basis)
    write_json_once(marker_path, marker)
    return marker


def run_h3_row(
    *,
    plan_row: Mapping[str, Any],
    logical_stage: str,
    worker_index: int,
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    row = canonical_plan_row(plan_row)
    if logical_stage not in ENGINE_STAGE or row.get("logical_stage") != logical_stage:
        raise ValueError("M7F-H3 row/logical stage mismatch")
    attempt = int(row.get("attempt_index", 0))
    if attempt not in range(MAX_ATTEMPTS_PER_IDENTITY):
        raise ValueError("M7F-H3 permits at most one engineering retry per scientific identity")
    prior = completed_marker(result_root, row, logical_stage)
    if prior is not None:
        return prior[1]
    retry_started = row.get("retry_invocation_started_at")
    started = (
        datetime.fromisoformat(str(retry_started).replace("Z", "+00:00"))
        if retry_started
        else datetime.now(timezone.utc)
    )
    started_at = started.isoformat()
    gpu = _gpu_runtime(enforce_target_gpu=enforce_target_gpu)
    if logical_stage == "R2" and int(row.get("continuation_from_update", 0)) == 0:
        engine_marker = _run_fresh_r2_seed(
            row=row,
            repo_root=repo_root,
            cache_root=cache_root,
            result_root=result_root,
            source_commit=source_commit,
            source_tree=source_tree,
            enforce_target_gpu=enforce_target_gpu,
        )
    else:
        engine_marker = run_frozen_plan_row(
            plan_row=row,
            stage=ENGINE_STAGE[logical_stage],
            repo_root=repo_root,
            cache_root=cache_root,
            result_root=result_root,
            source_commit=source_commit,
            source_tree=source_tree,
            enforce_target_gpu=enforce_target_gpu,
        )
    root = attempt_root(result_root, row, logical_stage, attempt)
    engine_name = {
        "CANARY": f"M6F_M7F_CANARY_{int(row['endpoint_update']):06d}_COMPLETE.json",
        "SEARCH": f"M6F_M7F_SEARCH_{int(row['endpoint_update']):06d}_COMPLETE.json",
        "PRELIMINARY": f"M6F_M7F_PRELIMINARY_{int(row['endpoint_update']):06d}_COMPLETE.json",
    }[ENGINE_STAGE[logical_stage]]
    engine_path = root / engine_name
    if engine_path.is_symlink() or not engine_path.is_file() or engine_marker.get("status") != "PASS":
        raise RuntimeError("frozen engine did not publish its governed completion marker")
    validate_engine_marker(
        engine_marker,
        row,
        logical_stage=logical_stage,
        expected_attempt_index=attempt,
    )
    ended = datetime.now(timezone.utc)
    basis = {
        "logical_stage": logical_stage,
        "engine_stage": ENGINE_STAGE[logical_stage],
        "physical_run_id": row["physical_run_id"],
        "dataset": row["dataset"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "path_ids": list(row["path_ids"]),
        "model_seed": int(row["model_seed"]),
        "attempt_index": attempt,
        "continuation_from_update": int(row.get("continuation_from_update", 0)),
        "endpoint_update": int(row["endpoint_update"]),
        "plan_seal_identity": row["plan_seal_identity"],
        "engine_marker_identity": engine_marker["identity"],
        "engine_marker_relative_path": engine_path.resolve(strict=True).relative_to(
            Path(result_root).resolve()
        ).as_posix(),
        "engine_marker_sha256": sha256_file(engine_path),
        "worker_index": int(worker_index),
        **gpu,
        "started_at": started_at,
        "ended_at": ended.isoformat(),
        "invocation_started_at": started_at,
        "invocation_ended_at": ended.isoformat(),
        "invocation_wall_seconds": (ended - started).total_seconds(),
        "parent_endpoint": int(row.get("continuation_from_update", 0)),
        "child_endpoint": int(row["endpoint_update"]),
        "incremental_optimizer_steps": int(row["endpoint_update"]) - int(row.get("continuation_from_update", 0)),
        "engineering_retry_of_attempt": row.get("engineering_retry_of_attempt", ""),
        "engineering_retry_replayed_from_zero": bool(row.get("engineering_retry_replayed_from_zero", False)),
        "fresh_r2_frozen_config_adapter": (
            logical_stage == "R2" and int(row.get("continuation_from_update", 0)) == 0
        ),
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
        "labels_mounted_during_training": False,
        "downstream_evaluations_during_training": 0,
    }
    marker = envelope("metachorus.v53.m7f-h3.run-complete.v1", "PASS", basis)
    write_json_once(root / h3_marker_name(logical_stage), marker)
    return marker


def _atomic_claim(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        data = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _failure_fingerprint(exc: BaseException) -> str:
    return canonical_json_hash(
        {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "last_traceback_line": traceback.format_exc().strip().splitlines()[-1],
        }
    )


def _failures(queue_root: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted((queue_root / "failures").glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("M7F-H3 failure registry contains a non-regular member")
        values.append(read_json(path))
    return values


def _validated_failure(
    queue_root: Path,
    row: Mapping[str, Any],
    *,
    logical_stage: str,
    plan_seal_identity: str,
    attempt: int,
) -> dict[str, Any] | None:
    path = queue_root / "failures" / f"{row['physical_run_id']}__attempt_{attempt:02d}.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("M7F-H3 failure evidence is not a regular file")
    failure = read_json(path)
    basis = failure.get("basis", {})
    if (
        failure.get("schema_version") != "metachorus.v53.m7f-h3.run-failure.v1"
        or failure.get("status") != "FAILED_PRESERVED"
        or failure.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or int(basis.get("attempt_index", -1)) != int(attempt)
        or basis.get("plan_seal_identity") != plan_seal_identity
        or basis.get("scientific_metric_triggered_retry") is not False
        or basis.get("retry_allowed") != (attempt + 1 < MAX_ATTEMPTS_PER_IDENTITY)
    ):
        raise ValueError("M7F-H3 failure evidence identity is invalid")
    return failure


def _execution_row_for_retry(
    queue_root: Path,
    row: Mapping[str, Any],
    *,
    logical_stage: str,
    plan_seal_identity: str,
) -> dict[str, Any] | None:
    value = dict(row)
    attempt = int(value.get("attempt_index", 0))
    failure = _validated_failure(
        queue_root,
        value,
        logical_stage=logical_stage,
        plan_seal_identity=plan_seal_identity,
        attempt=attempt,
    )
    if failure is None:
        return value
    next_attempt = attempt + 1
    if next_attempt >= MAX_ATTEMPTS_PER_IDENTITY:
        return None
    if _validated_failure(
        queue_root,
        value,
        logical_stage=logical_stage,
        plan_seal_identity=plan_seal_identity,
        attempt=next_attempt,
    ) is not None:
        return None
    value["attempt_index"] = next_attempt
    value["engineering_retry_of_attempt"] = attempt
    return value


def _plan_row_for_identity(
    result: Path,
    *,
    logical_stage: str,
    physical_run_id: str,
) -> dict[str, Any]:
    plan_name = f"DBLP_H3_{logical_stage}_RUN_PLAN.csv"
    seal_name = f"DBLP_H3_{logical_stage}_RUN_PLAN_SEAL.json"
    rows = read_csv(result / plan_name)
    seal = read_json(result / seal_name)
    plan = validate_plan(rows, seal, logical_stage=logical_stage)
    matches = [row for row in plan if str(row["physical_run_id"]) == physical_run_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"M7F-H3 retry lineage lacks one {logical_stage} row for {physical_run_id}"
        )
    return dict(matches[0])


def _continuation_row_in_attempt(
    result: Path,
    row: Mapping[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    value = dict(row)
    start = int(value["continuation_from_update"])
    root = attempt_root(result, value, str(value["logical_stage"]), attempt)
    config_path = root / "RESOLVED_CONFIG.json"
    checkpoint = root / "checkpoints" / f"update_{start:06d}.pt"
    paused = root / f"RUN_PAUSED_{start:06d}.json"
    sidecar = root / f"SIDECAR_CAPTURE_RECEIPT_PART_{start:06d}.json"
    boundary = root / f"M6F_M7F_SEARCH_{start:06d}_COMPLETE.json"
    if not all(
        path.is_file() and not path.is_symlink()
        for path in (config_path, checkpoint, paused, sidecar, boundary)
    ):
        raise RuntimeError("M7F-H3 retry replay did not materialize its continuation boundary")
    config = read_json(config_path)
    config_identity = canonical_json_hash(config)
    config_seal = str(config.get("plan_seal_hash", ""))
    expected_config_seal = str(value.get("run_config_plan_seal_identity", ""))
    boundary_value = read_json(boundary)
    boundary_basis = boundary_value.get("basis", {})
    if (
        len(config_seal) != 64
        or config_seal != expected_config_seal
        or boundary_value.get("status") != "PASS"
        or boundary_value.get("identity") != canonical_json_hash(boundary_basis)
        or int(boundary_basis.get("endpoint_update", -1)) != start
        or boundary_basis.get("physical_run_id") != value.get("physical_run_id")
        or boundary_basis.get("candidate_set_hash") != value.get("candidate_set_hash")
        or int(boundary_basis.get("model_seed", -1)) != int(value.get("model_seed", -2))
        or boundary_basis.get("checkpoint_sha256") != sha256_file(checkpoint)
        or boundary_basis.get("run_identity") != config_identity
    ):
        raise ValueError("M7F-H3 retry continuation boundary identity is invalid")
    value.update({
        "attempt_index": attempt,
        "resume_checkpoint_path": str(checkpoint.resolve(strict=True)),
        "resume_checkpoint_sha256": sha256_file(checkpoint),
        "run_config_plan_seal_identity": config_seal,
        "run_config_identity": config_identity,
        "status": "AUTHORIZED",
    })
    return value


def _materialize_retry_lineage(
    *,
    row: Mapping[str, Any],
    logical_stage: str,
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool,
) -> dict[str, Any]:
    """Replay prerequisite checkpoints before a retry."""

    target = canonical_plan_row(row)
    attempt = int(target.get("attempt_index", 0))
    start = int(target.get("continuation_from_update", 0))
    if attempt != 1 or start == 0:
        return target
    result = Path(result_root).resolve()
    physical = str(target["physical_run_id"])
    seed_ordinal = int(target["model_seed_ordinal"])
    replay_stages: list[str]
    if logical_stage == "R2":
        replay_stages = ["R1"]
    elif logical_stage == "R3":
        replay_stages = (["R1", "R2"] if seed_ordinal == 0 else ["R2"])
    elif logical_stage == "R4":
        replay_stages = (
            ["R1", "R2", "R3"] if seed_ordinal == 0 else ["R2", "R3"]
        )
    else:
        raise ValueError("M7F-H3 only search continuations require retry lineage replay")
    replay_started_at = utc_now()
    for stage in replay_stages:
        replay = _plan_row_for_identity(
            result,
            logical_stage=stage,
            physical_run_id=physical,
        )
        replay["attempt_index"] = attempt
        if stage == "R2" and int(replay["continuation_from_update"]) == 0:
            _run_fresh_r2_seed(
                row=replay,
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=result,
                source_commit=source_commit,
                source_tree=source_tree,
                enforce_target_gpu=enforce_target_gpu,
            )
        else:
            if int(replay["continuation_from_update"]) > 0:
                replay = _continuation_row_in_attempt(
                    result,
                    replay,
                    attempt=attempt,
                )
            run_frozen_plan_row(
                plan_row=replay,
                stage=ENGINE_STAGE[stage],
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=result,
                source_commit=source_commit,
                source_tree=source_tree,
                enforce_target_gpu=enforce_target_gpu,
            )
    target = _continuation_row_in_attempt(result, target, attempt=attempt)
    target["engineering_retry_of_attempt"] = 0
    target["engineering_retry_replayed_from_zero"] = True
    target["retry_invocation_started_at"] = replay_started_at
    return target


def _common_root_pause(queue_root: Path) -> dict[str, Any] | None:
    grouped: dict[str, set[str]] = {}
    for failure in _failures(queue_root):
        basis = failure.get("basis", {})
        grouped.setdefault(str(basis.get("error_fingerprint", "")), set()).add(
            str(basis.get("physical_run_id", ""))
        )
    offenders = sorted(
        fingerprint
        for fingerprint, identities in grouped.items()
        if fingerprint and len(identities) >= COMMON_ROOT_CAUSE_PAUSE_THRESHOLD
    )
    if not offenders:
        return None
    basis = {
        "status": "COMMON_ROOT_CAUSE_PAUSE",
        "error_fingerprints": offenders,
        "threshold": COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
        "new_claims_allowed": False,
    }
    pause = envelope("metachorus.v53.m7f-h3.common-root-pause.v1", "PAUSED", basis)
    write_json_once(queue_root / "DBLP_H3_COMMON_ROOT_CAUSE_PAUSE.json", pause)
    return pause


def run_dynamic_worker(
    *,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    logical_stage: str,
    worker_index: int,
    worker_count: int,
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    if worker_count != EXPECTED_GPU_COUNT or worker_index not in range(EXPECTED_GPU_COUNT):
        raise ValueError("M7F-H3 dynamic queue requires exactly four workers indexed 0..3")
    plan = validate_plan(rows, seal, logical_stage=logical_stage)
    result = Path(result_root).resolve()
    queue_root = result / "dynamic_queue" / logical_stage.casefold()
    worker_basis = {
        "logical_stage": logical_stage,
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "plan_seal_identity": seal["identity"],
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "launch_nonce": os.environ.get("DBLP_H3_LAUNCH_NONCE", "LOCAL_TEST_RUN"),
        "started_at": utc_now(),
        **_gpu_runtime(enforce_target_gpu=enforce_target_gpu),
    }
    worker_receipt_root = queue_root / "workers" / str(worker_basis["launch_nonce"])
    write_json_once(
        worker_receipt_root / f"worker_{worker_index}_STARTED.json",
        envelope("metachorus.v53.m7f-h3.worker-started.v1", "STARTED", worker_basis),
    )
    completed = 0
    claimed = 0
    failed = 0
    for row in sorted(plan, key=lambda value: (int(value.get("queue_ordinal", 0)), str(value["physical_run_id"]))):
        if (queue_root / "DBLP_H3_COMMON_ROOT_CAUSE_PAUSE.json").exists():
            break
        if completed_marker(result, row, logical_stage) is not None:
            continue
        execution_row = _execution_row_for_retry(
            queue_root,
            row,
            logical_stage=logical_stage,
            plan_seal_identity=str(seal["identity"]),
        )
        if execution_row is None:
            continue
        attempt = int(execution_row.get("attempt_index", 0))
        if attempt not in range(MAX_ATTEMPTS_PER_IDENTITY):
            raise ValueError("M7F-H3 retry attempt exceeds the frozen maximum")
        preferred = row.get("preferred_gpu_slot", "")
        if preferred not in (None, "") and int(preferred) != worker_index:
            continue
        claim_path = queue_root / "claims" / f"{row['physical_run_id']}__attempt_{attempt:02d}.claim"
        claim = {
            "physical_run_id": row["physical_run_id"],
            "logical_stage": logical_stage,
            "attempt_index": attempt,
            "worker_index": worker_index,
            "plan_seal_identity": seal["identity"],
            "claimed_at": utc_now(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }
        if not _atomic_claim(claim_path, claim):
            continue
        claimed += 1
        attempt_started = datetime.now(timezone.utc)
        try:
            execution_row = _materialize_retry_lineage(
                row=execution_row,
                logical_stage=logical_stage,
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=result,
                source_commit=source_commit,
                source_tree=source_tree,
                enforce_target_gpu=enforce_target_gpu,
            )
            run_h3_row(
                plan_row=execution_row,
                logical_stage=logical_stage,
                worker_index=worker_index,
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=result,
                source_commit=source_commit,
                source_tree=source_tree,
                enforce_target_gpu=enforce_target_gpu,
            )
            completed += 1
        except BaseException as exc:
            failed += 1
            failed_at = datetime.now(timezone.utc)
            fingerprint = _failure_fingerprint(exc)
            basis = {
                "logical_stage": logical_stage,
                "physical_run_id": row["physical_run_id"],
                "attempt_index": attempt,
                "worker_index": worker_index,
                "plan_seal_identity": seal["identity"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_fingerprint": fingerprint,
                "invocation_started_at": attempt_started.isoformat(),
                "invocation_ended_at": failed_at.isoformat(),
                "invocation_wall_seconds": (failed_at - attempt_started).total_seconds(),
                "gpu_uuid": worker_basis["gpu_uuid"],
                "gpu_pci_bus_id": worker_basis["gpu_pci_bus_id"],
                "runtime_domain": worker_basis["runtime_domain"],
                "failed_at": failed_at.isoformat(),
                "retry_allowed": attempt == 0,
                "scientific_metric_triggered_retry": False,
            }
            write_json_once(
                queue_root / "failures" / f"{row['physical_run_id']}__attempt_{attempt:02d}.json",
                envelope("metachorus.v53.m7f-h3.run-failure.v1", "FAILED_PRESERVED", basis),
            )
            _common_root_pause(queue_root)
            if (queue_root / "DBLP_H3_COMMON_ROOT_CAUSE_PAUSE.json").exists():
                break
    final_basis = {
        **worker_basis,
        "ended_at": utc_now(),
        "claimed_run_count": claimed,
        "completed_run_count": completed,
        "failed_run_count": failed,
        "queue_paused": (queue_root / "DBLP_H3_COMMON_ROOT_CAUSE_PAUSE.json").is_file(),
    }
    receipt = envelope("metachorus.v53.m7f-h3.worker-complete.v1", "PASS" if failed == 0 else "FAILED_PRESERVED", final_basis)
    write_json_once(worker_receipt_root / f"worker_{worker_index}_COMPLETE.json", receipt)
    return receipt


__all__ = [
    "ENGINE_STAGE",
    "h3_marker_name",
    "attempt_root",
    "completed_marker",
    "validate_engine_marker",
    "run_h3_row",
    "run_dynamic_worker",
]
