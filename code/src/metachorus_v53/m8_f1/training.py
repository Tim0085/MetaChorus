"""Four-worker execution for the frozen M8-F1 completion matrix."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from metachorus_v53.m6f_m7f_fast.training import (
    _cache_contract,
    _canary_engineering_audit,
)
from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession

from .common import (
    canonical_json_hash,
    envelope,
    read_csv,
    read_json,
    sha256_file,
    write_json_once,
)
from .constants import (
    CANARY_ENDPOINT,
    COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
    DATASETS,
    DATASET_SPECS,
    EXPECTED_GPU_COUNT,
    FINAL10_CHECKPOINT_SCHEDULE,
    FINAL10_ENDPOINT,
    MAX_ATTEMPTS_PER_IDENTITY,
)
from .planning import canonical_plan_row, validate_plan


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def marker_name(logical_stage: str) -> str:
    if logical_stage not in {"CANARY", "FINAL10"}:
        raise ValueError("unsupported M8-F1 logical stage")
    return f"M8_F1_{logical_stage}_COMPLETE.json"


def _branch(logical_stage: str) -> str:
    return "canary_runs" if logical_stage == "CANARY" else "final10_runs"


def attempt_root(
    result_root: str | Path, row: Mapping[str, Any], logical_stage: str, attempt: int
) -> Path:
    if int(attempt) not in range(MAX_ATTEMPTS_PER_IDENTITY):
        raise ValueError("M8-F1 permits attempt 0 and one engineering retry")
    return (
        Path(result_root).resolve()
        / _branch(logical_stage)
        / str(row["physical_run_id"])
        / f"attempt_{int(attempt):02d}"
    )


def _marker_paths(
    result_root: str | Path, row: Mapping[str, Any], logical_stage: str
) -> list[Path]:
    parent = Path(result_root).resolve() / _branch(logical_stage) / str(row["physical_run_id"])
    return [
        attempt / marker_name(logical_stage)
        for attempt in sorted(parent.glob("attempt_*"))
        if (attempt / marker_name(logical_stage)).is_file()
        and not (attempt / marker_name(logical_stage)).is_symlink()
    ]


def validate_completion_marker(
    marker: Mapping[str, Any], row: Mapping[str, Any], *, logical_stage: str
) -> dict[str, Any]:
    value = canonical_plan_row(row)
    basis = marker.get("basis", {})
    expected_endpoint = CANARY_ENDPOINT if logical_stage == "CANARY" else FINAL10_ENDPOINT
    spec = DATASET_SPECS.get(str(value.get("dataset")), {})
    if (
        marker.get("schema_version") != "metachorus.v53.m8-f1.run-complete.v1"
        or marker.get("status") != "PASS"
        or marker.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or basis.get("physical_run_id") != value.get("physical_run_id")
        or basis.get("logical_role") != value.get("logical_role")
        or basis.get("dataset") != value.get("dataset")
        or basis.get("dataset_identity") != spec.get("dataset_identity")
        or basis.get("cache_identity") != spec.get("cache_identity")
        or basis.get("candidate_set_id") != value.get("candidate_set_id")
        or basis.get("candidate_set_hash") != value.get("candidate_set_hash")
        or int(basis.get("candidate_set_ordinal", -1))
        != int(value.get("candidate_set_ordinal", -2))
        or tuple(basis.get("path_ids", [])) != tuple(value.get("path_ids", []))
        or int(basis.get("model_seed", -1)) != int(value.get("model_seed", -2))
        or int(basis.get("continuation_from_update", -1)) != 0
        or int(basis.get("endpoint_update", -1)) != expected_endpoint
        or int(basis.get("optimizer_steps", -1)) != expected_endpoint
        or basis.get("plan_seal_identity") != value.get("plan_seal_identity")
        or basis.get("training_config_identity") != value.get("training_config_identity")
        or re.fullmatch(r"[0-9a-f]{64}", str(basis.get("resolved_run_config_identity", "")))
        is None
        or int(basis.get("attempt_index", -1)) not in range(MAX_ATTEMPTS_PER_IDENTITY)
        or (
            int(basis.get("attempt_index", -1)) == 0
            and basis.get("engineering_retry_of_attempt") not in {None, ""}
        )
        or (
            int(basis.get("attempt_index", -1)) == 1
            and int(basis.get("engineering_retry_of_attempt", -1)) != 0
        )
        or int(basis.get("worker_index", -1)) not in range(EXPECTED_GPU_COUNT)
        or not str(basis.get("gpu_uuid", "")).startswith("GPU-")
        or not str(basis.get("gpu_pci_bus_id", ""))
        or (
            logical_stage == "CANARY"
            and int(basis.get("worker_index", -1)) != int(value.get("preferred_gpu_slot", -2))
        )
        or basis.get("labels_mounted_during_training") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
        or basis.get("training_from_zero") is not True
        or basis.get("search_checkpoint_reused") is not False
        or basis.get("one_process_per_gpu") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
        or basis.get("cpu_test_double") is not False
    ):
        raise ValueError("M8-F1 completion marker identity is invalid")
    return dict(basis)


def completed_marker(
    result_root: str | Path, row: Mapping[str, Any], logical_stage: str
) -> tuple[Path, dict[str, Any]] | None:
    matches = _marker_paths(result_root, row, logical_stage)
    if len(matches) > 1:
        raise RuntimeError(f"multiple M8-F1 completion markers: {row['physical_run_id']}")
    if not matches:
        return None
    marker = read_json(matches[0])
    basis = validate_completion_marker(marker, row, logical_stage=logical_stage)
    config_path = matches[0].parent / "RESOLVED_CONFIG.json"
    if config_path.is_symlink() or not config_path.is_file() or config_path.stat().st_nlink != 1:
        raise ValueError("M8-F1 resolved run config is missing or unsafe")
    config = O0RunConfig.from_mapping(read_json(config_path))
    config.validate()
    value = canonical_plan_row(row)
    expected_schedule = (CANARY_ENDPOINT,) if logical_stage == "CANARY" else FINAL10_CHECKPOINT_SCHEDULE
    if (
        config.identity() != basis["resolved_run_config_identity"]
        or Path(config.output_root).resolve() != matches[0].parent.resolve()
        or config.dataset != value["dataset"]
        or config.path_set_role != value["logical_role"]
        or config.path_set_id != value["candidate_set_id"]
        or config.candidate_set_hash != value["candidate_set_hash"]
        or tuple(config.path_ids) != tuple(value["path_ids"])
        or int(config.candidate_set_ordinal) != int(value["candidate_set_ordinal"])
        or int(config.model_seed) != int(value["model_seed"])
        or tuple(config.checkpoint_schedule) != tuple(expected_schedule)
        or tuple(config.representation_schedule or ())
        != tuple(value["representation_schedule"])
        or config.plan_seal_hash != value["plan_seal_identity"]
        or config.run_role != ("M8_F1_CANARY" if logical_stage == "CANARY" else "M8_F1_FINAL10")
        or config.device != "cuda:0"
    ):
        raise ValueError("M8-F1 resolved run config identity is invalid")
    return matches[0], marker


def _gpu_runtime(*, enforce_target_gpu: bool) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid = os.environ.get("M8_F1_GPU_UUID", "")
    pci_bus_id = os.environ.get("M8_F1_GPU_PCI_BUS_ID", "")
    runtime_domain = os.environ.get("M8_F1_RUNTIME_DOMAIN", socket.gethostname())
    if enforce_target_gpu:
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError("M8-F1 worker must see exactly one local CUDA device 0")
        name = torch.cuda.get_device_name(0)
        if not uuid or not pci_bus_id or not visible:
            raise RuntimeError("M8-F1 worker lacks frozen GPU UUID/PCI/visibility metadata")
        if (
            os.environ.get("LABEL_ACCESS") != "FORBIDDEN"
            or os.environ.get("METACHORUS_NETWORK_DISABLED") != "1"
        ):
            raise RuntimeError("M8-F1 formal worker lacks label/network isolation gates")
    else:
        name = os.environ.get("M8_F1_GPU_NAME", "CPU_TEST_DOUBLE")
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
        "cpu_test_double": not enforce_target_gpu,
    }


def build_run_config(
    *,
    row: Mapping[str, Any],
    logical_stage: str,
    cache_root: str | Path,
    output_root: str | Path,
    source_commit: str,
    source_tree: str,
    device: str = "cuda:0",
) -> O0RunConfig:
    value = canonical_plan_row(row)
    cache = _cache_contract(Path(cache_root).resolve())
    if value.get("cache_identity") != cache["cache_identity"]:
        raise ValueError("M8-F1 plan/cache identity mismatch")
    dataset = str(value.get("dataset"))
    if dataset not in DATASETS:
        raise ValueError("M8-F1 plan row has an unauthorized dataset")
    canary = logical_stage == "CANARY"
    endpoint = CANARY_ENDPOINT if canary else FINAL10_ENDPOINT
    schedule = (CANARY_ENDPOINT,) if canary else FINAL10_CHECKPOINT_SCHEDULE
    config = O0RunConfig(
        dataset=dataset,
        path_set_role=str(value["logical_role"]),
        path_set_id=str(value["candidate_set_id"]),
        path_ids=tuple(value["path_ids"]),
        candidate_set_hash=str(value["candidate_set_hash"]),
        model_seed=int(value["model_seed"]),
        max_updates=endpoint,
        capture_mode="on",
        checkpoint_schedule=tuple(schedule),
        representation_schedule=tuple(value["representation_schedule"]),
        cache_root=str(Path(cache_root).resolve()),
        output_root=str(Path(output_root).resolve()),
        device=str(device),
        source_commit=str(source_commit),
        source_tree=str(source_tree),
        cache_identity=cache["cache_identity"],
        augmentation_namespace=(
            f"m8-f1/{logical_stage.casefold()}/{dataset}/{value['physical_run_id']}"
        ),
        run_role="M8_F1_CANARY" if canary else "M8_F1_FINAL10",
        eligible_for_scientific_result=not canary,
        eligible_for_selector_development=False,
        physical_run_id=str(value["physical_run_id"]),
        candidate_set_ordinal=int(value["candidate_set_ordinal"]),
        source_data_identity=cache["source_data_identity"],
        projection_identity=cache["projection_identity"],
        candidate_bank_hash=cache["candidate_bank_hash"],
        plan_seal_hash=str(value["plan_seal_identity"]),
    )
    config.validate()
    return config


def run_m8_row(
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
    if row.get("logical_stage") != logical_stage or logical_stage not in {"CANARY", "FINAL10"}:
        raise ValueError("M8-F1 row/logical-stage mismatch")
    if logical_stage == "CANARY" and int(row["preferred_gpu_slot"]) != int(worker_index):
        raise ValueError("M8-F1 canary reached the wrong GPU worker")
    prior = completed_marker(result_root, row, logical_stage)
    if prior is not None:
        return prior[1]
    attempt = int(row.get("attempt_index", 0))
    root = attempt_root(result_root, row, logical_stage, attempt)
    if root.exists():
        raise FileExistsError(f"M8-F1 fresh attempt root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    gpu = _gpu_runtime(enforce_target_gpu=enforce_target_gpu)
    if logical_stage == "CANARY" and gpu["runtime_domain"] != str(row["planned_runtime_domain"]):
        raise RuntimeError("M8-F1 canary runtime domain differs from the sealed plan")
    config = build_run_config(
        row=row,
        logical_stage=logical_stage,
        cache_root=cache_root,
        output_root=root,
        source_commit=source_commit,
        source_tree=source_tree,
        device="cuda:0" if enforce_target_gpu else "cpu",
    )
    session = O0TrainingSession(config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu)
    endpoint = CANARY_ENDPOINT if logical_stage == "CANARY" else FINAL10_ENDPOINT
    run = session.run(stop_at_update=endpoint)
    if (
        run.get("labels_mounted") is not False
        or int(run.get("downstream_evaluations", -1)) != 0
        or int(run.get("optimizer_steps", -1)) != endpoint
    ):
        raise RuntimeError("M8-F1 O0 run violated labels/evaluation/update boundaries")
    write_json_once(root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    checkpoint = root / "checkpoints" / f"update_{endpoint:06d}.pt"
    representation = root / "representations" / f"update_{endpoint:06d}.npy"
    if (
        checkpoint.is_symlink()
        or representation.is_symlink()
        or not checkpoint.is_file()
        or not representation.is_file()
    ):
        raise RuntimeError("M8-F1 endpoint capture is incomplete")
    ended = datetime.now(timezone.utc)
    basis = {
        "logical_stage": logical_stage,
        "engine_run_role": config.run_role,
        "physical_run_id": row["physical_run_id"],
        "logical_role": row["logical_role"],
        "dataset": row["dataset"],
        "dataset_identity": row["dataset_identity"],
        "cache_identity": row["cache_identity"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "path_ids": list(row["path_ids"]),
        "model_seed": int(row["model_seed"]),
        "attempt_index": attempt,
        "continuation_from_update": 0,
        "endpoint_update": endpoint,
        "optimizer_steps": int(run["optimizer_steps"]),
        "plan_seal_identity": row["plan_seal_identity"],
        "training_config_identity": row["training_config_identity"],
        "resolved_run_config_identity": config.identity(),
        "checkpoint_relative_path": checkpoint.relative_to(Path(result_root).resolve()).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "representation_relative_path": representation.relative_to(Path(result_root).resolve()).as_posix(),
        "representation_sha256": sha256_file(representation),
        "model_total_seconds": float(run.get("model_total_seconds", 0.0)),
        "peak_gpu_allocated_bytes": int(run.get("peak_gpu_allocated_bytes", 0)),
        "peak_gpu_reserved_bytes": int(run.get("peak_gpu_reserved_bytes", 0)),
        "worker_index": int(worker_index),
        **gpu,
        "planned_runtime_domain": row["planned_runtime_domain"],
        "invocation_started_at": started.isoformat(),
        "invocation_ended_at": ended.isoformat(),
        "invocation_wall_seconds": (ended - started).total_seconds(),
        "training_from_zero": True,
        "search_checkpoint_reused": False,
        "labels_mounted_during_training": False,
        "downstream_evaluations_during_training": 0,
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
        "engineering_retry_of_attempt": row.get("engineering_retry_of_attempt", ""),
    }
    if logical_stage == "CANARY":
        basis["canary_engineering_audit"] = _canary_engineering_audit(
            session=session,
            config=config,
            run_root=root,
            repo_root=repo_root,
            enforce_target_gpu=enforce_target_gpu,
        )
    marker = envelope("metachorus.v53.m8-f1.run-complete.v1", "PASS", basis)
    write_json_once(root / marker_name(logical_stage), marker)
    return marker


def _atomic_claim(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        encoded = (
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("M8-F1 atomic claim write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _failure_fingerprint(exc: BaseException) -> str:
    message = f"{type(exc).__name__}:{str(exc).strip()}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _failure_path(queue_root: Path, physical_run_id: str, attempt: int) -> Path:
    return queue_root / "failures" / f"{physical_run_id}__attempt_{attempt:02d}.json"


def _write_failure(
    *, queue_root: Path, row: Mapping[str, Any], logical_stage: str, worker_index: int,
    attempt: int, exc: BaseException, started_at: str, gpu_runtime: Mapping[str, Any]
) -> dict[str, Any]:
    fingerprint = _failure_fingerprint(exc)
    ended_at = utc_now()
    started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
    ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    gpu = dict(gpu_runtime)
    basis = {
        "logical_stage": logical_stage,
        "physical_run_id": row["physical_run_id"],
        "dataset": row["dataset"],
        "logical_role": row["logical_role"],
        "model_seed": int(row["model_seed"]),
        "plan_seal_identity": row["plan_seal_identity"],
        "attempt_index": int(attempt),
        "engineering_retry_of_attempt": row.get("engineering_retry_of_attempt", ""),
        "worker_index": int(worker_index),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_fingerprint": fingerprint,
        "traceback": traceback.format_exc(),
        "started_at": started_at,
        "ended_at": ended_at,
        "invocation_started_at": started_at,
        "invocation_ended_at": ended_at,
        "invocation_wall_seconds": max((ended - started).total_seconds(), 1e-9),
        "gpu_uuid": gpu["gpu_uuid"],
        "gpu_pci_bus_id": gpu["gpu_pci_bus_id"],
        "runtime_domain": gpu["runtime_domain"],
        "retry_allowed": int(attempt) + 1 < MAX_ATTEMPTS_PER_IDENTITY,
        "scientific_metric_triggered_retry": False,
        "status": "FAILED_PRESERVED",
    }
    failure = envelope("metachorus.v53.m8-f1.run-failure.v1", "FAILED_PRESERVED", basis)
    write_json_once(_failure_path(queue_root, str(row["physical_run_id"]), attempt), failure)
    return failure


def _failure_rows(queue_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((queue_root / "failures").glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("M8-F1 failure registry contains a non-regular file")
        value = read_json(path)
        basis = value.get("basis", {})
        physical = str(basis.get("physical_run_id", ""))
        attempt = int(basis.get("attempt_index", -1))
        if (
            value.get("schema_version") != "metachorus.v53.m8-f1.run-failure.v1"
            or value.get("status") != "FAILED_PRESERVED"
            or value.get("identity") != canonical_json_hash(basis)
            or basis.get("scientific_metric_triggered_retry") is not False
            or attempt not in range(MAX_ATTEMPTS_PER_IDENTITY)
            or bool(basis.get("retry_allowed")) != (attempt + 1 < MAX_ATTEMPTS_PER_IDENTITY)
            or (
                attempt == 0
                and basis.get("engineering_retry_of_attempt") not in {None, ""}
            )
            or (
                attempt == 1
                and int(basis.get("engineering_retry_of_attempt", -1)) != 0
            )
            or path.name != f"{physical}__attempt_{attempt:02d}.json"
        ):
            raise ValueError("M8-F1 failure evidence is invalid")
        rows.append(dict(basis))
    return rows


def _common_root_pause(queue_root: Path) -> dict[str, Any] | None:
    pause_path = queue_root / "M8_F1_COMMON_ROOT_CAUSE_PAUSE.json"
    if pause_path.is_file() and not pause_path.is_symlink():
        pause = read_json(pause_path)
        basis = pause.get("basis", {})
        if (
            pause.get("schema_version")
            != "metachorus.v53.m8-f1.common-root-cause-pause.v1"
            or pause.get("status") != "COMMON_ROOT_CAUSE_PAUSE"
            or pause.get("identity") != canonical_json_hash(basis)
            or basis.get("new_claims_allowed") is not False
            or basis.get("scientific_metric_triggered_retry") is not False
            or int(basis.get("affected_identity_count", -1))
            < COMMON_ROOT_CAUSE_PAUSE_THRESHOLD
            or int(basis.get("affected_identity_count", -1))
            != len(set(map(str, basis.get("affected_physical_run_ids", []))))
        ):
            raise ValueError("M8-F1 common-root-cause pause evidence is invalid")
        return pause
    by_fingerprint: dict[str, set[str]] = {}
    for row in _failure_rows(queue_root):
        by_fingerprint.setdefault(str(row["error_fingerprint"]), set()).add(str(row["physical_run_id"]))
    affected = [
        (fingerprint, identities)
        for fingerprint, identities in by_fingerprint.items()
        if len(identities) >= COMMON_ROOT_CAUSE_PAUSE_THRESHOLD
    ]
    if not affected:
        return None
    fingerprint, identities = sorted(affected, key=lambda item: item[0])[0]
    pause = envelope(
        "metachorus.v53.m8-f1.common-root-cause-pause.v1",
        "COMMON_ROOT_CAUSE_PAUSE",
        {
            "error_fingerprint": fingerprint,
            "affected_physical_run_ids": sorted(identities),
            "affected_identity_count": len(identities),
            "new_claims_allowed": False,
            "scientific_metric_triggered_retry": False,
        },
    )
    try:
        write_json_once(pause_path, pause)
    except FileExistsError:
        return _common_root_pause(queue_root)
    return pause


def _next_attempt(queue_root: Path, row: Mapping[str, Any]) -> int | None:
    physical = str(row["physical_run_id"])
    failures = {
        (str(item["physical_run_id"]), int(item["attempt_index"])): item
        for item in _failure_rows(queue_root)
    }
    if (physical, 1) in failures:
        return None
    if (physical, 0) in failures:
        if failures[(physical, 0)].get("retry_allowed") is not True:
            return None
        return 1
    return 0


def run_dynamic_worker(
    *,
    repo_root: str | Path,
    cache_parent: str | Path,
    result_root: str | Path,
    plan_path: str | Path,
    seal_path: str | Path,
    logical_stage: str,
    worker_index: int,
    worker_count: int,
    source_commit: str,
    source_tree: str,
    launch_nonce: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    if int(worker_count) != EXPECTED_GPU_COUNT or int(worker_index) not in range(EXPECTED_GPU_COUNT):
        raise ValueError("M8-F1 requires exactly four GPU workers")
    rows = read_csv(plan_path)
    seal = read_json(seal_path)
    plan = validate_plan(rows, seal, logical_stage=logical_stage)
    queue_root = Path(result_root).resolve() / "queues" / logical_stage.casefold()
    worker_root = queue_root / "workers" / str(launch_nonce) / f"worker_{int(worker_index)}"
    worker_root.mkdir(parents=True, exist_ok=True)
    gpu = _gpu_runtime(enforce_target_gpu=enforce_target_gpu)
    started = envelope(
        "metachorus.v53.m8-f1.worker-started.v1",
        "STARTED",
        {
            "logical_stage": logical_stage,
            "worker_index": int(worker_index),
            "worker_count": int(worker_count),
            "launch_nonce": str(launch_nonce),
            "plan_seal_identity": seal["identity"],
            "source_commit": str(source_commit),
            "source_tree": str(source_tree),
            **gpu,
            "started_at": utc_now(),
            "labels_mounted_during_training": False,
            "one_process_per_gpu": True,
            "ddp_used": False,
            "data_parallel_used": False,
            "torchrun_used": False,
        },
    )
    write_json_once(worker_root / "STARTED.json", started)
    completed = 0
    failures = 0
    for base_row in sorted(plan, key=lambda row: int(row["queue_ordinal"])):
        if logical_stage == "CANARY" and int(base_row["preferred_gpu_slot"]) != int(worker_index):
            continue
        if _common_root_pause(queue_root) is not None:
            break
        if completed_marker(result_root, base_row, logical_stage) is not None:
            completed += 1
            continue
        attempt = _next_attempt(queue_root, base_row)
        while attempt is not None:
            row = canonical_plan_row(
                {
                    **base_row,
                    "attempt_index": attempt,
                    "engineering_retry_of_attempt": 0 if attempt == 1 else "",
                }
            )
            claim = queue_root / "claims" / f"{row['physical_run_id']}__attempt_{attempt:02d}.json"
            claim_basis = {
                "physical_run_id": row["physical_run_id"],
                "logical_stage": logical_stage,
                "plan_seal_identity": seal["identity"],
                "attempt_index": attempt,
                "worker_index": int(worker_index),
                "launch_nonce": str(launch_nonce),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at": utc_now(),
            }
            if not _atomic_claim(
                claim,
                envelope("metachorus.v53.m8-f1.queue-claim.v1", "CLAIMED", claim_basis),
            ):
                break
            invocation_started = utc_now()
            try:
                run_m8_row(
                    plan_row=row,
                    logical_stage=logical_stage,
                    worker_index=worker_index,
                    repo_root=repo_root,
                    cache_root=Path(cache_parent).resolve() / str(row["dataset"]),
                    result_root=result_root,
                    source_commit=source_commit,
                    source_tree=source_tree,
                    enforce_target_gpu=enforce_target_gpu,
                )
                completed += 1
                break
            except BaseException as exc:
                failures += 1
                evidence = _write_failure(
                    queue_root=queue_root,
                    row=row,
                    logical_stage=logical_stage,
                    worker_index=worker_index,
                    attempt=attempt,
                    exc=exc,
                    started_at=invocation_started,
                    gpu_runtime=gpu,
                )
                if _common_root_pause(queue_root) is not None:
                    break
                attempt = attempt + 1 if evidence["basis"]["retry_allowed"] else None
    complete = envelope(
        "metachorus.v53.m8-f1.worker-complete.v1",
        "PASS" if _common_root_pause(queue_root) is None else "COMMON_ROOT_CAUSE_PAUSE",
        {
            "logical_stage": logical_stage,
            "worker_index": int(worker_index),
            "worker_count": int(worker_count),
            "launch_nonce": str(launch_nonce),
            "plan_seal_identity": seal["identity"],
            "completed_identity_count": completed,
            "failure_attempt_count": failures,
            "ended_at": utc_now(),
            "labels_mounted_during_training": False,
        },
    )
    write_json_once(worker_root / "COMPLETE.json", complete)
    return complete


__all__ = [
    "utc_now",
    "marker_name",
    "attempt_root",
    "validate_completion_marker",
    "completed_marker",
    "build_run_config",
    "run_m8_row",
    "run_dynamic_worker",
]
