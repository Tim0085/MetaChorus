"""Two-worker, append-only execution over the frozen O0 trainer."""

from __future__ import annotations

import json
import os
import socket
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession

from .common import (
    canonical_json_hash,
    envelope,
    parse_bool,
    read_json,
    sha256_file,
    write_json_once,
)
from .constants import (
    CACHE_IDENTITIES,
    CANARY_ENDPOINT_UPDATE,
    CANDIDATE_BANK_IDENTITIES,
    COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
    DATASET_IDENTITIES,
    ENDPOINT_UPDATE,
    EVALUATION_MODEL_SEEDS,
    EXPECTED_GPU_COUNT,
    MAX_ATTEMPTS_PER_IDENTITY,
)
from .planning import build_canary_plan


STAGES = ("CANARY", "EVALUATION")
MARKER_NAMES = {
    "CANARY": "M8_F3R_RERUN_2G_CANARY_COMPLETE.json",
    "EVALUATION": "M8_F3R_RERUN_2G_EVALUATION_COMPLETE.json",
}
BRANCHES = {"CANARY": "canary_runs", "EVALUATION": "evaluation_runs"}
RUN_ROLES = {
    "CANARY": "M8_F3R_RERUN_2G_CANARY",
    "EVALUATION": "M8_F3R_RERUN_2G_EVALUATION",
}
SEAL_SCHEMAS = {
    "CANARY": "metachorus.v53.m8-f3r-rerun-2g.canary-plan-seal.v1",
    "EVALUATION": "metachorus.v53.m8-f3r-rerun-2g.evaluation-matrix-seal.v1",
}
SEAL_STATUSES = {
    "CANARY": "SEALED_BEFORE_CANARY_GPU_TRAINING",
    "EVALUATION": "EVALUATION_MATRIX_SEALED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hex64(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def _hex40(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase Git object identity")
    return text


def _canonical_row(value: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported M8-F3R-RERUN-2G stage: {stage}")
    row = dict(value)
    for field in ("model_seed", "start_update", "endpoint_update"):
        row[field] = int(row[field])
    row["path_ids"] = json.dumps(
        [str(item) for item in json.loads(str(row["path_ids"]))],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row["labels_mounted"] = parse_bool(row.get("labels_mounted", False))
    row["downstream_evaluations_during_training"] = int(
        row.get("downstream_evaluations_during_training", 0)
    )
    if stage == "CANARY":
        row["canary_ordinal"] = int(row["canary_ordinal"])
        row["gpu_slot"] = int(row["gpu_slot"])
        row["scientific_result"] = parse_bool(row.get("scientific_result", False))
        row["formal_run_restarts_from_zero"] = parse_bool(
            row.get("formal_run_restarts_from_zero", False)
        )
    return row


def _seal(seal: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    value = dict(seal)
    basis = value.get("basis", {})
    if (
        value.get("schema_version") != SEAL_SCHEMAS[stage]
        or value.get("status") != SEAL_STATUSES[stage]
        or value.get("identity") != canonical_json_hash(basis)
    ):
        raise ValueError(f"{stage} plan seal identity is invalid")
    return value


def validate_plan(
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    normalized = [_canonical_row(row, stage=stage) for row in rows]
    sealed = _seal(seal, stage=stage)
    basis = sealed["basis"]
    if int(basis.get("row_count", -1)) != len(normalized) or basis.get(
        "rows_identity"
    ) != canonical_json_hash(normalized):
        raise ValueError(f"{stage} plan rows differ from their seal")
    identities = [str(row.get("physical_run_id", "")) for row in normalized]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError(f"{stage} plan has empty or duplicate physical identities")
    for row in normalized:
        if (
            str(row.get("dataset", "")) not in {"freebase", "dblp"}
            or row["start_update"] != 0
            or row["labels_mounted"] is not False
            or row["downstream_evaluations_during_training"] != 0
            or len(str(row.get("set_hash", ""))) != 64
        ):
            raise ValueError(f"{stage} plan violates the frozen no-label fresh-run boundary")
    if stage == "CANARY":
        expected = build_canary_plan()
        if normalized != expected:
            raise ValueError("CANARY plan differs from the exact four frozen rows")
    else:
        expected_ids = {
            f"M8-F3R-2G-{row['dataset'].upper()}-{row['set_hash'][:12]}-E{EVALUATION_MODEL_SEEDS.index(row['model_seed']):02d}"
            for row in normalized
        }
        if (
            any(row["model_seed"] not in EVALUATION_MODEL_SEEDS for row in normalized)
            or any(row["endpoint_update"] != ENDPOINT_UPDATE for row in normalized)
            or set(identities) != expected_ids
            or any(str(row.get("status")) not in {"EXACT_REUSE", "NEW_RUN_REQUIRED", "BLOCKED"} for row in normalized)
        ):
            raise ValueError("EVALUATION matrix violates its frozen identity grid")
    for row in normalized:
        row["plan_seal_identity"] = str(sealed["identity"])
    return normalized


def schedule_assignments(
    rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    required = {
        str(row["physical_run_id"])
        for row in rows
        if str(row.get("status")) == "NEW_RUN_REQUIRED"
    }
    assignments: dict[str, int] = {}
    group_assignments: dict[tuple[str, int], int] = {}
    for raw in schedule:
        dataset = str(raw["dataset"])
        seed = int(raw["model_seed"])
        gpu = int(raw["gpu_slot"])
        physical = [str(value) for value in json.loads(str(raw["physical_run_ids"]))]
        if gpu not in range(EXPECTED_GPU_COUNT) or len(physical) != int(raw["run_count"]):
            raise ValueError("two-GPU schedule row is malformed")
        identity = (dataset, seed)
        prior = group_assignments.setdefault(identity, gpu)
        if prior != gpu:
            raise ValueError("dataset/model-seed group was split across GPUs")
        for run_id in physical:
            if run_id in assignments:
                raise ValueError("two-GPU schedule duplicates a physical run")
            assignments[run_id] = gpu
    if set(assignments) != required:
        raise ValueError("two-GPU schedule does not cover the exact new-run matrix")
    return assignments


def attempt_root(
    result_root: str | Path,
    row: Mapping[str, Any],
    stage: str,
    attempt: int,
) -> Path:
    if stage not in STAGES or int(attempt) not in range(MAX_ATTEMPTS_PER_IDENTITY):
        raise ValueError("invalid M8-F3R-RERUN-2G stage or attempt")
    return (
        Path(result_root).resolve()
        / BRANCHES[stage]
        / str(row["physical_run_id"])
        / f"attempt_{int(attempt):02d}"
    )


def _candidate(cache_root: Path, row: Mapping[str, Any]) -> tuple[int, list[str]]:
    all_sets = read_json(cache_root / "candidates" / "all_sets.json")
    matches = []
    for value in all_sets.get("sets", []):
        set_id = str(value.get("set_id", value.get("candidate_set_id", "")))
        set_hash = str(value.get("set_sha256", value.get("candidate_set_hash", "")))
        paths = [str(item) for item in value.get("path_ids", [])]
        if set_id == str(row["set_id"]) and set_hash == str(row["set_hash"]):
            matches.append(
                (
                    int(value.get("set_ordinal", value.get("ordinal", value.get("candidate_set_ordinal", -1)))),
                    paths,
                )
            )
    expected_paths = [str(item) for item in json.loads(str(row["path_ids"]))]
    if len(matches) != 1 or matches[0][0] < 0 or matches[0][1] != expected_paths:
        raise ValueError("plan candidate does not resolve from the sealed cache bank")
    return matches[0]


def _cache_metadata(cache_value: str | Path, dataset: str) -> dict[str, Any]:
    raw = Path(cache_value).absolute()
    if raw.is_symlink():
        raise ValueError("cache root cannot be a symlink")
    root = raw.resolve(strict=True)
    complete = read_json(root / "SANITIZED_CACHE_COMPLETE.json")
    data = read_json(root / "metadata" / "data_identity.json")
    projection = read_json(root / "metadata" / "projection_identity.json")
    all_sets = read_json(root / "candidates" / "all_sets.json")
    source_identity = str(data.get("source_identity_sha256", data.get("structural_data_identity", "")))
    bank_identity = str(all_sets.get("all_sets_sha256", all_sets.get("all_sets_identity", "")))
    projection_identity = str(projection.get("projection_identity", canonical_json_hash(projection)))
    if (
        complete.get("status") != "COMPLETE"
        or complete.get("cache_identity") != CACHE_IDENTITIES[dataset]
        or source_identity != DATASET_IDENTITIES[dataset]
        or bank_identity != CANDIDATE_BANK_IDENTITIES[dataset]
    ):
        raise ValueError("sealed cache metadata identity mismatch")
    return {
        "root": root,
        "cache_identity": str(complete["cache_identity"]),
        "source_data_identity": source_identity,
        "projection_identity": _hex64(projection_identity, "projection_identity"),
        "candidate_bank_identity": bank_identity,
    }


def _path_set_role(row: Mapping[str, Any]) -> str:
    role = str(row.get("path_set_role", ""))
    if role in {"MANUAL", "FULL_FROZEN"}:
        return role
    return "ABLATION_UNIQUE_SET"


def build_run_config(
    *,
    row: Mapping[str, Any],
    stage: str,
    cache_root: str | Path,
    output_root: str | Path,
    source_commit: str,
    source_tree: str,
    plan_seal_identity: str,
) -> O0RunConfig:
    normalized = _canonical_row(row, stage=stage)
    dataset = str(normalized["dataset"])
    metadata = _cache_metadata(cache_root, dataset)
    ordinal, paths = _candidate(metadata["root"], normalized)
    endpoint = CANARY_ENDPOINT_UPDATE if stage == "CANARY" else ENDPOINT_UPDATE
    checkpoint_schedule = (endpoint,) if stage == "CANARY" else (1000, 2000, ENDPOINT_UPDATE)
    config = O0RunConfig(
        dataset=dataset,
        path_set_role=_path_set_role(normalized),
        path_set_id=str(normalized["set_id"]),
        path_ids=tuple(paths),
        candidate_set_hash=str(normalized["set_hash"]),
        model_seed=int(normalized["model_seed"]),
        max_updates=endpoint,
        capture_mode="on",
        checkpoint_schedule=checkpoint_schedule,
        cache_root=str(metadata["root"]),
        output_root=str(Path(output_root).absolute()),
        device="cuda:0",
        source_commit=_hex40(source_commit, "source_commit"),
        source_tree=_hex40(source_tree, "source_tree"),
        cache_identity=metadata["cache_identity"],
        augmentation_namespace=f"M8_F3R_RERUN_2G::{normalized['physical_run_id']}",
        run_role=RUN_ROLES[stage],
        eligible_for_scientific_result=stage == "EVALUATION",
        eligible_for_selector_development=False,
        physical_run_id=str(normalized["physical_run_id"]),
        candidate_set_ordinal=ordinal,
        source_data_identity=metadata["source_data_identity"],
        projection_identity=metadata["projection_identity"],
        candidate_bank_hash=metadata["candidate_bank_identity"],
        plan_seal_hash=_hex64(plan_seal_identity, "plan_seal_identity"),
        representation_schedule=(endpoint,),
    )
    config.validate()
    return config


def _gpu_runtime(*, worker_index: int, enforce_target_gpu: bool) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid = os.environ.get("M8_F3R_RERUN_2G_GPU_UUID", "")
    pci = os.environ.get("M8_F3R_RERUN_2G_GPU_PCI_BUS_ID", "")
    declared_worker = os.environ.get("M8_F3R_RERUN_2G_WORKER_INDEX", "")
    runtime = os.environ.get("M8_F3R_RERUN_2G_RUNTIME_DOMAIN", socket.gethostname())
    if enforce_target_gpu:
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError("M8 worker must see exactly one local CUDA device 0")
        name = torch.cuda.get_device_name(0)
        if (
            "4090" not in name
            or not uuid.startswith("GPU-")
            or not pci
            or visible != str(worker_index)
            or declared_worker != str(worker_index)
        ):
            raise RuntimeError("M8 worker lacks the frozen RTX4090 UUID/PCI/visibility identity")
    else:
        name = os.environ.get("M8_F3R_RERUN_2G_GPU_NAME", "CPU_TEST_DOUBLE")
        uuid = uuid or "GPU-CPU-TEST-DOUBLE"
        pci = pci or "0000:00:00.0"
        visible = visible or "CPU_TEST_DOUBLE"
    return {
        "gpu_uuid": uuid,
        "gpu_pci_bus_id": pci,
        "gpu_name": name,
        "cuda_visible_devices": visible,
        "runtime_domain": runtime,
        "torch_cuda_device_count_inside_worker": int(torch.cuda.device_count()) if enforce_target_gpu else 1,
        "torch_current_local_device": int(torch.cuda.current_device()) if enforce_target_gpu else 0,
    }


def _artifact(receipt: Mapping[str, Any], collection: str, update: int, root: Path, branch: str) -> dict[str, Any]:
    rows = [row for row in receipt.get(collection, []) if int(row.get("update", -1)) == int(update)]
    if len(rows) != 1:
        raise ValueError(f"sidecar lacks the exact {collection} endpoint artifact")
    row = rows[0]
    path = root / branch / str(row["path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise ValueError(f"sidecar {collection} artifact identity changed")
    return {
        "relative_path": path.resolve(strict=True).relative_to(root.resolve()).as_posix(),
        "sha256": str(row["sha256"]),
        "size_bytes": int(row["size_bytes"]),
    }


def _validated_run_started_evidence(
    result_root: str | Path,
    row: Mapping[str, Any],
    stage: str,
    *,
    attempt: int,
    worker_index: int,
    plan_seal_identity: str,
    source_commit: str,
    source_tree: str,
    run_started_identity: str,
) -> dict[str, Any]:
    queue = Path(result_root).resolve() / "dynamic_queue" / stage.casefold()
    suffix = f"{row['physical_run_id']}__attempt_{attempt:02d}"
    claim = read_json(queue / "claims" / f"{suffix}.claim")
    event = read_json(queue / "run_events" / f"{suffix}_STARTED.json")
    claim_basis = claim.get("basis", {})
    event_basis = event.get("basis", {})
    expected = {
        "physical_run_id": row["physical_run_id"],
        "logical_stage": stage,
        "dataset": row["dataset"],
        "set_hash": row["set_hash"],
        "model_seed": int(row["model_seed"]),
        "endpoint_update": int(row["endpoint_update"]),
        "attempt_index": int(attempt),
        "worker_index": int(worker_index),
        "plan_seal_identity": plan_seal_identity,
        "source_commit": source_commit,
        "source_tree": source_tree,
    }
    if (
        attempt not in range(MAX_ATTEMPTS_PER_IDENTITY)
        or claim.get("schema_version")
        != "metachorus.v53.m8-f3r-rerun-2g.queue-claim.v1"
        or claim.get("status") != "CLAIMED"
        or claim.get("identity") != canonical_json_hash(claim_basis)
        or event.get("schema_version")
        != "metachorus.v53.m8-f3r-rerun-2g.run-started.v1"
        or event.get("status") != "STARTED"
        or event.get("identity") != canonical_json_hash(event_basis)
        or event.get("identity") != run_started_identity
        or event_basis.get("claim_identity") != claim.get("identity")
        or any(claim_basis.get(key) != value for key, value in expected.items())
        or any(event_basis.get(key) != value for key, value in expected.items())
        or any(event_basis.get(key) != value for key, value in claim_basis.items())
        or event_basis.get("labels_mounted_during_training") is not False
        or int(event_basis.get("downstream_evaluations_during_training", -1)) != 0
        or event_basis.get("one_process_per_gpu") is not True
        or event_basis.get("ddp_used") is not False
        or event_basis.get("data_parallel_used") is not False
        or event_basis.get("torchrun_used") is not False
    ):
        raise ValueError("M8 run STARTED/claim identity chain is invalid")
    return event


def completed_marker(
    result_root: str | Path, row: Mapping[str, Any], stage: str
) -> tuple[Path, dict[str, Any]] | None:
    parent = Path(result_root).resolve() / BRANCHES[stage] / str(row["physical_run_id"])
    matches = [
        path / MARKER_NAMES[stage]
        for path in sorted(parent.glob("attempt_*"))
        if (path / MARKER_NAMES[stage]).is_file() and not (path / MARKER_NAMES[stage]).is_symlink()
    ]
    if len(matches) > 1:
        raise RuntimeError("multiple governed completion markers claim one physical identity")
    if not matches:
        return None
    value = read_json(matches[0])
    basis = value.get("basis", {})
    attempt = int(matches[0].parent.name.removeprefix("attempt_"))
    _validated_run_started_evidence(
        result_root,
        row,
        stage,
        attempt=attempt,
        worker_index=int(basis.get("worker_index", -1)),
        plan_seal_identity=str(basis.get("plan_seal_identity", "")),
        source_commit=str(basis.get("source_commit", "")),
        source_tree=str(basis.get("source_tree", "")),
        run_started_identity=str(basis.get("run_started_identity", "")),
    )
    if (
        value.get("schema_version") != "metachorus.v53.m8-f3r-rerun-2g.run-complete.v1"
        or value.get("status") != "PASS"
        or value.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != stage
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or basis.get("dataset") != row.get("dataset")
        or basis.get("set_hash") != row.get("set_hash")
        or int(basis.get("model_seed", -1)) != int(row.get("model_seed", -2))
        or int(basis.get("endpoint_update", -1)) != int(row.get("endpoint_update", -2))
        or (
            row.get("plan_seal_identity") is not None
            and basis.get("plan_seal_identity") != row.get("plan_seal_identity")
        )
        or int(basis.get("attempt_index", -1)) != attempt
        or attempt not in range(MAX_ATTEMPTS_PER_IDENTITY)
        or basis.get("labels_mounted_during_training") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
        or basis.get("one_process_per_gpu") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
    ):
        raise ValueError("M8 completion marker identity is invalid")
    return matches[0], value


def run_row(
    *,
    plan_row: Mapping[str, Any],
    stage: str,
    worker_index: int,
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    plan_seal_identity: str,
    run_started_identity: str,
    attempt: int = 0,
    engineering_retry_of_attempt: str = "",
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    row = _canonical_row(plan_row, stage=stage)
    existing = completed_marker(result_root, row, stage)
    if existing is not None:
        return existing[1]
    _validated_run_started_evidence(
        result_root,
        row,
        stage,
        attempt=attempt,
        worker_index=worker_index,
        plan_seal_identity=plan_seal_identity,
        source_commit=source_commit,
        source_tree=source_tree,
        run_started_identity=run_started_identity,
    )
    root = attempt_root(result_root, row, stage, attempt)
    if root.exists():
        raise FileExistsError(root)
    config = build_run_config(
        row=row,
        stage=stage,
        cache_root=cache_root,
        output_root=root,
        source_commit=source_commit,
        source_tree=source_tree,
        plan_seal_identity=plan_seal_identity,
    )
    gpu = _gpu_runtime(worker_index=worker_index, enforce_target_gpu=enforce_target_gpu)
    started = datetime.now(timezone.utc)
    session = O0TrainingSession(
        config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu
    )
    engine = session.run()
    ended = datetime.now(timezone.utc)
    if engine.get("status") != "PASS" or int(engine.get("optimizer_steps", -1)) != config.max_updates:
        raise RuntimeError("frozen O0 engine did not complete the exact endpoint")
    write_json_once(root / "RESOLVED_CONFIG.json", asdict(config))
    engine_path = root / "RUN_COMPLETE.json"
    sidecar_path = root / "SIDECAR_CAPTURE_RECEIPT.json"
    if engine_path.is_symlink() or sidecar_path.is_symlink() or not engine_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError("frozen O0 engine lacks its governed receipts")
    sidecar = read_json(sidecar_path)
    if (
        sidecar.get("schema_version") != "metachorus.v53.o0-sidecar-capture.v1"
        or sidecar.get("labels_read") is not False
        or sidecar.get("downstream_evaluation_executed") is not False
    ):
        raise ValueError("O0 sidecar boundary is invalid")
    checkpoint = _artifact(sidecar, "checkpoints", config.max_updates, root, "checkpoints")
    representation = _artifact(sidecar, "representations", config.max_updates, root, "representations")
    checkpoint_roundtrip_status = "NOT_REQUIRED"
    if stage == "CANARY":
        roundtrip = session.resume(root / checkpoint["relative_path"])
        if int(roundtrip.get("payload", {}).get("exact_update", -1)) != CANARY_ENDPOINT_UPDATE:
            raise RuntimeError("canary checkpoint write/read roundtrip failed")
        checkpoint_roundtrip_status = "PASS"
    basis = {
        "logical_stage": stage,
        "physical_run_id": row["physical_run_id"],
        "dataset": row["dataset"],
        "dataset_identity": DATASET_IDENTITIES[str(row["dataset"])],
        "set_id": row["set_id"],
        "set_hash": row["set_hash"],
        "path_ids": json.loads(row["path_ids"]),
        "model_seed": row["model_seed"],
        "attempt_index": int(attempt),
        "start_update": 0,
        "endpoint_update": config.max_updates,
        "plan_seal_identity": plan_seal_identity,
        "run_started_identity": run_started_identity,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "run_config_identity": config.identity(),
        "engine_marker_relative_path": engine_path.resolve(strict=True).relative_to(Path(result_root).resolve()).as_posix(),
        "engine_marker_sha256": sha256_file(engine_path),
        "engine_config_hash": str(engine["config_hash"]),
        "checkpoint_relative_path": (root / checkpoint["relative_path"]).resolve(strict=True).relative_to(Path(result_root).resolve()).as_posix(),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_size_bytes": checkpoint["size_bytes"],
        "checkpoint_roundtrip_status": checkpoint_roundtrip_status,
        "representation_relative_path": (root / representation["relative_path"]).resolve(strict=True).relative_to(Path(result_root).resolve()).as_posix(),
        "representation_sha256": representation["sha256"],
        "representation_size_bytes": representation["size_bytes"],
        "model_total_seconds": float(engine["model_total_seconds"]),
        "peak_gpu_allocated_bytes": int(engine["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(engine["peak_gpu_reserved_bytes"]),
        "worker_index": int(worker_index),
        **gpu,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "engineering_retry_of_attempt": engineering_retry_of_attempt,
        "scientific_metric_triggered_retry": False,
        "labels_mounted_during_training": False,
        "downstream_evaluations_during_training": 0,
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
    }
    marker = envelope("metachorus.v53.m8-f3r-rerun-2g.run-complete.v1", "PASS", basis)
    write_json_once(root / MARKER_NAMES[stage], marker)
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


def _failure_fingerprint(error: BaseException) -> str:
    return canonical_json_hash(
        {
            "exception_type": type(error).__name__,
            "message": str(error),
            "last_traceback_line": traceback.format_exc().strip().splitlines()[-1],
        }
    )


def _validated_failure(queue: Path, row: Mapping[str, Any], stage: str, seal_identity: str, attempt: int) -> dict[str, Any] | None:
    path = queue / "failures" / f"{row['physical_run_id']}__attempt_{attempt:02d}.json"
    if not path.exists():
        return None
    value = read_json(path)
    basis = value.get("basis", {})
    if (
        path.is_symlink()
        or value.get("schema_version") != "metachorus.v53.m8-f3r-rerun-2g.run-failure.v1"
        or value.get("status") != "FAILED_PRESERVED"
        or value.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != stage
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or int(basis.get("attempt_index", -1)) != attempt
        or basis.get("plan_seal_identity") != seal_identity
        or basis.get("retry_allowed") != (attempt + 1 < MAX_ATTEMPTS_PER_IDENTITY)
        or basis.get("scientific_metric_triggered_retry") is not False
    ):
        raise ValueError("failure receipt identity is invalid")
    return value


def _retry_attempt(queue: Path, row: Mapping[str, Any], stage: str, seal_identity: str) -> tuple[int, str] | None:
    first = _validated_failure(queue, row, stage, seal_identity, 0)
    if first is None:
        return 0, ""
    second = _validated_failure(queue, row, stage, seal_identity, 1)
    if second is not None:
        return None
    return 1, str(first["identity"])


def _common_root_pause(queue: Path) -> dict[str, Any] | None:
    fingerprints: dict[str, set[str]] = {}
    for path in sorted((queue / "failures").glob("*.json")):
        value = read_json(path)
        basis = value.get("basis", {})
        fingerprints.setdefault(str(basis.get("error_fingerprint", "")), set()).add(str(basis.get("physical_run_id", "")))
    offenders = sorted(
        identity for identity, runs in fingerprints.items() if identity and len(runs) >= COMMON_ROOT_CAUSE_PAUSE_THRESHOLD
    )
    if not offenders:
        return None
    value = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.common-root-pause.v1",
        "PAUSED",
        {
            "status": "COMMON_ROOT_CAUSE_PAUSE",
            "error_fingerprints": offenders,
            "threshold": COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
            "new_claims_allowed": False,
        },
    )
    write_json_once(queue / "M8_F3R_RERUN_2G_COMMON_ROOT_CAUSE_PAUSE.json", value)
    return value


def run_dynamic_worker(
    *,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    stage: str,
    worker_index: int,
    worker_count: int,
    repo_root: str | Path,
    cache_roots: Mapping[str, str | Path],
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    schedule: Sequence[Mapping[str, Any]] = (),
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    if worker_count != EXPECTED_GPU_COUNT or worker_index not in range(EXPECTED_GPU_COUNT):
        raise ValueError("M8 dynamic queue requires exactly two workers indexed 0..1")
    if set(cache_roots) != {"freebase", "dblp"}:
        raise ValueError("worker cache roots must cover exactly Freebase and DBLP")
    plan = validate_plan(rows, seal, stage=stage)
    assignments = (
        {str(row["physical_run_id"]): int(row["gpu_slot"]) for row in plan}
        if stage == "CANARY"
        else schedule_assignments(plan, schedule)
    )
    result = Path(result_root).resolve()
    queue = result / "dynamic_queue" / stage.casefold()
    pause_path = queue / "M8_F3R_RERUN_2G_COMMON_ROOT_CAUSE_PAUSE.json"
    gpu = _gpu_runtime(worker_index=worker_index, enforce_target_gpu=enforce_target_gpu)
    nonce = os.environ.get("M8_F3R_RERUN_2G_LAUNCH_NONCE", "LOCAL_TEST_RUN")
    worker_basis = {
        "logical_stage": stage,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "plan_seal_identity": seal["identity"],
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "launch_nonce": nonce,
        "started_at": utc_now(),
        **gpu,
        "labels_mounted_during_training": False,
    }
    receipts = queue / "workers" / nonce
    write_json_once(
        receipts / f"worker_{worker_index}_STARTED.json",
        envelope("metachorus.v53.m8-f3r-rerun-2g.worker-started.v1", "STARTED", worker_basis),
    )
    completed = claimed = failed = 0
    executable = [
        row for row in plan if stage == "CANARY" or str(row.get("status")) == "NEW_RUN_REQUIRED"
    ]
    for row in sorted(executable, key=lambda value: str(value["physical_run_id"])):
        if pause_path.exists():
            break
        if assignments[str(row["physical_run_id"])] != worker_index:
            continue
        if completed_marker(result, row, stage) is not None:
            continue
        retry = _retry_attempt(queue, row, stage, str(seal["identity"]))
        if retry is None:
            continue
        attempt, retry_of = retry
        claim_path = queue / "claims" / f"{row['physical_run_id']}__attempt_{attempt:02d}.claim"
        claim_basis = {
            "physical_run_id": row["physical_run_id"],
            "logical_stage": stage,
            "dataset": row["dataset"],
            "set_hash": row["set_hash"],
            "model_seed": int(row["model_seed"]),
            "endpoint_update": int(row["endpoint_update"]),
            "attempt_index": attempt,
            "worker_index": worker_index,
            "plan_seal_identity": seal["identity"],
            "source_commit": source_commit,
            "source_tree": source_tree,
            "claimed_at": utc_now(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "launch_nonce": nonce,
        }
        claim = envelope(
            "metachorus.v53.m8-f3r-rerun-2g.queue-claim.v1",
            "CLAIMED",
            claim_basis,
        )
        if not _atomic_claim(claim_path, claim):
            continue
        claimed += 1
        run_started = envelope(
            "metachorus.v53.m8-f3r-rerun-2g.run-started.v1",
            "STARTED",
            {
                **claim_basis,
                "claim_identity": claim["identity"],
                "labels_mounted_during_training": False,
                "downstream_evaluations_during_training": 0,
                "one_process_per_gpu": True,
                "ddp_used": False,
                "data_parallel_used": False,
                "torchrun_used": False,
            },
        )
        write_json_once(
            queue
            / "run_events"
            / f"{row['physical_run_id']}__attempt_{attempt:02d}_STARTED.json",
            run_started,
        )
        started = datetime.now(timezone.utc)
        try:
            run_row(
                plan_row=row,
                stage=stage,
                worker_index=worker_index,
                repo_root=repo_root,
                cache_root=cache_roots[str(row["dataset"])],
                result_root=result,
                source_commit=source_commit,
                source_tree=source_tree,
                plan_seal_identity=str(seal["identity"]),
                run_started_identity=str(run_started["identity"]),
                attempt=attempt,
                engineering_retry_of_attempt=retry_of,
                enforce_target_gpu=enforce_target_gpu,
            )
            completed += 1
        except BaseException as error:
            failed += 1
            ended = datetime.now(timezone.utc)
            basis = {
                "logical_stage": stage,
                "physical_run_id": row["physical_run_id"],
                "attempt_index": attempt,
                "worker_index": worker_index,
                "plan_seal_identity": seal["identity"],
                "run_started_identity": run_started["identity"],
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_fingerprint": _failure_fingerprint(error),
                "invocation_started_at": started.isoformat(),
                "invocation_ended_at": ended.isoformat(),
                "invocation_wall_seconds": (ended - started).total_seconds(),
                "gpu_uuid": gpu["gpu_uuid"],
                "gpu_pci_bus_id": gpu["gpu_pci_bus_id"],
                "runtime_domain": gpu["runtime_domain"],
                "retry_allowed": attempt + 1 < MAX_ATTEMPTS_PER_IDENTITY,
                "scientific_metric_triggered_retry": False,
            }
            write_json_once(
                queue / "failures" / f"{row['physical_run_id']}__attempt_{attempt:02d}.json",
                envelope("metachorus.v53.m8-f3r-rerun-2g.run-failure.v1", "FAILED_PRESERVED", basis),
            )
            _common_root_pause(queue)
            if pause_path.exists():
                break
    final_basis = {
        **worker_basis,
        "ended_at": utc_now(),
        "claimed_run_count": claimed,
        "completed_run_count": completed,
        "failed_run_count": failed,
        "queue_paused": pause_path.is_file(),
    }
    receipt = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.worker-complete.v1",
        "PASS" if failed == 0 else "FAILED_PRESERVED",
        final_basis,
    )
    write_json_once(receipts / f"worker_{worker_index}_COMPLETE.json", receipt)
    return receipt


__all__ = [
    "STAGES",
    "MARKER_NAMES",
    "validate_plan",
    "schedule_assignments",
    "attempt_root",
    "build_run_config",
    "completed_marker",
    "run_row",
    "run_dynamic_worker",
]
