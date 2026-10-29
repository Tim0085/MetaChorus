"""Exact H7 state-copy canaries and strict R4 continuations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from metachorus_v53.audit.common import (
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
    utc_now,
)
from metachorus_v53.m4a_h4.continuation import checkpoint_state_audit
from metachorus_v53.m4a_h5.continuation import H5ContinuationSession
from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig

from .constants import (
    CANARY_CHECKPOINT_SCHEDULE,
    CANARY_ENDPOINT,
    DEVELOPMENT_MODEL_SEEDS,
    PARENT_UPDATE,
    R4_CHECKPOINT_SCHEDULE,
    R4_ENDPOINT,
)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def json_loads_list(value: Any) -> list[str]:
    observed = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(observed, list) or not all(isinstance(item, str) for item in observed):
        raise ValueError("path_ids_json is not a string list")
    return list(observed)


def validate_parent_template_config(
    parent_attempt_root: Path, expected_sha256: str
) -> Path:
    """Bind the runtime template to the immutable config recorded in the plan."""

    path = parent_attempt_root / "RESOLVED_CONFIG.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("H7 parent attempt lacks a regular RESOLVED_CONFIG.json")
    if sha256_file(path) != expected_sha256:
        raise ValueError("H7 parent template config hash differs from the sealed plan")
    return path


def validate_parent_boundary_marker(marker: Mapping[str, Any]) -> None:
    """Require an explicit successful update-2000 pause/stage boundary."""

    observed_update = marker.get("endpoint_update", marker.get("stop_update"))
    if (
        int(observed_update or -1) != PARENT_UPDATE
        or int(marker.get("optimizer_steps", -1)) != PARENT_UPDATE
        or marker.get("status") not in {"PASS", "PAUSED_AT_CHECKPOINT"}
    ):
        raise RuntimeError("H7 parent marker is not an exact update-2000 boundary")


def build_h7_config(
    template: Mapping[str, Any],
    *,
    dataset_cache_root: Path,
    output_root: Path,
    physical_run_id: str,
    candidate_set_id: str,
    candidate_set_hash: str,
    candidate_set_ordinal: int,
    path_ids: list[str] | tuple[str, ...],
    model_seed: int,
    plan_seal_sha256: str,
    run_role: str,
) -> O0RunConfig:
    """Build an H7 envelope while preserving the frozen scientific lineage."""

    if int(model_seed) not in DEVELOPMENT_MODEL_SEEDS:
        raise ValueError("H7 configuration received an unapproved development seed")
    if run_role not in {"M4A_H7_ENGINEERING_CANARY", "M4A_H7_R4_ROBUST_SHORTLIST"}:
        raise ValueError(run_role)
    canary = run_role == "M4A_H7_ENGINEERING_CANARY"
    value = dict(template)
    value.update(
        {
            "dataset": str(template.get("dataset", "")).casefold(),
            "cache_root": str(dataset_cache_root.resolve()),
            "output_root": str(output_root.resolve()),
            "physical_run_id": physical_run_id,
            "path_set_role": "Candidate",
            "path_set_id": candidate_set_id,
            "candidate_set_hash": candidate_set_hash,
            "candidate_set_ordinal": int(candidate_set_ordinal),
            "path_ids": list(path_ids),
            "model_seed": int(model_seed),
            "max_updates": CANARY_ENDPOINT if canary else R4_ENDPOINT,
            "checkpoint_schedule": list(
                CANARY_CHECKPOINT_SCHEDULE if canary else R4_CHECKPOINT_SCHEDULE
            ),
            "capture_mode": "on",
            "run_role": run_role,
            "eligible_for_scientific_result": not canary,
            "eligible_for_selector_development": not canary,
            "plan_seal_hash": plan_seal_sha256,
        }
    )
    config = O0RunConfig.from_mapping(value)
    config.validate()
    return config


def _copy_parent_attempt(
    *,
    parent_attempt_root: Path,
    run_root: Path,
    config: O0RunConfig,
    expected_checkpoint_sha256: str,
    expected_template_config_sha256: str,
    expected_parent_physical_run_id: str,
) -> dict[str, Any]:
    parent_config_path = validate_parent_template_config(
        parent_attempt_root, expected_template_config_sha256
    )
    receipt_path = run_root / "H7_CONTINUATION_BOOTSTRAP.json"
    if run_root.exists():
        if not receipt_path.is_file():
            raise FileExistsError(run_root)
        value = read_json(receipt_path)
        if value.get("basis", {}).get("h7_config_hash") != config.identity():
            raise ValueError("existing H7 bootstrap config identity differs")
        if value.get("basis", {}).get("parent_template_config_sha256") != expected_template_config_sha256:
            raise ValueError("existing H7 bootstrap parent config identity differs")
        return value
    run_root.mkdir(parents=True)
    try:
        for directory in ("checkpoints", "representations"):
            source = parent_attempt_root / directory
            destination = run_root / directory
            destination.mkdir()
            for member in source.iterdir():
                if not member.is_file() or member.is_symlink():
                    continue
                try:
                    update = int(member.stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                if update <= PARENT_UPDATE:
                    shutil.copy2(member, destination / member.name)
        shutil.copy2(parent_attempt_root / "RESOURCE_TRACE.csv", run_root / "RESOURCE_TRACE.csv")
        sidecar_receipt = _copy_parent_sidecar_receipts(
            parent_attempt_root=parent_attempt_root,
            run_root=run_root,
        )
        parent_config = read_json(parent_config_path)
        atomic_write_json(run_root / "PARENT_RESOLVED_CONFIG.json", parent_config)
        atomic_write_json(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
        checkpoint = run_root / "checkpoints" / f"update_{PARENT_UPDATE:06d}.pt"
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=PARENT_UPDATE,
            expected_sha256=expected_checkpoint_sha256,
        )
        if not audit["strict_state_complete"]:
            raise RuntimeError("H7 continuation parent lacks strict state")
        marker_candidates = (
            parent_attempt_root / f"H5_STAGE_{PARENT_UPDATE:06d}_COMPLETE.json",
            parent_attempt_root / f"H6_STAGE_{PARENT_UPDATE:06d}_COMPLETE.json",
            parent_attempt_root / f"RUN_PAUSED_{PARENT_UPDATE:06d}.json",
        )
        marker_path = next((path for path in marker_candidates if path.is_file()), None)
        if marker_path is None:
            raise RuntimeError("H7 parent attempt lacks an exact update-2000 marker")
        parent_marker = read_json(marker_path)
        validate_parent_boundary_marker(parent_marker)
        paused = {
            "schema_version": "metachorus.v53.m4a-h7.bootstrap-paused-run.v1",
            "created_at_utc": utc_now(),
            "status": "PAUSED_AT_CHECKPOINT",
            "config_hash": config.identity(),
            "physical_run_id": config.physical_run_id,
            "start_update": 1,
            "stop_update": PARENT_UPDATE,
            "optimizer_steps": PARENT_UPDATE,
            "optimizer_steps_this_invocation": 0,
            "wall_time_seconds": float(
                parent_marker.get("wall_time_seconds", parent_marker.get("model_total_seconds", 0.0))
            ),
            "model_total_seconds": float(parent_marker.get("model_total_seconds", 0.0)),
            "peak_gpu_allocated_bytes": int(parent_marker.get("peak_gpu_allocated_bytes", 0)),
            "peak_gpu_reserved_bytes": int(parent_marker.get("peak_gpu_reserved_bytes", 0)),
            "run_role": config.run_role,
            "eligible_for_scientific_result": config.eligible_for_scientific_result,
            "eligible_for_selector_development": config.eligible_for_selector_development,
            "labels_mounted": False,
            "downstream_evaluations": 0,
            "continuation_from_checkpoint": None,
            "continuation_same_physical_run": True,
            "bootstrap_parent_physical_run_id": expected_parent_physical_run_id,
            "trace_sha256": sha256_file(run_root / "RESOURCE_TRACE.csv"),
        }
        atomic_write_json(run_root / f"RUN_PAUSED_{PARENT_UPDATE:06d}.json", paused)
        basis = {
            "parent_physical_run_id": expected_parent_physical_run_id,
            "h7_physical_run_id": config.physical_run_id,
            "boundary_update": PARENT_UPDATE,
            "parent_checkpoint_sha256": expected_checkpoint_sha256,
            "parent_template_config_sha256": expected_template_config_sha256,
            "copied_checkpoint_sha256": sha256_file(checkpoint),
            "parent_trace_sha256": sha256_file(parent_attempt_root / "RESOURCE_TRACE.csv"),
            "copied_trace_sha256": sha256_file(run_root / "RESOURCE_TRACE.csv"),
            "parent_sidecar_receipt_name": sidecar_receipt["source_name"],
            "parent_sidecar_receipt_sha256": sidecar_receipt["source_sha256"],
            "copied_sidecar_receipt_name": sidecar_receipt["destination_name"],
            "copied_sidecar_receipt_sha256": sidecar_receipt["destination_sha256"],
            "h7_config_hash": config.identity(),
            "strict_state_complete": True,
            "parent_immutable": True,
        }
        receipt = {
            "schema_version": "metachorus.v53.m4a-h7.continuation-bootstrap.v1",
            "status": "PASS",
            "basis": basis,
            "identity": canonical_json_hash(basis),
        }
        atomic_write_json(receipt_path, receipt)
        return receipt
    except Exception:
        shutil.rmtree(run_root, ignore_errors=True)
        raise


def _copy_parent_sidecar_receipts(
    *,
    parent_attempt_root: Path,
    run_root: Path,
) -> dict[str, str]:
    """Copy the exact update-2000 sidecar receipt without occupying the H7 endpoint name."""
    boundary_name = f"SIDECAR_CAPTURE_RECEIPT_PART_{PARENT_UPDATE:06d}.json"
    boundary_source = parent_attempt_root / boundary_name
    complete_source = parent_attempt_root / "SIDECAR_CAPTURE_RECEIPT.json"
    if boundary_source.is_file() and not boundary_source.is_symlink():
        source = boundary_source
    elif complete_source.is_file() and not complete_source.is_symlink():
        source = complete_source
    else:
        raise RuntimeError("H7 parent attempt lacks an exact update-2000 sidecar receipt")
    destination = run_root / boundary_name
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    return {
        "source_name": source.name,
        "source_sha256": sha256_file(source),
        "destination_name": destination.name,
        "destination_sha256": sha256_file(destination),
    }


_MARKER_BASIS_FIELDS = {
    "physical_run_id",
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "model_seed",
    "continuation_from_update",
    "endpoint_update",
    "optimizer_steps",
    "optimizer_steps_this_invocation",
    "model_total_seconds",
    "model_seconds_this_invocation",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "checkpoint_sha256",
    "representation_sha256",
    "continuation_mode",
    "run_role",
    "plan_seal_sha256",
    "resume_audit",
    "eligible_for_scientific_result",
    "eligible_for_selector_development",
    "reusable_for_science",
    "labels_mounted",
    "downstream_evaluations",
    "status",
}


def validate_stage_marker(
    marker_path: Path,
    *,
    plan_row: Mapping[str, Any],
    endpoint_update: int,
    run_root: Path,
) -> dict[str, Any]:
    value = read_json(marker_path)
    basis = {key: value.get(key) for key in _MARKER_BASIS_FIELDS}
    expected_fields = _MARKER_BASIS_FIELDS | {"schema_version", "identity"}
    expected = {
        "physical_run_id": str(plan_row["physical_run_id"]),
        "dataset": str(plan_row["dataset"]),
        "candidate_set_id": str(plan_row["candidate_set_id"]),
        "candidate_set_hash": str(plan_row["candidate_set_hash"]),
        "candidate_set_ordinal": int(plan_row["candidate_set_ordinal"]),
        "model_seed": int(plan_row["model_seed"]),
        "continuation_from_update": PARENT_UPDATE,
        "endpoint_update": int(endpoint_update),
        "optimizer_steps": int(endpoint_update),
        "optimizer_steps_this_invocation": int(endpoint_update) - PARENT_UPDATE,
        "continuation_mode": str(plan_row["continuation_mode"]),
        "run_role": str(plan_row["run_role"]),
        "plan_seal_sha256": str(plan_row["plan_seal_sha256"]),
        "eligible_for_scientific_result": str(plan_row["stage"]) == "R4",
        "eligible_for_selector_development": str(plan_row["stage"]) == "R4",
        "reusable_for_science": str(plan_row["stage"]) == "R4",
        "labels_mounted": False,
        "downstream_evaluations": 0,
        "status": "PASS",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if value.get("schema_version") != "metachorus.v53.m4a-h7.stage-run-complete.v1":
        mismatches["schema_version"] = value.get("schema_version")
    if set(value) != expected_fields:
        mismatches["fields"] = {
            "missing": sorted(expected_fields - set(value)),
            "extra": sorted(set(value) - expected_fields),
        }
    if value.get("identity") != canonical_json_hash(basis):
        mismatches["identity"] = "canonical basis hash mismatch"
    checkpoint = run_root / "checkpoints" / f"update_{endpoint_update:06d}.pt"
    representation = run_root / "representations" / f"update_{endpoint_update:06d}.npy"
    if not checkpoint.is_file() or not representation.is_file():
        mismatches["endpoint_artifacts"] = "missing"
    else:
        if sha256_file(checkpoint) != value.get("checkpoint_sha256"):
            mismatches["checkpoint_sha256"] = "mismatch"
        if sha256_file(representation) != value.get("representation_sha256"):
            mismatches["representation_sha256"] = "mismatch"
    if mismatches:
        raise ValueError(f"invalid existing H7 stage marker: {mismatches}")
    return value


def run_plan_row(
    *,
    plan_row: Mapping[str, Any],
    repo_root: Path,
    cache_root: Path,
    result_root: Path,
    parent_attempt_root: Path,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    stage = str(plan_row["stage"])
    boundary = int(plan_row["continuation_from_update"])
    endpoint = int(plan_row["endpoint_update"])
    role = str(plan_row["run_role"])
    approved = {
        ("CANARY", PARENT_UPDATE, CANARY_ENDPOINT, "M4A_H7_ENGINEERING_CANARY"),
        ("R4", PARENT_UPDATE, R4_ENDPOINT, "M4A_H7_R4_ROBUST_SHORTLIST"),
    }
    if (stage, boundary, endpoint, role) not in approved:
        raise ValueError(f"unapproved H7 stage boundary: {stage}/{boundary}->{endpoint}/{role}")
    if _bool(plan_row["labels_mounted"]) or int(plan_row["downstream_evaluations_during_training"]) != 0:
        raise ValueError("H7 training plan cannot mount labels or run evaluators")
    run_id = str(plan_row["physical_run_id"])
    attempt_index = int(plan_row.get("attempt_index", 0))
    base = "canary_runs" if stage == "CANARY" else "runs"
    run_root = result_root / base / run_id / f"attempt_{attempt_index:02d}"
    expected_template_config_sha256 = str(plan_row["template_config_sha256"])
    parent_config_path = validate_parent_template_config(
        parent_attempt_root, expected_template_config_sha256
    )
    template = read_json(parent_config_path)
    config = build_h7_config(
        template,
        dataset_cache_root=cache_root / str(plan_row["dataset"]),
        output_root=run_root,
        physical_run_id=run_id,
        candidate_set_id=str(plan_row["candidate_set_id"]),
        candidate_set_hash=str(plan_row["candidate_set_hash"]),
        candidate_set_ordinal=int(plan_row["candidate_set_ordinal"]),
        path_ids=json_loads_list(plan_row["path_ids_json"]),
        model_seed=int(plan_row["model_seed"]),
        plan_seal_sha256=str(plan_row["plan_seal_sha256"]),
        run_role=role,
    )
    marker_name = (
        f"H7_CANARY_{CANARY_ENDPOINT:06d}_COMPLETE.json"
        if stage == "CANARY"
        else f"H7_STAGE_{endpoint:06d}_COMPLETE.json"
    )
    marker = run_root / marker_name
    if marker.is_file():
        return validate_stage_marker(
            marker, plan_row=plan_row, endpoint_update=endpoint, run_root=run_root
        )
    if not run_root.exists():
        _copy_parent_attempt(
            parent_attempt_root=parent_attempt_root,
            run_root=run_root,
            config=config,
            expected_checkpoint_sha256=str(plan_row["parent_checkpoint_sha256"]),
            expected_template_config_sha256=expected_template_config_sha256,
            expected_parent_physical_run_id=str(plan_row["parent_physical_run_id"]),
        )
    checkpoint = run_root / "checkpoints" / f"update_{PARENT_UPDATE:06d}.pt"
    audit = checkpoint_state_audit(
        checkpoint,
        expected_update=PARENT_UPDATE,
        expected_sha256=str(plan_row["parent_checkpoint_sha256"]),
    )
    if not audit["strict_state_complete"]:
        raise RuntimeError("H7 continuation source lacks strict state")
    session = H5ContinuationSession(
        config,
        repo_root=repo_root,
        expected_parent_physical_run_id=str(plan_row["parent_physical_run_id"]),
        expected_parent_checkpoint_sha256=str(plan_row["parent_checkpoint_sha256"]),
        expected_parent_update=PARENT_UPDATE,
        enforce_target_gpu=enforce_target_gpu,
    )
    run = session.run(resume_checkpoint=checkpoint, stop_at_update=endpoint)
    endpoint_checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    endpoint_representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    if not endpoint_checkpoint.is_file() or not endpoint_representation.is_file():
        raise RuntimeError("H7 endpoint capture is incomplete")
    resume_audit = dict(session.last_resume_audit or {})
    if "h5_physical_run_id" in resume_audit:
        resume_audit["h7_physical_run_id"] = resume_audit.pop("h5_physical_run_id")
    resume_audit["status"] = "STRICT_PARENT_TO_H7_TRANSITION_LOADED"
    basis = {
        "physical_run_id": run_id,
        "dataset": str(plan_row["dataset"]),
        "candidate_set_id": str(plan_row["candidate_set_id"]),
        "candidate_set_hash": str(plan_row["candidate_set_hash"]),
        "candidate_set_ordinal": int(plan_row["candidate_set_ordinal"]),
        "model_seed": int(plan_row["model_seed"]),
        "continuation_from_update": boundary,
        "endpoint_update": endpoint,
        "optimizer_steps": int(run["optimizer_steps"]),
        "optimizer_steps_this_invocation": int(run["optimizer_steps_this_invocation"]),
        "model_total_seconds": float(run["model_total_seconds"]),
        "model_seconds_this_invocation": float(run["wall_time_seconds_this_invocation"]),
        "peak_gpu_allocated_bytes": int(run["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(run["peak_gpu_reserved_bytes"]),
        "checkpoint_sha256": sha256_file(endpoint_checkpoint),
        "representation_sha256": sha256_file(endpoint_representation),
        "continuation_mode": str(plan_row["continuation_mode"]),
        "run_role": role,
        "plan_seal_sha256": str(plan_row["plan_seal_sha256"]),
        "resume_audit": resume_audit,
        "eligible_for_scientific_result": stage == "R4",
        "eligible_for_selector_development": stage == "R4",
        "reusable_for_science": stage == "R4",
        "labels_mounted": False,
        "downstream_evaluations": 0,
        "status": "PASS",
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h7.stage-run-complete.v1",
        **basis,
        "identity": canonical_json_hash(basis),
    }
    atomic_write_json(marker, payload)
    return payload


def read_plan_row(plan_path: Path, physical_run_id: str) -> dict[str, str]:
    with plan_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["physical_run_id"] == physical_run_id]
    if len(rows) != 1:
        raise ValueError(f"expected one H7 plan row for {physical_run_id}, observed {len(rows)}")
    return dict(rows[0])
