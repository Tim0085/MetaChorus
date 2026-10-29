"""H6 engineering canary, third-seed from-zero, and strict R4 continuation."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import subprocess
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
from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession

from .constants import (
    ALL_DEVELOPMENT_MODEL_SEEDS,
    CANARY_UPDATES,
    CHECKPOINT_SCHEDULE,
    R4_ENDPOINT,
    SEARCH_MODEL_SEEDS,
    THIRD_DEVELOPMENT_MODEL_SEED,
    THIRD_SEED_ENDPOINT,
)


def _git(repo_root: Path, value: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", value],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def h6_physical_run_id(
    stage: str,
    dataset: str,
    candidate_ordinal: int,
    model_seed: int,
) -> str:
    token_by_seed = {
        SEARCH_MODEL_SEEDS[0]: "S00",
        SEARCH_MODEL_SEEDS[1]: "S01",
        THIRD_DEVELOPMENT_MODEL_SEED: "S02",
    }
    if int(model_seed) not in token_by_seed:
        raise ValueError("unapproved H6 model seed")
    token = token_by_seed[int(model_seed)]
    prefix = {
        "CANARY": "M4A-H6-CANARY",
        "THIRD_SEED_R3_AUDIT": "M4A-H6-R3",
        "R4": "M4A-H6-R4",
    }.get(stage)
    if prefix is None:
        raise ValueError(f"unapproved H6 stage: {stage}")
    if stage in {"CANARY", "THIRD_SEED_R3_AUDIT"} and token != "S02":
        raise ValueError(f"{stage} must use the frozen third development seed")
    return f"{prefix}-{dataset.upper()}-C{int(candidate_ordinal):03d}-{token}"


def build_h6_config(
    template: Mapping[str, Any],
    *,
    repo_root: Path,
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
    preserve_parent_scientific_source: bool = False,
) -> O0RunConfig:
    if int(model_seed) not in ALL_DEVELOPMENT_MODEL_SEEDS:
        raise ValueError("H6 configuration received an unapproved development seed")
    if run_role not in {
        "M4A_H6_ENGINEERING_CANARY",
        "M4A_H6_THIRD_SEED_R3_AUDIT",
        "M4A_H6_R4_SEARCH",
    }:
        raise ValueError(run_role)
    canary = run_role == "M4A_H6_ENGINEERING_CANARY"
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
            "max_updates": CANARY_UPDATES if canary else R4_ENDPOINT,
            "checkpoint_schedule": [CANARY_UPDATES] if canary else list(CHECKPOINT_SCHEDULE),
            "capture_mode": "on",
            "run_role": run_role,
            "eligible_for_scientific_result": not canary,
            "eligible_for_selector_development": not canary,
            "plan_seal_hash": plan_seal_sha256,
        }
    )
    if not preserve_parent_scientific_source:
        value["augmentation_namespace"] = canonical_json_hash(
            {
                "stage": run_role,
                "physical_run_id": physical_run_id,
                "model_seed": int(model_seed),
            }
        )
    if not preserve_parent_scientific_source:
        value["source_commit"] = _git(repo_root, "HEAD")
        value["source_tree"] = _git(repo_root, "HEAD^{tree}")
    config = O0RunConfig.from_mapping(value)
    config.validate()
    return config


def _copy_parent_attempt(
    *,
    parent_attempt_root: Path,
    run_root: Path,
    config: O0RunConfig,
    boundary_update: int,
    expected_checkpoint_sha256: str,
    expected_parent_physical_run_id: str,
) -> dict[str, Any]:
    receipt_path = run_root / "H6_CONTINUATION_BOOTSTRAP.json"
    if run_root.exists():
        if not receipt_path.is_file():
            raise FileExistsError(run_root)
        value = read_json(receipt_path)
        if value.get("h6_config_hash") != config.identity():
            raise ValueError("existing H6 bootstrap config identity differs")
        return value
    run_root.mkdir(parents=True)
    try:
        for directory in ("checkpoints", "representations"):
            source = parent_attempt_root / directory
            destination = run_root / directory
            destination.mkdir()
            for member in source.iterdir():
                if not member.is_file():
                    continue
                try:
                    update = int(member.stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                if update <= int(boundary_update):
                    shutil.copy2(member, destination / member.name)
        shutil.copy2(parent_attempt_root / "RESOURCE_TRACE.csv", run_root / "RESOURCE_TRACE.csv")
        for source in parent_attempt_root.glob("SIDECAR_CAPTURE_RECEIPT*.json"):
            shutil.copy2(source, run_root / source.name)
        parent_config = read_json(parent_attempt_root / "RESOLVED_CONFIG.json")
        atomic_write_json(run_root / "PARENT_RESOLVED_CONFIG.json", parent_config)
        atomic_write_json(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
        checkpoint = run_root / "checkpoints" / f"update_{boundary_update:06d}.pt"
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=boundary_update,
            expected_sha256=expected_checkpoint_sha256,
        )
        if not audit["strict_state_complete"]:
            raise RuntimeError("H6 continuation parent lacks strict state")
        marker_candidates = (
            parent_attempt_root / f"H5_STAGE_{boundary_update:06d}_COMPLETE.json",
            parent_attempt_root / f"H6_STAGE_{boundary_update:06d}_COMPLETE.json",
            parent_attempt_root / f"RUN_PAUSED_{boundary_update:06d}.json",
            parent_attempt_root / "RUN_COMPLETE.json",
        )
        marker_path = next((path for path in marker_candidates if path.is_file()), None)
        if marker_path is None:
            raise RuntimeError("H6 parent attempt lacks an exact boundary marker")
        parent_marker = read_json(marker_path)
        paused = {
            "schema_version": "metachorus.v53.m4a-h6.bootstrap-paused-run.v1",
            "created_at_utc": utc_now(),
            "status": "PAUSED_AT_CHECKPOINT",
            "config_hash": config.identity(),
            "physical_run_id": config.physical_run_id,
            "start_update": 1,
            "stop_update": int(boundary_update),
            "optimizer_steps": int(boundary_update),
            "optimizer_steps_this_invocation": 0,
            "wall_time_seconds": float(
                parent_marker.get("wall_time_seconds", parent_marker.get("model_total_seconds", 0.0))
            ),
            "model_total_seconds": float(parent_marker.get("model_total_seconds", 0.0)),
            "peak_gpu_allocated_bytes": int(parent_marker.get("peak_gpu_allocated_bytes", 0)),
            "peak_gpu_reserved_bytes": int(parent_marker.get("peak_gpu_reserved_bytes", 0)),
            "run_role": config.run_role,
            "eligible_for_scientific_result": True,
            "eligible_for_selector_development": True,
            "labels_mounted": False,
            "downstream_evaluations": 0,
            "continuation_from_checkpoint": None,
            "continuation_same_physical_run": True,
            "bootstrap_parent_physical_run_id": expected_parent_physical_run_id,
            "trace_sha256": sha256_file(run_root / "RESOURCE_TRACE.csv"),
        }
        atomic_write_json(run_root / f"RUN_PAUSED_{boundary_update:06d}.json", paused)
        basis = {
            "parent_physical_run_id": expected_parent_physical_run_id,
            "h6_physical_run_id": config.physical_run_id,
            "boundary_update": int(boundary_update),
            "parent_checkpoint_sha256": expected_checkpoint_sha256,
            "copied_checkpoint_sha256": sha256_file(checkpoint),
            "parent_trace_sha256": sha256_file(parent_attempt_root / "RESOURCE_TRACE.csv"),
            "copied_trace_sha256": sha256_file(run_root / "RESOURCE_TRACE.csv"),
            "h6_config_hash": config.identity(),
            "strict_state_complete": True,
            "parent_immutable": True,
        }
        receipt = {
            "schema_version": "metachorus.v53.m4a-h6.continuation-bootstrap.v1",
            "status": "PASS",
            "basis": basis,
            "identity": canonical_json_hash(basis),
        }
        atomic_write_json(receipt_path, receipt)
        return receipt
    except Exception:
        shutil.rmtree(run_root, ignore_errors=True)
        raise


def validate_stage_marker(
    marker_path: Path,
    *,
    plan_row: Mapping[str, Any],
    endpoint_update: int,
    run_root: Path,
) -> dict[str, Any]:
    value = read_json(marker_path)
    marker_basis_fields = {
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
    expected_top_level = marker_basis_fields | {"schema_version", "identity"}
    observed_fields = set(value)
    marker_basis = {key: value.get(key) for key in marker_basis_fields}
    envelope_mismatches: dict[str, Any] = {}
    if value.get("schema_version") != "metachorus.v53.m4a-h6.stage-run-complete.v1":
        envelope_mismatches["schema_version"] = value.get("schema_version")
    if observed_fields != expected_top_level:
        envelope_mismatches["fields"] = {
            "missing": sorted(expected_top_level - observed_fields),
            "extra": sorted(observed_fields - expected_top_level),
        }
    if value.get("identity") != canonical_json_hash(marker_basis):
        envelope_mismatches["identity"] = "canonical basis hash mismatch"
    expected = {
        "physical_run_id": str(plan_row["physical_run_id"]),
        "dataset": str(plan_row["dataset"]),
        "candidate_set_id": str(plan_row["candidate_set_id"]),
        "candidate_set_hash": str(plan_row["candidate_set_hash"]),
        "candidate_set_ordinal": int(plan_row["candidate_set_ordinal"]),
        "model_seed": int(plan_row["model_seed"]),
        "continuation_from_update": int(plan_row["continuation_from_update"]),
        "endpoint_update": int(endpoint_update),
        "optimizer_steps": int(endpoint_update),
        "continuation_mode": str(plan_row["continuation_mode"]),
        "run_role": str(plan_row["run_role"]),
        "plan_seal_sha256": str(plan_row["plan_seal_sha256"]),
        "eligible_for_scientific_result": str(plan_row["stage"]) != "CANARY",
        "eligible_for_selector_development": str(plan_row["stage"]) != "CANARY",
        "reusable_for_science": str(plan_row["stage"]) != "CANARY",
        "labels_mounted": False,
        "downstream_evaluations": 0,
        "status": "PASS",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    checkpoint = run_root / "checkpoints" / f"update_{endpoint_update:06d}.pt"
    representation = run_root / "representations" / f"update_{endpoint_update:06d}.npy"
    if not checkpoint.is_file() or not representation.is_file():
        mismatches["endpoint_artifacts"] = "missing"
    else:
        if sha256_file(checkpoint) != value.get("checkpoint_sha256"):
            mismatches["checkpoint_sha256"] = "mismatch"
        if sha256_file(representation) != value.get("representation_sha256"):
            mismatches["representation_sha256"] = "mismatch"
    mismatches.update(envelope_mismatches)
    if mismatches:
        raise ValueError(f"invalid existing H6 stage marker: {mismatches}")
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
        ("CANARY", 0, 20, "M4A_H6_ENGINEERING_CANARY"),
        ("THIRD_SEED_R3_AUDIT", 0, 2000, "M4A_H6_THIRD_SEED_R3_AUDIT"),
        ("R4", 2000, 3000, "M4A_H6_R4_SEARCH"),
    }
    if (stage, boundary, endpoint, role) not in approved:
        raise ValueError(f"unapproved H6 stage boundary: {stage}/{boundary}->{endpoint}/{role}")
    run_id = str(plan_row["physical_run_id"])
    attempt_index = int(plan_row.get("attempt_index", 0))
    base = "canary_runs" if stage == "CANARY" else "runs"
    run_root = result_root / base / run_id / f"attempt_{attempt_index:02d}"
    template = read_json(parent_attempt_root / "RESOLVED_CONFIG.json")
    path_ids = json_loads_list(plan_row["path_ids_json"])
    config = build_h6_config(
        template,
        repo_root=repo_root,
        dataset_cache_root=cache_root / str(plan_row["dataset"]),
        output_root=run_root,
        physical_run_id=run_id,
        candidate_set_id=str(plan_row["candidate_set_id"]),
        candidate_set_hash=str(plan_row["candidate_set_hash"]),
        candidate_set_ordinal=int(plan_row["candidate_set_ordinal"]),
        path_ids=path_ids,
        model_seed=int(plan_row["model_seed"]),
        plan_seal_sha256=str(plan_row["plan_seal_sha256"]),
        run_role=role,
        preserve_parent_scientific_source=stage == "R4",
    )
    marker_name = (
        "H6_CANARY_000020_COMPLETE.json"
        if stage == "CANARY"
        else f"H6_STAGE_{endpoint:06d}_COMPLETE.json"
    )
    marker = run_root / marker_name
    if marker.is_file():
        return validate_stage_marker(
            marker, plan_row=plan_row, endpoint_update=endpoint, run_root=run_root
        )
    if boundary == 0:
        if run_root.exists():
            raise RuntimeError("preserve the failed H6 from-zero attempt and allocate a new attempt")
        session: O0TrainingSession
        session = O0TrainingSession(config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu)
        run = session.run(stop_at_update=endpoint)
        atomic_write_json(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
        resume_audit = None
    else:
        if not run_root.exists():
            _copy_parent_attempt(
                parent_attempt_root=parent_attempt_root,
                run_root=run_root,
                config=config,
                boundary_update=boundary,
                expected_checkpoint_sha256=str(plan_row["parent_checkpoint_sha256"]),
                expected_parent_physical_run_id=str(plan_row["parent_physical_run_id"]),
            )
        checkpoint = run_root / "checkpoints" / f"update_{boundary:06d}.pt"
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=boundary,
            expected_sha256=str(plan_row["parent_checkpoint_sha256"]),
        )
        if not audit["strict_state_complete"]:
            raise RuntimeError("H6 continuation source lacks strict state")
        session = H5ContinuationSession(
            config,
            repo_root=repo_root,
            expected_parent_physical_run_id=str(plan_row["parent_physical_run_id"]),
            expected_parent_checkpoint_sha256=str(plan_row["parent_checkpoint_sha256"]),
            expected_parent_update=boundary,
            enforce_target_gpu=enforce_target_gpu,
        )
        run = session.run(resume_checkpoint=checkpoint, stop_at_update=endpoint)
        resume_audit = session.last_resume_audit
    endpoint_checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    endpoint_representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    if not endpoint_checkpoint.is_file() or not endpoint_representation.is_file():
        raise RuntimeError("H6 endpoint capture is incomplete")
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
        "eligible_for_scientific_result": stage != "CANARY",
        "eligible_for_selector_development": stage != "CANARY",
        "reusable_for_science": stage != "CANARY",
        "labels_mounted": False,
        "downstream_evaluations": 0,
        "status": "PASS",
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h6.stage-run-complete.v1",
        **basis,
        "identity": canonical_json_hash(basis),
    }
    atomic_write_json(marker, payload)
    return payload


def json_loads_list(value: Any) -> list[str]:
    import json

    observed = json.loads(str(value)) if not isinstance(value, list) else value
    if not isinstance(observed, list) or not all(isinstance(item, str) for item in observed):
        raise ValueError("path_ids_json is not a string list")
    return list(observed)


def read_plan_row(plan_path: Path, physical_run_id: str) -> dict[str, str]:
    with plan_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["physical_run_id"] == physical_run_id]
    if len(rows) != 1:
        raise ValueError(f"expected one H6 plan row for {physical_run_id}, observed {len(rows)}")
    return dict(rows[0])
