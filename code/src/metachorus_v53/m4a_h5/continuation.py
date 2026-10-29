"""Strict H1/H4-to-H5 continuation and exact multi-fidelity boundaries."""

from __future__ import annotations

import csv
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
from metachorus_v53.training.o0_checkpoint_io import load_checkpoint
from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession

from .constants import (
    ALL_DEVELOPMENT_MODEL_SEEDS,
    CHECKPOINT_SCHEDULE,
    SEARCH_MODEL_SEEDS,
    THIRD_DEVELOPMENT_MODEL_SEED,
)


ALLOWED_LINEAGE_TRANSITIONS = {"config_hash", "physical_run_id", "plan_seal_hash"}
STABLE_LINEAGE_KEYS = (
    "source_commit",
    "source_tree",
    "model_py_sha256",
    "data_cache_hash",
    "path_set_hash",
    "model_seed",
    "augmentation_namespace",
    "trainer_id",
    "dataset",
    "candidate_set_ordinal",
    "source_data_identity",
    "projection_identity",
    "score_execution_contract_hash",
    "diagnostic_source_identity_hash",
    "diagnostic_config_hash",
    "bank_contract_hash",
    "candidate_bank_hash",
    "precision_runtime_contract_hash",
)


def h5_physical_run_id(dataset: str, candidate_ordinal: int, model_seed: int) -> str:
    seed = int(model_seed)
    if seed == SEARCH_MODEL_SEEDS[0]:
        token = "S00"
    elif seed == SEARCH_MODEL_SEEDS[1]:
        token = "S01"
    elif seed == THIRD_DEVELOPMENT_MODEL_SEED:
        token = "S02"
    else:
        raise ValueError(f"unapproved H5 model seed: {model_seed}")
    return f"M4A-H5-{dataset.upper()}-C{int(candidate_ordinal):03d}-{token}"


def build_h5_config(
    parent_resolved_config: Mapping[str, Any],
    *,
    cache_root: Path,
    output_root: Path,
    physical_run_id: str,
    plan_seal_sha256: str,
    model_seed: int,
    run_role: str,
) -> O0RunConfig:
    if int(model_seed) not in ALL_DEVELOPMENT_MODEL_SEEDS:
        raise ValueError("H5 configuration received an unapproved development seed")
    if run_role not in {"M4A_H5_SEARCH", "M4A_H5_THIRD_SEED_AUDIT"}:
        raise ValueError(run_role)
    value = dict(parent_resolved_config)
    value.update(
        {
            "cache_root": str(cache_root.resolve()),
            "output_root": str(output_root.resolve()),
            "max_updates": 3000,
            "checkpoint_schedule": list(CHECKPOINT_SCHEDULE),
            "physical_run_id": physical_run_id,
            "model_seed": int(model_seed),
            "run_role": run_role,
            "eligible_for_scientific_result": True,
            "eligible_for_selector_development": True,
            "plan_seal_hash": plan_seal_sha256,
        }
    )
    config = O0RunConfig.from_mapping(value)
    config.validate()
    return config


def validate_scientific_lineage_transition(
    *, parent_metadata: Mapping[str, Any], config: O0RunConfig, expected_parent_run_id: str
) -> dict[str, Any]:
    expected = {
        "source_commit": config.source_commit,
        "source_tree": config.source_tree,
        "data_cache_hash": config.cache_identity,
        "path_set_hash": config.candidate_set_hash,
        "model_seed": config.model_seed,
        "augmentation_namespace": config.augmentation_namespace,
        "trainer_id": config.trainer_id,
        "dataset": config.dataset,
        "candidate_set_ordinal": config.candidate_set_ordinal,
        "source_data_identity": config.source_data_identity,
        "projection_identity": config.projection_identity,
        "score_execution_contract_hash": config.score_execution_contract_hash,
        "diagnostic_source_identity_hash": config.diagnostic_source_identity_hash,
        "diagnostic_config_hash": config.diagnostic_config_hash,
        "bank_contract_hash": config.bank_contract_hash,
        "candidate_bank_hash": config.candidate_bank_hash,
        "precision_runtime_contract_hash": config.precision_runtime_contract_hash,
    }
    mismatches = {
        key: {"parent": parent_metadata.get(key), "expected": value}
        for key, value in expected.items()
        if parent_metadata.get(key) != value
    }
    if parent_metadata.get("physical_run_id") != expected_parent_run_id:
        mismatches["physical_run_id"] = {
            "parent": parent_metadata.get("physical_run_id"),
            "expected": expected_parent_run_id,
        }
    if mismatches:
        raise ValueError(f"parent-to-H5 scientific lineage mismatch: {mismatches}")
    basis = {
        "matched_keys": sorted(expected),
        "parent_physical_run_id": expected_parent_run_id,
        "h5_physical_run_id": config.physical_run_id,
        "allowed_transition_keys": sorted(ALLOWED_LINEAGE_TRANSITIONS),
        "same_candidate_seed_scientific_identity": True,
    }
    return {**basis, "identity": canonical_json_hash(basis)}


class H5ContinuationSession(O0TrainingSession):
    """Permit one audited envelope transition at an exact H1/H4 boundary."""

    def __init__(
        self,
        config: O0RunConfig,
        *,
        repo_root: str | Path,
        expected_parent_physical_run_id: str | None = None,
        expected_parent_checkpoint_sha256: str | None = None,
        expected_parent_update: int | None = None,
        enforce_target_gpu: bool = True,
    ):
        self.expected_parent_physical_run_id = expected_parent_physical_run_id
        self.expected_parent_checkpoint_sha256 = expected_parent_checkpoint_sha256
        self.expected_parent_update = expected_parent_update
        self.last_resume_audit: dict[str, Any] | None = None
        super().__init__(config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu)

    def resume(self, checkpoint: str | Path) -> dict[str, Any]:
        source = Path(checkpoint)
        import torch

        payload = torch.load(source, map_location="cpu", weights_only=False)
        observed_update = int(payload.get("exact_update", -1))
        if (
            self.expected_parent_physical_run_id is not None
            and self.expected_parent_update == observed_update
        ):
            audit = checkpoint_state_audit(
                source,
                expected_update=observed_update,
                expected_sha256=self.expected_parent_checkpoint_sha256,
            )
            if not audit["strict_state_complete"]:
                raise ValueError("parent boundary checkpoint lacks strict continuation state")
            metadata = payload.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("parent checkpoint lacks lineage metadata")
            lineage = validate_scientific_lineage_transition(
                parent_metadata=metadata,
                config=self.config,
                expected_parent_run_id=self.expected_parent_physical_run_id,
            )
            expected_metadata = {
                key: metadata[key] for key in STABLE_LINEAGE_KEYS if key in metadata
            }
            receipt = load_checkpoint(
                source,
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=None,
                expected_metadata=expected_metadata,
            )
            self.current_update = observed_update
            self.last_resume_audit = {
                **audit,
                "lineage_identity": lineage["identity"],
                "parent_physical_run_id": self.expected_parent_physical_run_id,
                "h5_physical_run_id": self.config.physical_run_id,
                "rng_state_roundtrip": bool(receipt["rng_state_roundtrip"]),
                "status": "STRICT_PARENT_TO_H5_TRANSITION_LOADED",
            }
            return receipt
        receipt = super().resume(source)
        self.last_resume_audit = {
            "parent_physical_run_id": self.config.physical_run_id,
            "h5_physical_run_id": self.config.physical_run_id,
            "observed_update": int(receipt["payload"]["exact_update"]),
            "rng_state_roundtrip": bool(receipt["rng_state_roundtrip"]),
            "status": "H5_EXACT_BOUNDARY_LOADED",
        }
        return receipt


def bootstrap_parent_attempt(
    *,
    parent_attempt_root: Path,
    h5_run_root: Path,
    config: O0RunConfig,
    boundary_update: int,
    expected_checkpoint_sha256: str,
    expected_parent_physical_run_id: str,
) -> dict[str, Any]:
    """Copy immutable parent state into a new H5 append-only run root."""

    receipt_path = h5_run_root / "CONTINUATION_BOOTSTRAP.json"
    if h5_run_root.exists():
        if not receipt_path.is_file():
            raise FileExistsError(h5_run_root)
        value = read_json(receipt_path)
        if value.get("h5_config_hash") != config.identity() or int(value.get("boundary_update", -1)) != int(boundary_update):
            raise ValueError("existing H5 bootstrap identity differs")
        return value
    h5_run_root.mkdir(parents=True)
    try:
        for directory in ("checkpoints", "representations"):
            source = parent_attempt_root / directory
            destination = h5_run_root / directory
            destination.mkdir()
            for member in source.iterdir():
                if member.is_file():
                    token = member.stem.rsplit("_", 1)[-1]
                    try:
                        update = int(token)
                    except ValueError:
                        continue
                    if update <= int(boundary_update):
                        shutil.copy2(member, destination / member.name)
        shutil.copy2(parent_attempt_root / "RESOURCE_TRACE.csv", h5_run_root / "RESOURCE_TRACE.csv")
        for source in parent_attempt_root.glob("SIDECAR_CAPTURE_RECEIPT*.json"):
            shutil.copy2(source, h5_run_root / source.name)
        parent_config = read_json(parent_attempt_root / "RESOLVED_CONFIG.json")
        atomic_write_json(h5_run_root / "PARENT_RESOLVED_CONFIG.json", parent_config)
        atomic_write_json(h5_run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
        checkpoint = h5_run_root / "checkpoints" / f"update_{int(boundary_update):06d}.pt"
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=int(boundary_update),
            expected_sha256=expected_checkpoint_sha256,
        )
        if not audit["strict_state_complete"]:
            raise ValueError("copied parent checkpoint lacks strict continuation state")
        parent_marker_candidates = (
            parent_attempt_root / f"H4_STAGE_{int(boundary_update):06d}_COMPLETE.json",
            parent_attempt_root / f"H5_STAGE_{int(boundary_update):06d}_COMPLETE.json",
            parent_attempt_root / f"RUN_PAUSED_{int(boundary_update):06d}.json",
            parent_attempt_root / "RUN_COMPLETE.json",
        )
        parent_marker_path = next((path for path in parent_marker_candidates if path.is_file()), None)
        if parent_marker_path is None:
            raise RuntimeError("parent attempt lacks an exact boundary marker")
        parent_marker = read_json(parent_marker_path)
        paused = {
            "schema_version": "metachorus.v53.m4a-h5.bootstrap-paused-run.v1",
            "created_at_utc": utc_now(),
            "status": "PAUSED_AT_CHECKPOINT",
            "config_hash": config.identity(),
            "physical_run_id": config.physical_run_id,
            "start_update": 1,
            "stop_update": int(boundary_update),
            "optimizer_steps": int(boundary_update),
            "optimizer_steps_this_invocation": 0,
            "wall_time_seconds": float(parent_marker.get("wall_time_seconds", parent_marker.get("model_total_seconds", 0.0))),
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
            "trace_sha256": sha256_file(h5_run_root / "RESOURCE_TRACE.csv"),
        }
        atomic_write_json(
            h5_run_root / f"RUN_PAUSED_{int(boundary_update):06d}.json", paused
        )
        basis = {
            "parent_physical_run_id": expected_parent_physical_run_id,
            "h5_physical_run_id": config.physical_run_id,
            "boundary_update": int(boundary_update),
            "parent_checkpoint_sha256": expected_checkpoint_sha256,
            "copied_checkpoint_sha256": sha256_file(checkpoint),
            "parent_trace_sha256": sha256_file(parent_attempt_root / "RESOURCE_TRACE.csv"),
            "copied_trace_sha256": sha256_file(h5_run_root / "RESOURCE_TRACE.csv"),
            "h5_config_hash": config.identity(),
            "strict_state_complete": True,
            "parent_immutable": True,
        }
        receipt = {
            "schema_version": "metachorus.v53.m4a-h5.continuation-bootstrap.v1",
            "status": "PASS",
            **basis,
            "identity": canonical_json_hash(basis),
        }
        atomic_write_json(receipt_path, receipt)
        return receipt
    except Exception:
        shutil.rmtree(h5_run_root, ignore_errors=True)
        raise


def validate_stage_marker(
    marker_path: Path,
    *,
    plan_row: Mapping[str, Any],
    endpoint_update: int,
    run_root: Path,
) -> dict[str, Any]:
    value = read_json(marker_path)
    expected = {
        "physical_run_id": str(plan_row["physical_run_id"]),
        "dataset": str(plan_row["dataset"]),
        "candidate_set_hash": str(plan_row["candidate_set_hash"]),
        "model_seed": int(plan_row["model_seed"]),
        "endpoint_update": int(endpoint_update),
        "status": "PASS",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    checkpoint = run_root / "checkpoints" / f"update_{int(endpoint_update):06d}.pt"
    representation = run_root / "representations" / f"update_{int(endpoint_update):06d}.npy"
    if not checkpoint.is_file() or not representation.is_file():
        mismatches["endpoint_artifacts"] = "missing"
    else:
        if sha256_file(checkpoint) != value.get("checkpoint_sha256"):
            mismatches["checkpoint_sha256"] = "mismatch"
        if sha256_file(representation) != value.get("representation_sha256"):
            mismatches["representation_sha256"] = "mismatch"
    if mismatches:
        raise ValueError(f"invalid existing H5 stage marker: {mismatches}")
    return value


def run_to_fidelity(
    *,
    plan_row: Mapping[str, Any],
    repo_root: Path,
    cache_root: Path,
    h5_result_root: Path,
    parent_attempt_root: Path,
    endpoint_update: int,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    endpoint = int(endpoint_update)
    if endpoint not in {1000, 2000, 3000}:
        raise ValueError("H5 endpoint must be 1000, 2000, or 3000")
    continuation_from = int(plan_row["continuation_from_update"])
    if (continuation_from, endpoint) not in {
        (300, 1000),
        (1000, 2000),
        (2000, 3000),
        (0, 1000),
        (0, 2000),
    }:
        raise ValueError(f"unapproved H5 continuation boundary: {continuation_from}->{endpoint}")
    run_id = str(plan_row["physical_run_id"])
    attempt_index = int(plan_row.get("attempt_index", 0))
    run_root = h5_result_root / "runs" / run_id / f"attempt_{attempt_index:02d}"
    parent_config = read_json(parent_attempt_root / "RESOLVED_CONFIG.json")
    run_role = (
        "M4A_H5_THIRD_SEED_AUDIT"
        if int(plan_row["model_seed"]) == THIRD_DEVELOPMENT_MODEL_SEED
        else "M4A_H5_SEARCH"
    )
    config = build_h5_config(
        parent_config,
        cache_root=cache_root / str(plan_row["dataset"]),
        output_root=run_root,
        physical_run_id=run_id,
        plan_seal_sha256=str(plan_row["plan_seal_sha256"]),
        model_seed=int(plan_row["model_seed"]),
        run_role=run_role,
    )
    marker = run_root / f"H5_STAGE_{endpoint:06d}_COMPLETE.json"
    if marker.is_file():
        return validate_stage_marker(
            marker, plan_row=plan_row, endpoint_update=endpoint, run_root=run_root
        )
    mode = str(plan_row["continuation_mode"])
    if continuation_from == 0:
        approved_from_zero = {
            ("FROM_ZERO_MISSING_R2_STATE", 1000),
            ("FROM_ZERO_THIRD_DEVELOPMENT_SEED", 2000),
        }
        if (mode, endpoint) not in approved_from_zero:
            raise ValueError("unapproved H5 from-zero recovery mode/endpoint")
        if run_root.exists():
            raise RuntimeError("preserve the failed from-zero attempt and allocate a new attempt index")
        session = H5ContinuationSession(
            config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu
        )
        run = session.run(stop_at_update=endpoint)
        atomic_write_json(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    else:
        if not run_root.exists():
            bootstrap_parent_attempt(
                parent_attempt_root=parent_attempt_root,
                h5_run_root=run_root,
                config=config,
                boundary_update=continuation_from,
                expected_checkpoint_sha256=str(plan_row["parent_checkpoint_sha256"]),
                expected_parent_physical_run_id=str(plan_row["parent_physical_run_id"]),
            )
        checkpoint = run_root / "checkpoints" / f"update_{continuation_from:06d}.pt"
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=continuation_from,
            expected_sha256=str(plan_row["parent_checkpoint_sha256"]),
        )
        if not audit["strict_state_complete"]:
            raise RuntimeError("H5 continuation source lacks strict state")
        session = H5ContinuationSession(
            config,
            repo_root=repo_root,
            expected_parent_physical_run_id=str(plan_row["parent_physical_run_id"]),
            expected_parent_checkpoint_sha256=str(plan_row["parent_checkpoint_sha256"]),
            expected_parent_update=continuation_from,
            enforce_target_gpu=enforce_target_gpu,
        )
        run = session.run(resume_checkpoint=checkpoint, stop_at_update=endpoint)
    endpoint_checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    endpoint_representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    if not endpoint_checkpoint.is_file() or not endpoint_representation.is_file():
        raise RuntimeError("H5 endpoint capture is incomplete")
    basis = {
        "physical_run_id": run_id,
        "dataset": str(plan_row["dataset"]),
        "candidate_set_hash": str(plan_row["candidate_set_hash"]),
        "model_seed": int(plan_row["model_seed"]),
        "continuation_from_update": continuation_from,
        "endpoint_update": endpoint,
        "optimizer_steps": int(run["optimizer_steps"]),
        "optimizer_steps_this_invocation": int(run["optimizer_steps_this_invocation"]),
        "model_total_seconds": float(run["model_total_seconds"]),
        "model_seconds_this_invocation": float(run["wall_time_seconds_this_invocation"]),
        "peak_gpu_allocated_bytes": int(run["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(run["peak_gpu_reserved_bytes"]),
        "checkpoint_sha256": sha256_file(endpoint_checkpoint),
        "representation_sha256": sha256_file(endpoint_representation),
        "continuation_mode": mode,
        "resume_audit": session.last_resume_audit,
        "labels_mounted": False,
        "status": "PASS",
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h5.stage-run-complete.v1",
        **basis,
        "identity": canonical_json_hash(basis),
    }
    atomic_write_json(marker, payload)
    return payload


def read_plan_row(plan_path: Path, physical_run_id: str) -> dict[str, str]:
    with plan_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["physical_run_id"] == physical_run_id]
    if len(rows) != 1:
        raise ValueError(f"expected one H5 plan row for {physical_run_id}, observed {len(rows)}")
    return dict(rows[0])
