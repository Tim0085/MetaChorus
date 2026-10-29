"""Strict H1->H4 continuation and exact multi-fidelity boundary handling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from metachorus_v53.audit.common import (
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
    utc_now,
)
from metachorus_v53.training.o0_checkpoint_io import (
    load_checkpoint,
    nested_state_sha256,
)
from metachorus_v53.training.o0_codefaithful_runner import (
    O0RunConfig,
    O0TrainingSession,
)
from .constants import CHECKPOINT_SCHEDULE, RESERVED_MODEL_SEED, SEARCH_MODEL_SEEDS


MODEL_CATEGORY_PREFIXES = {
    "online_encoder": ("online_encoder.",),
    "target_encoders": ("target_encoder1.", "target_encoder2."),
    "projectors": ("global_projector.", "local_projector."),
    "loss_weight_parameters": ("loss_weights", "weight"),
}
RNG_KEYS = {
    "python",
    "numpy",
    "torch_cpu",
    "torch_cuda_all",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "deterministic_algorithms",
}
LINEAGE_METADATA_KEYS = (
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


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def checkpoint_state_audit(
    checkpoint: str | Path,
    *,
    expected_update: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint)
    digest = sha256_file(path)
    payload = _load_checkpoint_payload(path)
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    rng_state = payload.get("rng_state")
    metadata = payload.get("metadata")
    model_keys = set(model_state) if isinstance(model_state, Mapping) else set()
    online = any(key.startswith("online_encoder.") for key in model_keys)
    targets = all(any(key.startswith(prefix) for key in model_keys) for prefix in ("target_encoder1.", "target_encoder2."))
    projectors = all(any(key.startswith(prefix) for key in model_keys) for prefix in ("global_projector.", "local_projector."))
    loss_weights = {"loss_weights", "weight"}.issubset(model_keys)
    optimizer = (
        isinstance(optimizer_state, Mapping)
        and isinstance(optimizer_state.get("state"), Mapping)
        and isinstance(optimizer_state.get("param_groups"), list)
        and bool(optimizer_state.get("param_groups"))
    )
    scheduler = "scheduler_state" in payload and payload.get("scheduler_state") is None
    rng = isinstance(rng_state, Mapping) and RNG_KEYS.issubset(rng_state)
    augmentation = (
        isinstance(metadata, Mapping)
        and isinstance(metadata.get("augmentation_namespace"), str)
        and bool(metadata.get("augmentation_namespace"))
        and rng
    )
    source_data_path = (
        isinstance(metadata, Mapping)
        and all(
            isinstance(metadata.get(key), str) and bool(metadata.get(key))
            for key in (
                "source_commit",
                "source_tree",
                "model_py_sha256",
                "data_cache_hash",
                "path_set_hash",
                "source_data_identity",
                "projection_identity",
                "candidate_bank_hash",
            )
        )
    )
    exact = int(payload.get("exact_update", -1)) == int(expected_update)
    schema = payload.get("schema_version") == "metachorus.v53.o0-complete-checkpoint.v1"
    label_free = (
        payload.get("contains_evaluation_information") is False
        and payload.get("contains_labels") is False
        and payload.get("contains_class_count") is False
    )
    sha_ok = expected_sha256 is None or digest == expected_sha256
    components = {
        "online_encoder_present": online,
        "target_encoders_present": targets,
        "projectors_present": projectors,
        "loss_weight_parameters_present": loss_weights,
        "optimizer_state_present": optimizer,
        "scheduler_null_explicit": scheduler,
        "python_numpy_torch_rng_present": rng,
        "augmentation_state_present": augmentation,
        "exact_update_verified": exact,
        "source_data_path_hashes_present": source_data_path,
        "label_free_checkpoint": label_free,
        "schema_supported": schema,
        "checkpoint_sha256_verified": sha_ok,
    }
    complete = all(components.values())
    return {
        **components,
        "checkpoint_sha256": digest,
        "checkpoint_size_bytes": path.stat().st_size,
        "expected_update": int(expected_update),
        "observed_update": int(payload.get("exact_update", -1)),
        "augmentation_state_mode": "GLOBAL_RNG_PLUS_FROZEN_NAMESPACE",
        "scheduler_state": "null" if payload.get("scheduler_state") is None else "non_null",
        "rng_state_sha256": str(payload.get("rng_state_sha256", "")),
        "model_state_sha256": nested_state_sha256(model_state) if isinstance(model_state, Mapping) else "",
        "optimizer_state_sha256": nested_state_sha256(optimizer_state) if isinstance(optimizer_state, Mapping) else "",
        "strict_state_complete": complete,
        "continuation_mode": "STRICT_CONTINUATION" if complete else "RESTARTED_FROM_ZERO_STATE_INCOMPLETE",
        "status": "PASS" if complete else "INCOMPLETE",
    }


def h4_physical_run_id(dataset: str, candidate_ordinal: int, model_seed: int) -> str:
    token = (
        "S00"
        if int(model_seed) == SEARCH_MODEL_SEEDS[0]
        else "S01"
        if int(model_seed) == SEARCH_MODEL_SEEDS[1]
        else "RA"
        if int(model_seed) == RESERVED_MODEL_SEED
        else None
    )
    if token is None:
        raise ValueError(f"unapproved H4 model seed: {model_seed}")
    return f"M4A-H4-{dataset.upper()}-C{int(candidate_ordinal):03d}-{token}"


def build_h4_config(
    parent_resolved_config: Mapping[str, Any],
    *,
    cache_root: Path,
    output_root: Path,
    physical_run_id: str,
    plan_seal_sha256: str,
    model_seed: int | None = None,
    reserved_audit: bool = False,
) -> O0RunConfig:
    value = dict(parent_resolved_config)
    value.update(
        {
            "cache_root": str(cache_root.resolve()),
            "output_root": str(output_root.resolve()),
            "max_updates": 3000,
            "checkpoint_schedule": list(CHECKPOINT_SCHEDULE),
            "physical_run_id": physical_run_id,
            "model_seed": int(model_seed) if model_seed is not None else int(value["model_seed"]),
            "run_role": "M4A_H4_RESERVED_AUDIT" if reserved_audit else "M4A_H4_SEARCH",
            "eligible_for_scientific_result": True,
            "eligible_for_selector_development": True,
            "plan_seal_hash": plan_seal_sha256,
        }
    )
    config = O0RunConfig.from_mapping(value)
    config.validate()
    return config


def validate_lineage_transition(
    *,
    parent_metadata: Mapping[str, Any],
    config: O0RunConfig,
    expected_parent_physical_run_id: str,
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
    if parent_metadata.get("physical_run_id") != expected_parent_physical_run_id:
        mismatches["physical_run_id"] = {
            "parent": parent_metadata.get("physical_run_id"),
            "expected": expected_parent_physical_run_id,
        }
    basis = {
        "matched_keys": sorted(expected),
        "allowed_transition_keys": ["config_hash", "physical_run_id", "plan_seal_hash"],
        "mismatches": mismatches,
        "same_candidate_seed_identity": not mismatches,
        "augmentation_state_mode": "GLOBAL_RNG_PLUS_FROZEN_NAMESPACE",
    }
    if mismatches:
        raise ValueError(f"H1->H4 lineage mismatch: {mismatches}")
    return {**basis, "identity": canonical_json_hash(basis)}


class H4ContinuationSession(O0TrainingSession):
    """O0 session that permits one audited H1 config-envelope transition at update 300."""

    def __init__(
        self,
        config: O0RunConfig,
        *,
        repo_root: str | Path,
        expected_parent_physical_run_id: str | None = None,
        expected_parent_checkpoint_sha256: str | None = None,
        enforce_target_gpu: bool = True,
    ):
        self.expected_parent_physical_run_id = expected_parent_physical_run_id
        self.expected_parent_checkpoint_sha256 = expected_parent_checkpoint_sha256
        self.last_resume_audit: dict[str, Any] | None = None
        super().__init__(config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu)

    def resume(self, checkpoint: str | Path) -> dict[str, Any]:
        source = Path(checkpoint)
        payload = _load_checkpoint_payload(source)
        if (
            int(payload.get("exact_update", -1)) == 300
            and self.expected_parent_physical_run_id is not None
        ):
            audit = checkpoint_state_audit(
                source,
                expected_update=300,
                expected_sha256=self.expected_parent_checkpoint_sha256,
            )
            if not audit["strict_state_complete"]:
                raise ValueError("parent update-300 checkpoint is not strict-continuation complete")
            lineage = validate_lineage_transition(
                parent_metadata=payload["metadata"],
                config=self.config,
                expected_parent_physical_run_id=self.expected_parent_physical_run_id,
            )
            expected_metadata = {
                key: payload["metadata"][key]
                for key in LINEAGE_METADATA_KEYS
                if key in payload["metadata"]
            }
            receipt = load_checkpoint(
                source,
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=None,
                expected_metadata=expected_metadata,
            )
            self.current_update = int(receipt["payload"]["exact_update"])
            self.last_resume_audit = {
                **audit,
                "lineage_identity": lineage["identity"],
                "parent_physical_run_id": self.expected_parent_physical_run_id,
                "h4_physical_run_id": self.config.physical_run_id,
                "rng_state_roundtrip": bool(receipt["rng_state_roundtrip"]),
                "status": "STRICT_CONTINUATION_LOADED",
            }
            return receipt
        receipt = super().resume(source)
        self.last_resume_audit = {
            "parent_physical_run_id": self.config.physical_run_id,
            "h4_physical_run_id": self.config.physical_run_id,
            "observed_update": int(receipt["payload"]["exact_update"]),
            "rng_state_roundtrip": bool(receipt["rng_state_roundtrip"]),
            "status": "H4_EXACT_BOUNDARY_LOADED",
        }
        return receipt


def bootstrap_h1_run(
    *,
    parent_attempt_root: Path,
    h4_run_root: Path,
    config: O0RunConfig,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    if h4_run_root.exists():
        receipt_path = h4_run_root / "CONTINUATION_BOOTSTRAP.json"
        if not receipt_path.is_file():
            raise FileExistsError(h4_run_root)
        receipt = read_json(receipt_path)
        if receipt.get("h4_config_hash") != config.identity():
            raise ValueError("existing bootstrap config identity differs")
        return receipt
    h4_run_root.mkdir(parents=True)
    try:
        for directory in ("checkpoints", "representations"):
            shutil.copytree(parent_attempt_root / directory, h4_run_root / directory)
        shutil.copy2(parent_attempt_root / "RESOURCE_TRACE.csv", h4_run_root / "RESOURCE_TRACE.csv")
        parent_sidecar = read_json(parent_attempt_root / "SIDECAR_CAPTURE_RECEIPT.json")
        parent_sidecar["checkpoint_schedule"] = list(CHECKPOINT_SCHEDULE)
        atomic_write_json(
            h4_run_root / "SIDECAR_CAPTURE_RECEIPT_PART_000300.json",
            parent_sidecar,
        )
        parent_run = read_json(parent_attempt_root / "RUN_COMPLETE.json")
        parent_config = read_json(parent_attempt_root / "RESOLVED_CONFIG.json")
        checkpoint_path = h4_run_root / "checkpoints" / "update_000300.pt"
        audit = checkpoint_state_audit(
            checkpoint_path,
            expected_update=300,
            expected_sha256=expected_checkpoint_sha256,
        )
        if not audit["strict_state_complete"]:
            raise ValueError("bootstrap checkpoint is incomplete")
        lineage = validate_lineage_transition(
            parent_metadata=_load_checkpoint_payload(checkpoint_path)["metadata"],
            config=config,
            expected_parent_physical_run_id=str(parent_config["physical_run_id"]),
        )
        atomic_write_json(h4_run_root / "PARENT_H1_RESOLVED_CONFIG.json", parent_config)
        atomic_write_json(h4_run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
        marker = {
            "schema_version": "metachorus.v53.m4a-h4.bootstrap-paused-run.v1",
            "created_at_utc": utc_now(),
            "status": "PAUSED_AT_CHECKPOINT",
            "config_hash": config.identity(),
            "physical_run_id": config.physical_run_id,
            "start_update": 1,
            "stop_update": 300,
            "optimizer_steps": 300,
            "optimizer_steps_this_invocation": 0,
            "wall_time_seconds": float(parent_run["wall_time_seconds"]),
            "model_total_seconds": float(parent_run["model_total_seconds"]),
            "peak_gpu_allocated_bytes": int(parent_run["peak_gpu_allocated_bytes"]),
            "peak_gpu_reserved_bytes": int(parent_run["peak_gpu_reserved_bytes"]),
            "run_role": config.run_role,
            "eligible_for_scientific_result": True,
            "eligible_for_selector_development": True,
            "labels_mounted": False,
            "downstream_evaluations": 0,
            "continuation_from_checkpoint": None,
            "continuation_same_physical_run": True,
            "bootstrap_parent_physical_run_id": parent_config["physical_run_id"],
            "trace_sha256": sha256_file(h4_run_root / "RESOURCE_TRACE.csv"),
        }
        atomic_write_json(h4_run_root / "RUN_PAUSED_000300.json", marker)
        basis = {
            "parent_physical_run_id": parent_config["physical_run_id"],
            "h4_physical_run_id": config.physical_run_id,
            "parent_attempt_terminal_status": parent_run["status"],
            "parent_checkpoint_sha256": expected_checkpoint_sha256,
            "copied_checkpoint_sha256": sha256_file(checkpoint_path),
            "parent_trace_sha256": sha256_file(parent_attempt_root / "RESOURCE_TRACE.csv"),
            "copied_trace_sha256": sha256_file(h4_run_root / "RESOURCE_TRACE.csv"),
            "h4_config_hash": config.identity(),
            "strict_state_complete": True,
            "lineage_identity": lineage["identity"],
            "parent_immutable": True,
        }
        receipt = {
            "schema_version": "metachorus.v53.m4a-h4.continuation-bootstrap.v1",
            "status": "PASS",
            **basis,
            "identity": canonical_json_hash(basis),
        }
        atomic_write_json(h4_run_root / "CONTINUATION_BOOTSTRAP.json", receipt)
        return receipt
    except Exception:
        shutil.rmtree(h4_run_root, ignore_errors=True)
        raise


def validate_stage_completion_marker(
    marker_path: Path,
    *,
    plan_row: Mapping[str, Any],
    endpoint_update: int,
    run_root: Path,
) -> dict[str, Any]:
    """Accept an existing completion marker only after full identity/hash binding."""

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
        raise ValueError(f"invalid existing H4 stage marker {marker_path}: {mismatches}")
    return value


def run_to_fidelity(
    *,
    plan_row: Mapping[str, Any],
    repo_root: Path,
    h1_result_root: Path,
    cache_root: Path,
    h4_result_root: Path,
    endpoint_update: int,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    endpoint = int(endpoint_update)
    if endpoint not in {1000, 2000, 3000}:
        raise ValueError("H4 endpoint must be 1000, 2000, or 3000")
    run_id = str(plan_row["physical_run_id"])
    run_root = h4_result_root / "runs" / run_id / "attempt_00"
    parent_attempt = h1_result_root / str(plan_row["parent_attempt_relative_path"])
    parent_config = read_json(parent_attempt / "RESOLVED_CONFIG.json")
    config = build_h4_config(
        parent_config,
        cache_root=cache_root / str(plan_row["dataset"]),
        output_root=run_root,
        physical_run_id=run_id,
        plan_seal_sha256=str(plan_row["plan_seal_sha256"]),
        model_seed=int(plan_row["model_seed"]),
        reserved_audit=bool(str(plan_row.get("run_role", "")) == "RESERVED_AUDIT"),
    )
    mode = str(plan_row["continuation_mode"])
    if endpoint == 1000 and not run_root.exists():
        if mode == "STRICT_CONTINUATION":
            bootstrap_h1_run(
                parent_attempt_root=parent_attempt,
                h4_run_root=run_root,
                config=config,
                expected_checkpoint_sha256=str(plan_row["parent_checkpoint_sha256"]),
            )
        elif mode not in {"RESTARTED_FROM_ZERO_STATE_INCOMPLETE", "FROM_ZERO_RESERVED_AUDIT"}:
            raise ValueError(f"unknown continuation mode: {mode}")
    stage_marker = run_root / f"H4_STAGE_{endpoint:06d}_COMPLETE.json"
    if stage_marker.is_file():
        return validate_stage_completion_marker(
            stage_marker,
            plan_row=plan_row,
            endpoint_update=endpoint,
            run_root=run_root,
        )
    if mode in {"RESTARTED_FROM_ZERO_STATE_INCOMPLETE", "FROM_ZERO_RESERVED_AUDIT"} and run_root.exists():
        raise RuntimeError(
            "an incomplete from-zero attempt already exists; preserve it and create an audited retry instead of resuming"
        )
    if mode in {"RESTARTED_FROM_ZERO_STATE_INCOMPLETE", "FROM_ZERO_RESERVED_AUDIT"} and not run_root.exists():
        session = H4ContinuationSession(
            config,
            repo_root=repo_root,
            enforce_target_gpu=enforce_target_gpu,
        )
        run = session.run(stop_at_update=endpoint)
        atomic_write_json(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    else:
        boundary = 300 if endpoint == 1000 else 1000 if endpoint == 2000 else 2000
        if boundary in {1000, 2000}:
            predecessor = run_root / f"H4_STAGE_{boundary:06d}_COMPLETE.json"
            if not predecessor.is_file():
                raise RuntimeError(f"missing predecessor completion marker: {predecessor.name}")
            validate_stage_completion_marker(
                predecessor,
                plan_row=plan_row,
                endpoint_update=boundary,
                run_root=run_root,
            )
        checkpoint = run_root / "checkpoints" / f"update_{boundary:06d}.pt"
        boundary_audit = checkpoint_state_audit(checkpoint, expected_update=boundary)
        if not boundary_audit["strict_state_complete"]:
            raise RuntimeError(f"incomplete strict continuation state at update {boundary}")
        session = H4ContinuationSession(
            config,
            repo_root=repo_root,
            expected_parent_physical_run_id=(
                str(plan_row["parent_physical_run_id"]) if boundary == 300 else None
            ),
            expected_parent_checkpoint_sha256=(
                str(plan_row["parent_checkpoint_sha256"]) if boundary == 300 else None
            ),
            enforce_target_gpu=enforce_target_gpu,
        )
        run = session.run(resume_checkpoint=checkpoint, stop_at_update=endpoint)
    endpoint_checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    endpoint_representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    if not endpoint_checkpoint.is_file() or not endpoint_representation.is_file():
        raise RuntimeError("H4 endpoint capture is incomplete")
    basis = {
        "physical_run_id": run_id,
        "dataset": plan_row["dataset"],
        "candidate_set_hash": plan_row["candidate_set_hash"],
        "model_seed": int(plan_row["model_seed"]),
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
        "schema_version": "metachorus.v53.m4a-h4.stage-run-complete.v1",
        **basis,
        "identity": canonical_json_hash(basis),
    }
    atomic_write_json(stage_marker, payload)
    return payload


def read_plan_row(plan_path: Path, physical_run_id: str) -> dict[str, str]:
    with plan_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["physical_run_id"] == physical_run_id]
    if len(rows) != 1:
        raise ValueError(f"expected one plan row for {physical_run_id}, observed {len(rows)}")
    return dict(rows[0])
