"""O0-CodeFaithful execution envelope for M6F search and M7F preliminary runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file
from metachorus_v53.m4a_h4.continuation import checkpoint_state_audit
from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession

from .common import envelope, write_json_once
from .constants import (
    CANARY_ENDPOINT,
    EVAL_MODEL_SEEDS,
    PRELIM_ENDPOINT,
    SEARCH_ENDPOINTS,
    SEARCH_MODEL_SEEDS,
)
from .dataset_specs import DATASET_SPECS
from .planning import canary_run_id, physical_run_id


STAGES = {"CANARY", "SEARCH", "PRELIMINARY"}


def _canary_engineering_audit(
    *,
    session: O0TrainingSession,
    config: O0RunConfig,
    run_root: Path,
    repo_root: str | Path,
    enforce_target_gpu: bool,
) -> dict[str, Any]:
    endpoint = CANARY_ENDPOINT
    checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    sidecar_path = run_root / "SIDECAR_CAPTURE_RECEIPT.json"
    sidecar = read_json(sidecar_path)
    if (
        sidecar.get("schema_version") != "metachorus.v53.o0-sidecar-capture.v1"
        or list(sidecar.get("checkpoint_schedule", [])) != [endpoint]
        or sidecar.get("labels_read") is not False
        or sidecar.get("downstream_evaluation_executed") is not False
        or len(sidecar.get("checkpoints", [])) != 1
        or len(sidecar.get("representations", [])) != 1
        or sidecar["checkpoints"][0].get("sha256") != sha256_file(checkpoint)
        or sidecar["representations"][0].get("sha256") != sha256_file(representation)
    ):
        raise RuntimeError("fast canary write/capture audit failed")
    strict = checkpoint_state_audit(
        checkpoint,
        expected_update=endpoint,
        expected_sha256=sha256_file(checkpoint),
    )
    if strict.get("strict_state_complete") is not True:
        raise RuntimeError("fast canary checkpoint is not strict-state complete")
    trained = session.state_snapshot()
    restored_session = O0TrainingSession(
        config,
        repo_root=repo_root,
        enforce_target_gpu=enforce_target_gpu,
    )
    loaded = restored_session.resume(checkpoint)
    restored = restored_session.state_snapshot()
    state_fields = ("model_state_sha256", "optimizer_state_sha256", "target_state_sha256")
    optimizer = loaded.get("payload", {}).get("optimizer_state", {})
    state_rows = optimizer.get("state", {}) if isinstance(optimizer, Mapping) else {}
    step_values: list[int] = []
    if isinstance(state_rows, Mapping):
        for state in state_rows.values():
            if not isinstance(state, Mapping) or "step" not in state:
                raise RuntimeError("fast canary optimizer state lacks a step counter")
            value = state["step"]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise RuntimeError("fast canary optimizer step counter is not scalar")
                value = value.detach().cpu().item()
            step_values.append(int(value))
    if (
        int(loaded.get("payload", {}).get("exact_update", -1)) != endpoint
        or loaded.get("rng_state_roundtrip") is not True
        or restored_session.current_update != endpoint
        or any(trained[field] != restored[field] for field in state_fields)
        or trained["state_dict_keys"] != restored["state_dict_keys"]
        or not step_values
        or set(step_values) != {endpoint}
    ):
        raise RuntimeError("fast canary fresh-session checkpoint restore failed")
    basis = {
        "checkpoint_write_pass": True,
        "checkpoint_read_pass": True,
        "checkpoint_restored_in_fresh_session": True,
        "checkpoint_strict_state_complete": True,
        "checkpoint_rng_roundtrip": True,
        "checkpoint_state_roundtrip": True,
        "path_resolution_pass": True,
        "optimizer_step_pass": True,
        "optimizer_steps": endpoint,
        "optimizer_state_entry_count": len(step_values),
        "optimizer_step_values": sorted(set(step_values)),
        "sidecar_receipt_sha256": sha256_file(sidecar_path),
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
    }
    return {**basis, "identity": canonical_json_hash(basis), "status": "PASS"}


def _cache_contract(cache_root: Path) -> dict[str, Any]:
    complete = read_json(cache_root / "SANITIZED_CACHE_COMPLETE.json")
    manifest = read_json(cache_root / "SANITIZED_CACHE_MANIFEST.json")
    projection = read_json(cache_root / "metadata" / "projection_identity.json")
    all_sets = read_json(cache_root / "candidates" / "all_sets.json")
    if (
        complete.get("status") != "COMPLETE"
        or complete.get("cache_identity") != manifest.get("cache_identity")
        or projection.get("path_library_identity") != manifest["identity_basis"].get("path_library_identity")
        or all_sets.get("all_sets_identity") != manifest["identity_basis"].get("all_sets_identity")
    ):
        raise ValueError("fast cache contract is not sealed")
    return {
        "cache_identity": str(complete["cache_identity"]),
        "source_data_identity": str(manifest["identity_basis"]["structural_data_identity"]),
        "projection_identity": canonical_json_hash(projection),
        "candidate_bank_hash": str(all_sets["all_sets_identity"]),
    }


def _validate_plan_row(row: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported fast stage: {stage}")
    value = dict(row)
    dataset = str(value.get("dataset", ""))
    if dataset not in DATASET_SPECS:
        raise ValueError("fast plan row has an invalid dataset")
    path_ids = value.get("path_ids")
    if isinstance(path_ids, str):
        path_ids = json.loads(path_ids)
    if not isinstance(path_ids, list) or len(path_ids) != DATASET_SPECS[dataset].view_budget:
        raise ValueError("fast plan row violates fixed set cardinality")
    if (
        value.get("labels_mounted") is not False
        or int(value.get("downstream_evaluations_during_training", -1)) != 0
        or value.get("status") not in {"PENDING_GPU", "AUTHORIZED", "PAUSED_AT_CHECKPOINT"}
    ):
        raise ValueError("fast training cannot mount labels or run downstream evaluation")
    if int(value.get("attempt_index", 0)) not in {0, 1}:
        raise ValueError("fast run allows at most one engineering retry")
    start = int(value.get("continuation_from_update", 0))
    endpoint = int(value.get("endpoint_update", -1))
    if stage == "CANARY":
        valid_boundary = start == 0 and endpoint == CANARY_ENDPOINT and value.get("stage") == "CANARY"
    elif stage == "SEARCH":
        expected_start = {
            SEARCH_ENDPOINTS[0]: 0,
            **{
                SEARCH_ENDPOINTS[index]: SEARCH_ENDPOINTS[index - 1]
                for index in range(1, len(SEARCH_ENDPOINTS))
            },
        }
        valid_boundary = (
            endpoint in expected_start
            and start == expected_start[endpoint]
            and value.get("stage") in {"SEARCH", "R1_ALL_SETS_0000_TO_0300"}
            and (value.get("stage") != "R1_ALL_SETS_0000_TO_0300" or endpoint == SEARCH_ENDPOINTS[0])
        )
    else:
        valid_boundary = start == 0 and endpoint == PRELIM_ENDPOINT and value.get("stage") == "PRELIMINARY"
    if not valid_boundary:
        raise ValueError("fast plan row violates the frozen stage boundary")
    value["path_ids"] = list(map(str, path_ids))
    return value


def validate_run_marker(
    marker: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    *,
    stage: str,
    endpoint: int,
    expected_attempt_index: int | None = None,
) -> dict[str, Any]:
    """Bind an append-only completion marker to one exact frozen plan row."""

    row = _validate_plan_row(plan_row, stage)
    basis = marker.get("basis", {})
    expected_start = int(row.get("continuation_from_update", 0))
    attempt = int(basis.get("attempt_index", -1))
    if (
        marker.get("status") != "PASS"
        or marker.get("identity") != canonical_json_hash(basis)
        or basis.get("stage") != stage
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or basis.get("dataset") != row.get("dataset")
        or basis.get("candidate_set_id") != row.get("candidate_set_id")
        or basis.get("candidate_set_hash") != row.get("candidate_set_hash")
        or int(basis.get("candidate_set_ordinal", -1)) != int(row.get("candidate_set_ordinal", -2))
        or list(basis.get("path_ids", [])) != list(row.get("path_ids", []))
        or int(basis.get("model_seed", -1)) != int(row.get("model_seed", -2))
        or int(basis.get("continuation_from_update", -1)) != expected_start
        or int(row.get("endpoint_update", -1)) != int(endpoint)
        or int(basis.get("endpoint_update", -1)) != int(endpoint)
        or int(basis.get("optimizer_steps", -1)) != int(endpoint)
        or attempt not in {0, 1}
        or (expected_attempt_index is not None and attempt != int(expected_attempt_index))
        or basis.get("labels_mounted") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
    ):
        raise ValueError("fast completion marker does not match the frozen plan identity")
    return dict(basis)


def build_run_config(
    *,
    plan_row: Mapping[str, Any],
    stage: str,
    cache_root: str | Path,
    output_root: str | Path,
    source_commit: str,
    source_tree: str,
) -> O0RunConfig:
    row = _validate_plan_row(plan_row, stage)
    dataset = str(row["dataset"])
    cache = _cache_contract(Path(cache_root).resolve())
    candidate_ordinal = int(row["candidate_set_ordinal"])
    model_seed = int(row["model_seed"])
    governance_plan_seal_identity = str(row.get("plan_seal_identity", row.get("plan_seal_hash", "")))
    plan_seal_identity = str(row.get("run_config_plan_seal_identity", governance_plan_seal_identity))
    start = int(row.get("continuation_from_update", 0))
    if start == 0 and any(key in row for key in ("run_config_plan_seal_identity", "run_config_identity")):
        raise ValueError("fresh fast run cannot override its training config seal identity")
    if start and (
        "run_config_plan_seal_identity" not in row
        or len(str(row.get("run_config_identity", ""))) != 64
    ):
        raise ValueError("fast continuation requires its sealed training config identity")
    if len(plan_seal_identity) != 64:
        raise ValueError("fast run requires a sealed plan identity")
    if stage == "CANARY":
        run_role = "M6F_FAST_ENGINEERING_CANARY"
        max_updates = CANARY_ENDPOINT
        schedule = (CANARY_ENDPOINT,)
        seeds = SEARCH_MODEL_SEEDS
        seed_ordinal = seeds.index(model_seed)
        expected_id = canary_run_id(dataset, candidate_ordinal, seed_ordinal)
        scientific = False
        selector = False
    elif stage == "SEARCH":
        run_role = "M6F_FAST_SEARCH"
        max_updates = SEARCH_ENDPOINTS[-1]
        schedule = SEARCH_ENDPOINTS
        seeds = SEARCH_MODEL_SEEDS
        seed_ordinal = seeds.index(model_seed)
        expected_id = physical_run_id(dataset, candidate_ordinal, seed_ordinal)
        scientific = True
        selector = True
    else:
        run_role = "M7F_FAST_PRELIMINARY"
        max_updates = PRELIM_ENDPOINT
        schedule = (1000, 2000, 3000)
        seeds = EVAL_MODEL_SEEDS
        seed_ordinal = seeds.index(model_seed)
        expected_id = physical_run_id(dataset, candidate_ordinal, seed_ordinal, prefix="M7F")
        scientific = True
        selector = False
    if str(row["physical_run_id"]) != expected_id:
        raise ValueError("fast physical run identity differs from the frozen matrix")
    if str(row.get("cache_identity", cache["cache_identity"])) != cache["cache_identity"]:
        raise ValueError("fast plan/cache identity mismatch")
    config = O0RunConfig(
        dataset=dataset,
        path_set_role="Candidate",
        path_set_id=str(row["candidate_set_id"]),
        path_ids=tuple(row["path_ids"]),
        candidate_set_hash=str(row["candidate_set_hash"]),
        model_seed=model_seed,
        max_updates=max_updates,
        capture_mode="on",
        checkpoint_schedule=tuple(schedule),
        cache_root=str(Path(cache_root).resolve()),
        output_root=str(Path(output_root).resolve()),
        device="cuda:0",
        source_commit=str(source_commit),
        source_tree=str(source_tree),
        cache_identity=cache["cache_identity"],
        augmentation_namespace=f"m6f-m7f-fast/{stage.casefold()}/{dataset}/{expected_id}",
        run_role=run_role,
        eligible_for_scientific_result=scientific,
        eligible_for_selector_development=selector,
        physical_run_id=expected_id,
        candidate_set_ordinal=candidate_ordinal,
        source_data_identity=cache["source_data_identity"],
        projection_identity=cache["projection_identity"],
        candidate_bank_hash=cache["candidate_bank_hash"],
        plan_seal_hash=plan_seal_identity,
    )
    if start and config.identity() != str(row["run_config_identity"]):
        raise ValueError("fast continuation training config identity mismatch")
    return config


def run_plan_row(
    *,
    plan_row: Mapping[str, Any],
    stage: str,
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    row = _validate_plan_row(plan_row, stage)
    result = Path(result_root).resolve()
    attempt_index = int(row.get("attempt_index", 0))
    branch = {"CANARY": "canary_runs", "SEARCH": "search_runs", "PRELIMINARY": "preliminary_runs"}[stage]
    run_root = result / branch / str(row["physical_run_id"]) / f"attempt_{attempt_index:02d}"
    endpoint = int(row["endpoint_update"])
    marker = run_root / f"M6F_M7F_{stage}_{endpoint:06d}_COMPLETE.json"
    if marker.is_file():
        value = read_json(marker)
        validate_run_marker(
            value,
            row,
            stage=stage,
            endpoint=endpoint,
            expected_attempt_index=attempt_index,
        )
        return value
    start = int(row.get("continuation_from_update", 0))
    if start:
        if stage != "SEARCH" or not run_root.is_dir() or run_root.is_symlink():
            raise ValueError("only SEARCH may continue inside an existing regular attempt root")
        resume_path = Path(str(row["resume_checkpoint_path"])).resolve(strict=True)
        expected_resume = run_root / "checkpoints" / f"update_{start:06d}.pt"
        if resume_path != expected_resume.resolve(strict=True):
            raise ValueError("fast continuation checkpoint is outside the same physical attempt")
        if sha256_file(resume_path) != str(row["resume_checkpoint_sha256"]):
            raise ValueError("fast continuation checkpoint hash mismatch")
        prior_marker = run_root / f"RUN_PAUSED_{start:06d}.json"
        prior_sidecar = run_root / f"SIDECAR_CAPTURE_RECEIPT_PART_{start:06d}.json"
        prior_boundary = run_root / f"M6F_M7F_SEARCH_{start:06d}_COMPLETE.json"
        if not all(path.is_file() and not path.is_symlink() for path in (prior_marker, prior_sidecar, prior_boundary)):
            raise RuntimeError("fast continuation boundary is incomplete")
        prior = read_json(prior_boundary)
        if (
            prior.get("status") != "PASS"
            or int(prior.get("basis", {}).get("endpoint_update", -1)) != start
            or prior.get("basis", {}).get("checkpoint_sha256") != str(row["resume_checkpoint_sha256"])
        ):
            raise ValueError("fast continuation boundary identity mismatch")
    else:
        if run_root.exists():
            raise FileExistsError(f"fast run attempt is not fresh: {run_root}")
        run_root.parent.mkdir(parents=True, exist_ok=True)
    config = build_run_config(
        plan_row=row,
        stage=stage,
        cache_root=cache_root,
        output_root=run_root,
        source_commit=source_commit,
        source_tree=source_tree,
    )
    session = O0TrainingSession(config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu)
    if stage == "CANARY":
        run = session.run(stop_at_update=CANARY_ENDPOINT)
    elif stage == "SEARCH":
        if start:
            run = session.run(resume_checkpoint=resume_path, stop_at_update=endpoint)
        else:
            run = session.run(stop_at_update=endpoint)
    else:
        run = session.run(stop_at_update=PRELIM_ENDPOINT)
    if (
        run.get("labels_mounted") is not False
        or int(run.get("downstream_evaluations", -1)) != 0
        or int(run.get("optimizer_steps", -1)) != int(row["endpoint_update"])
    ):
        raise RuntimeError("fast O0 result violated labels/evaluation/update boundaries")
    write_json_once(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    if not checkpoint.is_file() or not representation.is_file():
        raise RuntimeError("fast O0 endpoint capture is incomplete")
    basis = {
        "stage": stage,
        "physical_run_id": row["physical_run_id"],
        "dataset": row["dataset"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "path_ids": list(row["path_ids"]),
        "model_seed": int(row["model_seed"]),
        "attempt_index": attempt_index,
        "continuation_from_update": int(row.get("continuation_from_update", 0)),
        "endpoint_update": endpoint,
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
        "search_checkpoint_reused": stage != "PRELIMINARY" and int(row.get("continuation_from_update", 0)) > 0,
        "preliminary_started_from_zero": stage != "PRELIMINARY" or int(row.get("continuation_from_update", 0)) == 0,
        "run_identity": config.identity(),
    }
    if stage == "CANARY":
        basis["canary_engineering_audit"] = _canary_engineering_audit(
            session=session,
            config=config,
            run_root=run_root,
            repo_root=repo_root,
            enforce_target_gpu=enforce_target_gpu,
        )
    marker = envelope(
        "metachorus.v53.m6f-m7f.fast-run-complete.v1",
        "PASS",
        basis,
    )
    write_json_once(run_root / f"M6F_M7F_{stage}_{endpoint:06d}_COMPLETE.json", marker)
    return marker


__all__ = ["STAGES", "build_run_config", "run_plan_row", "validate_run_marker"]
