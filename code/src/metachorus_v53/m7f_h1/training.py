"""O0-CodeFaithful execution envelope for M7F-H1 GPU runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file
from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession

from .common import envelope, write_json_once
from .planning import validate_plan


STAGE_ROLE = {
    "IMDB_TRAJECTORY": "M7F_H1_IMDB_TRAJECTORY",
    "MANUAL_BASELINE": "M7F_H1_MANUAL_BASELINE",
    "BUDGET_CANARY": "M7F_H1_BUDGET_CANARY",
}


def _cache_contract(cache_root: Path) -> dict[str, str]:
    complete = read_json(cache_root / "SANITIZED_CACHE_COMPLETE.json")
    manifest = read_json(cache_root / "SANITIZED_CACHE_MANIFEST.json")
    projection = read_json(cache_root / "metadata" / "projection_identity.json")
    all_sets = read_json(cache_root / "candidates" / "all_sets.json")
    if (
        complete.get("status") != "COMPLETE"
        or complete.get("cache_identity") != manifest.get("cache_identity")
        or projection.get("path_library_identity")
        != manifest.get("identity_basis", {}).get("path_library_identity")
        or all_sets.get("all_sets_identity")
        != manifest.get("identity_basis", {}).get("all_sets_identity")
    ):
        raise ValueError("M7F-H1 cache contract is not sealed")
    return {
        "dataset": str(complete["dataset"]),
        "cache_identity": str(complete["cache_identity"]),
        "source_data_identity": str(
            manifest["identity_basis"]["structural_data_identity"]
        ),
        "projection_identity": canonical_json_hash(projection),
        "candidate_bank_hash": str(all_sets["all_sets_identity"]),
    }


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    for key in ("path_ids", "checkpoint_schedule", "representation_schedule"):
        if isinstance(value.get(key), str):
            value[key] = json.loads(value[key])
        value[key] = list(value[key])
    for key in (
        "candidate_set_ordinal",
        "model_seed",
        "model_seed_ordinal",
        "continuation_from_update",
        "endpoint_update",
        "downstream_evaluations_during_training",
        "attempt_index",
    ):
        value[key] = int(value[key])
    for key in (
        "training_from_zero",
        "labels_mounted",
        "eligible_for_scientific_result",
        "eligible_for_selector_development",
        "budget_canary_not_scientific_selection",
    ):
        raw = value[key]
        value[key] = raw.casefold() == "true" if isinstance(raw, str) else bool(raw)
    stage = str(value.get("stage", ""))
    if (
        stage not in STAGE_ROLE
        or value["continuation_from_update"] != 0
        or value["training_from_zero"] is not True
        or value["labels_mounted"] is not False
        or value["downstream_evaluations_during_training"] != 0
        or value.get("status") not in {"PENDING_GPU", "AUTHORIZED"}
        or value["attempt_index"] not in {0, 1}
        or len(str(value.get("plan_seal_identity", ""))) != 64
    ):
        raise ValueError("M7F-H1 training row violates the frozen boundary")
    if stage == "BUDGET_CANARY" and (
        value["eligible_for_scientific_result"]
        or value["eligible_for_selector_development"]
        or value["budget_canary_not_scientific_selection"] is not True
    ):
        raise ValueError("budget canary cannot enter scientific selection")
    return value


def build_run_config(
    *,
    plan_row: Mapping[str, Any],
    cache_root: str | Path,
    output_root: str | Path,
    source_commit: str,
    source_tree: str,
) -> O0RunConfig:
    row = _normalize_row(plan_row)
    cache = _cache_contract(Path(cache_root).resolve())
    if (
        cache["dataset"] != row["dataset"]
        or cache["cache_identity"] != row["cache_identity"]
        or cache["candidate_bank_hash"] != row["all_sets_identity"]
    ):
        raise ValueError("M7F-H1 plan/cache identity mismatch")
    config = O0RunConfig(
        dataset=str(row["dataset"]),
        path_set_role=str(row["path_set_role"]),
        path_set_id=str(row["candidate_set_id"]),
        path_ids=tuple(map(str, row["path_ids"])),
        candidate_set_hash=str(row["candidate_set_hash"]),
        model_seed=int(row["model_seed"]),
        max_updates=int(row["endpoint_update"]),
        capture_mode="on",
        checkpoint_schedule=tuple(map(int, row["checkpoint_schedule"])),
        representation_schedule=tuple(
            map(int, row["representation_schedule"])
        ),
        cache_root=str(Path(cache_root).resolve()),
        output_root=str(Path(output_root).resolve()),
        device="cuda:0",
        source_commit=str(source_commit),
        source_tree=str(source_tree),
        cache_identity=cache["cache_identity"],
        augmentation_namespace=(
            f"m7f-h1/{str(row['stage']).casefold()}/"
            f"{row['dataset']}/{row['physical_run_id']}"
        ),
        run_role=STAGE_ROLE[str(row["stage"])],
        eligible_for_scientific_result=bool(
            row["eligible_for_scientific_result"]
        ),
        eligible_for_selector_development=bool(
            row["eligible_for_selector_development"]
        ),
        physical_run_id=str(row["physical_run_id"]),
        candidate_set_ordinal=int(row["candidate_set_ordinal"]),
        source_data_identity=cache["source_data_identity"],
        projection_identity=cache["projection_identity"],
        candidate_bank_hash=cache["candidate_bank_hash"],
        plan_seal_hash=str(row["plan_seal_identity"]),
    )
    config.validate()
    return config


def _capture_rows(
    result_root: Path, run_root: Path, sidecar: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    for role, rows, branch, output in (
        ("checkpoint", sidecar.get("checkpoints", []), "checkpoints", checkpoint_rows),
        (
            "representation",
            sidecar.get("representations", []),
            "representations",
            representation_rows,
        ),
    ):
        for row in rows:
            update = int(row["update"])
            suffix = ".pt" if role == "checkpoint" else ".npy"
            source = run_root / branch / f"update_{update:06d}{suffix}"
            cursor = result_root
            for part in source.relative_to(result_root).parts:
                cursor /= part
                if cursor.is_symlink():
                    raise RuntimeError(f"M7F-H1 {role} capture uses a symlink")
            if (
                not source.is_file()
                or sha256_file(source) != row.get("sha256")
            ):
                raise RuntimeError(f"M7F-H1 {role} capture changed")
            if role == "representation":
                values = np.load(source, allow_pickle=False, mmap_mode="r")
                if values.ndim != 2 or not np.isfinite(values).all():
                    raise RuntimeError("M7F-H1 representation is not finite 2D")
            output.append(
                {
                    "update": update,
                    "relative_path": source.relative_to(result_root).as_posix(),
                    "sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                }
            )
    return checkpoint_rows, representation_rows


def validate_run_marker(
    marker: Mapping[str, Any], plan_row: Mapping[str, Any]
) -> dict[str, Any]:
    row = _normalize_row(plan_row)
    basis = marker.get("basis", {})
    if (
        marker.get("status") != "PASS"
        or marker.get("identity") != canonical_json_hash(basis)
        or basis.get("stage") != row["stage"]
        or basis.get("physical_run_id") != row["physical_run_id"]
        or basis.get("dataset") != row["dataset"]
        or basis.get("candidate_set_hash") != row["candidate_set_hash"]
        or int(basis.get("candidate_set_ordinal", -1))
        != row["candidate_set_ordinal"]
        or int(basis.get("model_seed", -1)) != row["model_seed"]
        or int(basis.get("optimizer_steps", -1)) != row["endpoint_update"]
        or int(basis.get("attempt_index", -1)) != row["attempt_index"]
        or basis.get("labels_mounted") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
    ):
        raise ValueError("M7F-H1 completion marker differs from the plan")
    return dict(basis)


def run_plan_row(
    *,
    plan_row: Mapping[str, Any],
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    row = _normalize_row(plan_row)
    result = Path(result_root).resolve()
    branch = {
        "IMDB_TRAJECTORY": "imdb_trajectory_runs",
        "MANUAL_BASELINE": "manual_runs",
        "BUDGET_CANARY": "budget_canary_runs",
    }[row["stage"]]
    run_root = (
        result
        / branch
        / str(row["physical_run_id"])
        / f"attempt_{row['attempt_index']:02d}"
    )
    marker_path = run_root / f"M7F_H1_{row['stage']}_COMPLETE.json"
    if marker_path.is_symlink():
        raise ValueError("M7F-H1 completion marker symlinks are forbidden")
    if marker_path.exists():
        if not marker_path.is_file():
            raise ValueError("M7F-H1 completion marker is not a regular file")
        marker = read_json(marker_path)
        validate_run_marker(marker, row)
        return marker
    if run_root.is_symlink():
        raise ValueError("M7F-H1 run-root symlinks are forbidden")
    if run_root.exists():
        raise FileExistsError(f"M7F-H1 run attempt is not fresh: {run_root}")
    cursor = result
    for part in run_root.relative_to(result).parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("M7F-H1 run-root symlink chains are forbidden")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    config = build_run_config(
        plan_row=row,
        cache_root=cache_root,
        output_root=run_root,
        source_commit=source_commit,
        source_tree=source_tree,
    )
    session = O0TrainingSession(
        config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu
    )
    run = session.run(stop_at_update=int(row["endpoint_update"]))
    if (
        run.get("status") != "PASS"
        or run.get("labels_mounted") is not False
        or int(run.get("downstream_evaluations", -1)) != 0
        or int(run.get("optimizer_steps", -1)) != int(row["endpoint_update"])
    ):
        raise RuntimeError("M7F-H1 O0 result violates the training boundary")
    write_json_once(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    sidecar = run.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise RuntimeError("M7F-H1 run lacks the sidecar receipt")
    if (
        list(sidecar.get("checkpoint_schedule", []))
        != list(row["checkpoint_schedule"])
        or list(sidecar.get("representation_schedule", []))
        != list(row["representation_schedule"])
    ):
        raise RuntimeError("M7F-H1 capture schedules changed at runtime")
    checkpoints, representations = _capture_rows(result, run_root, sidecar)
    checkpoint_overhead_seconds = sum(
        float(row.get("save_wall_time_seconds") or 0.0)
        + float(row.get("hash_wall_time_seconds") or 0.0)
        for row in sidecar.get("checkpoint_io", [])
    )
    run_output_bytes = 0
    for artifact in run_root.rglob("*"):
        if artifact.is_symlink():
            raise RuntimeError("M7F-H1 run output contains a symlink")
        if artifact.is_file():
            run_output_bytes += artifact.stat().st_size
    if run_output_bytes <= 0:
        raise RuntimeError("M7F-H1 run output byte count is not positive")
    basis = {
        "stage": row["stage"],
        "physical_run_id": row["physical_run_id"],
        "dataset": row["dataset"],
        "path_set_role": row["path_set_role"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": row["candidate_set_ordinal"],
        "path_ids": list(row["path_ids"]),
        "model_seed": row["model_seed"],
        "model_seed_ordinal": row["model_seed_ordinal"],
        "attempt_index": row["attempt_index"],
        "optimizer_steps": int(run["optimizer_steps"]),
        "checkpoint_schedule": list(row["checkpoint_schedule"]),
        "representation_schedule": list(row["representation_schedule"]),
        "checkpoints": checkpoints,
        "representations": representations,
        "model_total_seconds": float(run.get("model_total_seconds", 0.0)),
        "wall_time_seconds": float(run.get("wall_time_seconds", 0.0)),
        "checkpoint_overhead_seconds": checkpoint_overhead_seconds,
        "run_output_bytes": run_output_bytes,
        "peak_gpu_allocated_bytes": int(
            run.get("peak_gpu_allocated_bytes", 0)
        ),
        "peak_gpu_reserved_bytes": int(run.get("peak_gpu_reserved_bytes", 0)),
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "budget_canary_not_scientific_selection": row[
            "budget_canary_not_scientific_selection"
        ],
        "run_identity": config.identity(),
        "plan_seal_identity": row["plan_seal_identity"],
    }
    marker = envelope(
        "metachorus.v53.m7f-h1.run-complete.v1", "PASS", basis
    )
    write_json_once(marker_path, marker)
    return marker


def run_plan_shard(
    *,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    stage: str,
    shard_index: int,
    shard_count: int,
    repo_root: str | Path,
    cache_parent: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    if shard_count != 2 or shard_index not in {0, 1}:
        raise ValueError("M7F-H1 requires exactly two run-level GPU shards")
    plan = validate_plan(rows, seal, stage=stage)
    selected = [row for index, row in enumerate(plan) if index % 2 == shard_index]
    completed: list[dict[str, Any]] = []
    failures: list[str] = []
    governed_result = Path(result_root).resolve()
    for row in selected:
        try:
            completed.append(
                run_plan_row(
                    plan_row=row,
                    repo_root=repo_root,
                    cache_root=Path(cache_parent) / row["dataset"],
                    result_root=result_root,
                    source_commit=source_commit,
                    source_tree=source_tree,
                    enforce_target_gpu=enforce_target_gpu,
                )
            )
        except Exception as error:
            signature = {
                "failure_kind": type(error).__name__,
                "failure_message_sha256": canonical_json_hash(str(error)),
            }
            failure = envelope(
                "metachorus.v53.m7f-h1.failed-attempt.v1",
                "FAILED_PRESERVED",
                {
                    "stage": stage,
                    "physical_run_id": row["physical_run_id"],
                    "dataset": row["dataset"],
                    "attempt_index": row["attempt_index"],
                    **signature,
                    "root_cause_identity": canonical_json_hash(signature),
                    "labels_mounted": False,
                    "scientific_metrics_observed": False,
                    "preserved_for_audit": True,
                },
            )
            failure_path = (
                governed_result
                / "failed_attempts"
                / str(row["dataset"])
                / stage.casefold()
                / f"{row['physical_run_id']}__attempt_{row['attempt_index']:02d}.json"
            )
            write_json_once(failure_path, failure)
            failures.append(str(row["physical_run_id"]))
    if failures:
        raise RuntimeError(
            f"M7F-H1 shard preserved failed identities: {sorted(failures)!r}"
        )
    return {
        "status": "PASS",
        "stage": stage,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "planned": len(selected),
        "completed": len(completed),
        "labels_mounted": False,
    }


__all__ = [
    "STAGE_ROLE",
    "build_run_config",
    "validate_run_marker",
    "run_plan_row",
    "run_plan_shard",
]
