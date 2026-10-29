"""H6 execution shards, registry closure, three-seed gate, and conditional R4."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file
from metachorus_v53.m4a_h4.continuation import checkpoint_state_audit
from metachorus_v53.m4a_h5.truth import bind_development_truth

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    safe_relative,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    ALL_DEVELOPMENT_MODEL_SEEDS,
    CONTROLS,
    EXPECTED_POOL_COUNTS,
    EXPECTED_POOL_TOTAL,
    FUTURE_S_EVAL_MODEL_SEEDS,
    PARENT_HISTORICAL_FAILED_ATTEMPTS,
    SEARCH_MODEL_SEEDS,
    THIRD_DEVELOPMENT_MODEL_SEED,
)
from .continuation import run_plan_row, validate_stage_marker
from .evaluation import STAGE_PREFIX, canonical_plan_row, run_stage_evaluation
from .failure import write_failure
from .planning import (
    build_r4_plan,
    build_third_seed_plan,
    validate_canary_gate,
    validate_canary_plan_rows,
    validate_pool_seal,
    validate_r4_parent_lineage,
    validate_r4_plan_rows,
    validate_third_seed_parent_lineage,
    validate_third_seed_plan_rows,
)
from .protocol import (
    aggregate_three_seed_objectives,
    choose_single_r4_set,
    classification_seed_stability,
    clustering_seed_stability,
    joint_normalized_midrank,
    manual_safety,
    pareto_order,
    r4_promotion,
    three_seed_gate_decision,
    three_seed_pairwise_reliability,
    truth_fidelity,
)


def _stable_shard(run_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(shard_count)


def _plan_seal_path(result_root: Path, stage: str) -> Path:
    if stage == "CANARY":
        return result_root / "M4A_H6_CANARY_PLAN_SEAL.json"
    return result_root / f"{STAGE_PREFIX[stage]}_PLAN_SEAL.json"


def _validate_plan_seal(
    *, plan_path: Path, result_root: Path, stage: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not rows or any(str(row["stage"]) != stage for row in rows):
        raise RuntimeError(f"invalid or empty H6 {stage} plan")
    seal = read_json(_plan_seal_path(result_root, stage))
    basis = seal.get("basis", {})
    canonical_rows = [canonical_plan_row(row) for row in rows]
    if (
        seal.get("schema_version") != "metachorus.v53.m4a-h6.execution-plan-seal.v1"
        or seal.get("status") != "SEALED_BEFORE_GPU_TRAINING"
        or seal.get("identity") != canonical_json_hash(basis)
        or basis.get("stage") != stage
        or int(basis.get("row_count", -1)) != len(rows)
        or basis.get("rows_identity") != canonical_json_hash(canonical_rows)
        or any(str(row.get("plan_seal_sha256")) != seal.get("identity") for row in rows)
        or basis.get("labels_mounted") is not False
        or basis.get("run_level_parallel_only") is not True
        or int(basis.get("maximum_gpu_count", -1)) != 2
        or basis.get("h5_locked_report_access") is not False
        or basis.get("s_eval_started") is not False
    ):
        raise RuntimeError(f"H6 {stage} plan seal mismatch")
    return seal


def _failure_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = read_json(path)
    basis = value.get("basis", {})
    if (
        value.get("schema_version") != "metachorus.v53.m4a-h6.failed-attempt.v1"
        or value.get("status") != "FAILED_PRESERVED"
        or value.get("identity") != canonical_json_hash(basis)
    ):
        raise RuntimeError(f"invalid H6 failed-attempt receipt: {path}")
    return value, basis


def _root_cause_counts(result_root: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    root = result_root / "failed_attempts"
    if not root.is_dir():
        return output
    for path in root.rglob("*__attempt_*.json"):
        _, basis = _failure_payload(path)
        root_cause = str(basis["root_cause_identity"])
        output.setdefault(root_cause, set()).add(str(basis["physical_run_id"]))
    return output


def run_plan_shard(
    *,
    plan_path: Path,
    stage: str,
    shard_index: int,
    shard_count: int,
    repo_root: Path,
    cache_root: Path,
    h5_result_root: Path,
    h6_result_root: Path,
    enforce_target_gpu: bool = True,
    retry_failed: bool = False,
) -> dict[str, Any]:
    if stage not in {"CANARY", "THIRD_SEED_R3_AUDIT", "R4"}:
        raise ValueError(stage)
    if not (0 <= int(shard_index) < int(shard_count) <= 2):
        raise ValueError("H6 permits one or two run-level shards only")
    if (h6_result_root / "M4A_H6_COMMON_ROOT_CAUSE_PAUSE.json").exists():
        raise RuntimeError("H6 is paused by the common-root-cause fuse")
    all_rows = read_csv(plan_path)
    _validate_plan_seal(
        plan_path=plan_path, result_root=h6_result_root, stage=stage, rows=all_rows
    )
    if stage == "CANARY":
        validate_canary_plan_rows(all_rows)
    elif stage == "THIRD_SEED_R3_AUDIT":
        validate_canary_gate(h6_result_root)
        pool, _ = validate_pool_seal(h6_result_root, h5_result_root)
        validate_third_seed_plan_rows(all_rows, pool_rows=pool)
        validate_third_seed_parent_lineage(
            all_rows,
            h5_result_root=h5_result_root,
            h6_result_root=h6_result_root,
        )
    else:
        validate_r4_plan_rows(all_rows)
        validate_r4_parent_lineage(
            all_rows,
            h5_result_root=h5_result_root,
            h6_result_root=h6_result_root,
        )
    rows = [
        row
        for row in all_rows
        if _stable_shard(str(row["physical_run_id"]), shard_count) == int(shard_index)
    ]
    failed_root = h6_result_root / "failed_attempts" / stage.lower()
    failed_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures: list[str] = []
    for row in rows:
        working = dict(row)
        run_id = str(row["physical_run_id"])
        planned_attempt = int(row.get("attempt_index", 0))
        receipt_path = failed_root / f"{run_id}__attempt_{planned_attempt:02d}.json"
        resolution_path = failed_root / f"{run_id}__retry_resolution.json"
        all_receipts = sorted((h6_result_root / "failed_attempts").rglob(f"{run_id}__attempt_*.json"))
        if resolution_path.is_file():
            resolution = read_json(resolution_path)
            basis = resolution.get("basis", {})
            if (
                resolution.get("schema_version") != "metachorus.v53.m4a-h6.retry-resolution.v1"
                or resolution.get("status") != "ENGINEERING_RETRY_RECOVERED"
                or resolution.get("identity") != canonical_json_hash(basis)
                or basis.get("physical_run_id") != run_id
                or basis.get("stage") != stage
                or int(basis.get("failed_attempt_index", -1)) != planned_attempt
                or int(basis.get("successful_attempt_index", -1)) != planned_attempt + 1
            ):
                raise RuntimeError(f"invalid H6 retry resolution: {resolution_path}")
            working["attempt_index"] = int(basis["successful_attempt_index"])
        elif receipt_path.is_file():
            _failure_payload(receipt_path)
            if not retry_failed:
                raise RuntimeError(
                    f"{run_id} has a preserved failure; use the single explicit engineering retry"
                )
            if len(all_receipts) != 1:
                raise RuntimeError(f"H6 engineering retry limit reached for {run_id}")
            working["attempt_index"] = planned_attempt + 1
        elif retry_failed and all_receipts:
            raise RuntimeError(f"retry lineage mismatch for {run_id}")
        parent_source = str(working["parent_source"])
        if parent_source in {"H5_R3_TEMPLATE_CONFIG_ONLY", "H5_RESULT_ROOT"}:
            parent_root = h5_result_root
        elif parent_source == "H6_RESULT_ROOT":
            parent_root = h6_result_root
        else:
            raise RuntimeError(f"unsupported H6 parent source: {parent_source}")
        relative = safe_relative(str(working["parent_attempt_relative_path"]))
        parent_attempt = parent_root.joinpath(*relative.parts)
        try:
            marker = run_plan_row(
                plan_row=working,
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=h6_result_root,
                parent_attempt_root=parent_attempt,
                enforce_target_gpu=enforce_target_gpu,
            )
            if int(working["attempt_index"]) != planned_attempt:
                failed_value, _ = _failure_payload(receipt_path)
                resolution_basis = {
                    "physical_run_id": run_id,
                    "stage": stage,
                    "failed_attempt_index": planned_attempt,
                    "successful_attempt_index": int(working["attempt_index"]),
                    "failed_attempt_receipt_sha256": sha256_file(receipt_path),
                    "failed_attempt_identity": failed_value["identity"],
                    "successful_stage_marker_identity": marker["identity"],
                    "scientific_metrics_observed_before_retry": False,
                    "engineering_retry_count": 1,
                }
                write_json_once(
                    resolution_path,
                    envelope(
                        "metachorus.v53.m4a-h6.retry-resolution.v1",
                        "ENGINEERING_RETRY_RECOVERED",
                        resolution_basis,
                    ),
                )
            completed += 1
        except Exception as exc:
            attempt_index = int(working.get("attempt_index", 0))
            failure_path = failed_root / f"{run_id}__attempt_{attempt_index:02d}.json"
            root_cause_basis = {
                "failure_kind": type(exc).__name__,
                "error_message": str(exc),
            }
            failure_basis = {
                "physical_run_id": run_id,
                "stage": stage,
                "attempt_index": attempt_index,
                **root_cause_basis,
                "root_cause_identity": canonical_json_hash(root_cause_basis),
                "scientific_metrics_observed": False,
                "optimizer_steps_after_boundary": 0,
                "engineering_retry_limit": 1,
                "labels_mounted": False,
                "h5_locked_report_access": False,
                "s_eval_started": False,
            }
            write_json_once(
                failure_path,
                envelope(
                    "metachorus.v53.m4a-h6.failed-attempt.v1",
                    "FAILED_PRESERVED",
                    failure_basis,
                ),
            )
            failures.append(run_id)
            root_causes = _root_cause_counts(h6_result_root)
            affected = root_causes[failure_basis["root_cause_identity"]]
            if stage != "CANARY" and len(affected) >= 2:
                pause_basis = {
                    "root_cause_identity": failure_basis["root_cause_identity"],
                    "affected_physical_run_ids": sorted(affected),
                    "affected_scientific_identity_count": len(affected),
                    "threshold": 2,
                    "fanout_must_stop": True,
                    "completed_runs_preserved": True,
                    "h5_locked_report_access": False,
                    "s_eval_started": False,
                }
                write_json_once(
                    h6_result_root / "M4A_H6_COMMON_ROOT_CAUSE_PAUSE.json",
                    envelope(
                        "metachorus.v53.m4a-h6.common-root-cause-pause.v1",
                        "IMMEDIATE_COMMON_ROOT_CAUSE_PAUSE",
                        pause_basis,
                    ),
                )
                raise RuntimeError("H6 immediate common-root-cause pause") from exc
    summary = {
        "stage": stage,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "planned_on_shard": len(rows),
        "completed_training": completed,
        "failed": len(failures),
        "failed_run_ids": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    if failures:
        raise RuntimeError(f"H6 {stage} shard failures: {failures[:3]}")
    return summary


def close_canary(*, result_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = result_root / "M4A_H6_CANARY_RUN_PLAN.csv"
    rows = read_csv(plan_path)
    _validate_plan_seal(plan_path=plan_path, result_root=result_root, stage="CANARY", rows=rows)
    validate_canary_plan_rows(rows)
    records: list[dict[str, Any]] = []
    for row in rows:
        run_parent = result_root / "canary_runs" / str(row["physical_run_id"])
        attempts = [
            attempt
            for attempt in sorted(run_parent.glob("attempt_*"))
            if (attempt / "H6_CANARY_000020_COMPLETE.json").is_file()
        ]
        if len(attempts) != 1:
            raise RuntimeError(f"H6 canary requires one completed attempt: {row['physical_run_id']}")
        attempt = attempts[0]
        marker_path = attempt / "H6_CANARY_000020_COMPLETE.json"
        marker = validate_stage_marker(marker_path, plan_row=row, endpoint_update=20, run_root=attempt)
        checkpoint = attempt / "checkpoints" / "update_000020.pt"
        representation = attempt / "representations" / "update_000020.npy"
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=20,
            expected_sha256=str(marker["checkpoint_sha256"]),
        )
        array = np.load(representation, allow_pickle=False)
        with (attempt / "RESOURCE_TRACE.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            trace = list(csv.DictReader(stream))
        if (
            not audit["strict_state_complete"]
            or not trace
            or int(trace[-1]["update"]) != 20
            or not np.isfinite(array).all()
            or int(marker["optimizer_steps"]) != 20
            or int(marker["optimizer_steps_this_invocation"]) != 20
            or marker["eligible_for_scientific_result"] is not False
            or marker["reusable_for_science"] is not False
            or marker["labels_mounted"] is not False
            or int(marker["downstream_evaluations"]) != 0
        ):
            raise RuntimeError(f"H6 canary contract failed: {row['physical_run_id']}")
        records.append(
            {
                "physical_run_id": str(row["physical_run_id"]),
                "dataset": str(row["dataset"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "model_seed": int(row["model_seed"]),
                "successful_attempt_index": int(attempt.name.rsplit("_", 1)[-1]),
                "optimizer_steps": 20,
                "model_total_seconds": float(marker["model_total_seconds"]),
                "model_seconds_this_invocation": float(marker["model_seconds_this_invocation"]),
                "gpu_preference": int(row["gpu_preference"]),
                "checkpoint_sha256": str(marker["checkpoint_sha256"]),
                "representation_sha256": str(marker["representation_sha256"]),
                "resource_trace_sha256": sha256_file(attempt / "RESOURCE_TRACE.csv"),
                "stage_marker_identity": str(marker["identity"]),
                "checkpoint_roundtrip_state_complete": True,
                "representation_finite": True,
                "scientific_metrics_observed": False,
                "reusable_for_science": False,
                "labels_mounted": False,
                "status": "PASS",
            }
        )
    results_basis = {
        "run_count": 2,
        "records": sorted(records, key=lambda row: row["dataset"]),
        "all_optimizer_steps_exact": True,
        "all_checkpoint_roundtrips_pass": True,
        "scientific_metrics_observed": False,
        "fanout_started_before_canary_pass": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    results = envelope("metachorus.v53.m4a-h6.canary-results.v1", "PASS", results_basis)
    write_json_once(result_root / "M4A_H6_CANARY_RESULTS.json", results)
    decision_basis = {
        "canary_results_identity": results["identity"],
        "acm_pass": True,
        "freebase_pass": True,
        "fanout_authorized": True,
        "fanout_started_before_decision": False,
        "scientific_metrics_observed": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    decision = envelope(
        "metachorus.v53.m4a-h6.canary-gate-decision.v1",
        "H6_CANARY_PASS",
        decision_basis,
    )
    write_json_once(result_root / "M4A_H6_CANARY_GATE_DECISION.json", decision)
    append_jsonl_once(
        result_root / "M4A_H6_EXECUTION_LOG.jsonl",
        {
            "event_id": "H6_CANARY_GATE_PASS",
            "status": "H6_CANARY_PASS",
            "canary_results_identity": results["identity"],
            "fanout_authorized": True,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    return results, decision


def close_stage_registry(
    *, stage: str, plan_path: Path, result_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if stage not in STAGE_PREFIX:
        raise ValueError(stage)
    rows = read_csv(plan_path)
    _validate_plan_seal(plan_path=plan_path, result_root=result_root, stage=stage, rows=rows)
    if stage == "THIRD_SEED_R3_AUDIT":
        pool, _ = validate_pool_seal(result_root)
        validate_third_seed_plan_rows(rows, pool_rows=pool)
    else:
        validate_r4_plan_rows(rows)
    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for row in rows:
        endpoint = int(row["endpoint_update"])
        run_parent = result_root / "runs" / str(row["physical_run_id"])
        attempts = [
            attempt
            for attempt in sorted(run_parent.glob("attempt_*"))
            if (attempt / f"H6_STAGE_{endpoint:06d}_COMPLETE.json").is_file()
        ]
        if len(attempts) != 1:
            raise RuntimeError(
                f"H6 {stage} requires exactly one completed attempt for "
                f"{row['physical_run_id']}, observed {len(attempts)}"
            )
        attempt = attempts[0]
        marker_path = attempt / f"H6_STAGE_{endpoint:06d}_COMPLETE.json"
        marker = validate_stage_marker(marker_path, plan_row=row, endpoint_update=endpoint, run_root=attempt)
        checkpoint = attempt / "checkpoints" / f"update_{endpoint:06d}.pt"
        representation = attempt / "representations" / f"update_{endpoint:06d}.npy"
        strict = checkpoint_state_audit(
            checkpoint,
            expected_update=endpoint,
            expected_sha256=str(marker["checkpoint_sha256"]),
        )
        if not strict["strict_state_complete"]:
            raise RuntimeError(f"H6 strict checkpoint state incomplete: {row['physical_run_id']}")
        representation_array = np.load(representation, allow_pickle=False)
        if (
            representation_array.ndim != 2
            or representation_array.size == 0
            or not np.isfinite(representation_array).all()
        ):
            raise RuntimeError(f"H6 representation is empty, non-matrix, or non-finite: {row['physical_run_id']}")
        common = {
            "physical_run_id": str(row["physical_run_id"]),
            "dataset": str(row["dataset"]),
            "candidate_set_id": str(row["candidate_set_id"]),
            "candidate_set_hash": str(row["candidate_set_hash"]),
            "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
            "model_seed": int(row["model_seed"]),
            "endpoint_update": endpoint,
            "successful_attempt_index": int(attempt.name.rsplit("_", 1)[-1]),
        }
        run_rows.append(
            {
                **common,
                "continuation_mode": str(marker["continuation_mode"]),
                "optimizer_steps": int(marker["optimizer_steps"]),
                "optimizer_steps_this_invocation": int(marker["optimizer_steps_this_invocation"]),
                "model_total_seconds": float(marker["model_total_seconds"]),
                "model_seconds_this_invocation": float(marker["model_seconds_this_invocation"]),
                "peak_gpu_allocated_bytes": int(marker["peak_gpu_allocated_bytes"]),
                "peak_gpu_reserved_bytes": int(marker["peak_gpu_reserved_bytes"]),
                "training_reexecuted": int(attempt.name.rsplit("_", 1)[-1]) > int(row.get("attempt_index", 0)),
                "labels_mounted": False,
                "status": "PASS",
            }
        )
        checkpoint_rows.append(
            {
                **common,
                "source_root": "H6_RESULT_ROOT",
                "checkpoint_relative_path": checkpoint.relative_to(result_root).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "representation_relative_path": representation.relative_to(result_root).as_posix(),
                "representation_sha256": sha256_file(representation),
                "representation_size_bytes": representation.stat().st_size,
                "strict_state_complete": True,
                "status": "PASS",
            }
        )
    prefix = STAGE_PREFIX[stage]
    run_path = result_root / f"{prefix}_RUN_REGISTRY.csv"
    checkpoint_path = result_root / f"{prefix}_CHECKPOINT_REGISTRY.csv"
    write_csv_once(run_path, run_rows, fieldnames=list(run_rows[0]))
    write_csv_once(checkpoint_path, checkpoint_rows, fieldnames=list(checkpoint_rows[0]))
    plan_seal_path = _plan_seal_path(result_root, stage)
    seal_basis = {
        "stage": stage,
        "endpoint_update": int(rows[0]["endpoint_update"]),
        "plan_sha256": sha256_file(plan_path),
        "plan_seal_sha256": sha256_file(plan_seal_path),
        "run_registry_sha256": sha256_file(run_path),
        "checkpoint_registry_sha256": sha256_file(checkpoint_path),
        "run_count": len(run_rows),
        "checkpoint_count": len(checkpoint_rows),
        "all_status_pass": True,
    }
    write_json_once(
        result_root / f"{prefix}_REGISTRY_SEAL.json",
        envelope(
            "metachorus.v53.m4a-h6.stage-registry-seal.v1",
            "SEALED_BEFORE_STAGE_EVALUATION",
            seal_basis,
        ),
    )
    return run_rows, checkpoint_rows


def _filter_parent_rows(
    rows: Sequence[Mapping[str, Any]], pool_hashes: set[str]
) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row["candidate_set_hash"]) in pool_hashes]


def _reliability_reasons(
    pairwise: Sequence[Mapping[str, Any]],
    classification: Sequence[Mapping[str, Any]],
    clustering: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons = [
        f"{row['dataset']}:{row['task_field']}:pairwise_reliability"
        for row in pairwise
        if row["status"] != "RELIABLE"
    ]
    reasons.extend(
        f"{row['dataset']}:seed_{row['model_seed']}:classification_stability"
        for row in classification
        if row["status"] != "RELIABLE"
    )
    reasons.extend(
        f"{row['dataset']}:seed_{row['model_seed']}:clustering_stability"
        for row in clustering
        if row["status"] != "RELIABLE"
    )
    return reasons


def evaluate_third_seed_and_gate(
    *,
    result_root: Path,
    h5_result_root: Path,
    h2_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    plan_path = result_root / "TC_LPMSO_H6_THIRD_SEED_RUN_PLAN.csv"
    close_stage_registry(stage="THIRD_SEED_R3_AUDIT", plan_path=plan_path, result_root=result_root)
    third = run_stage_evaluation(
        stage="THIRD_SEED_R3_AUDIT",
        plan_path=plan_path,
        result_root=result_root,
        h5_result_root=h5_result_root,
        private_mount_root=private_mount_root,
        workers=workers,
    )
    pool = read_csv(result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL.csv")
    pool_hashes = {str(row["candidate_set_hash"]) for row in pool}
    old_search = _filter_parent_rows(
        read_csv(h5_result_root / "TC_LPMSO_H5_R3_SEARCH_METRICS.csv"), pool_hashes
    )
    old_classification = _filter_parent_rows(
        read_csv(h5_result_root / "TC_LPMSO_H5_R3_CLASSIFICATION_METRICS.csv"), pool_hashes
    )
    old_clustering = _filter_parent_rows(
        read_csv(h5_result_root / "TC_LPMSO_H5_R3_CLUSTERING_METRICS_SUMMARY.csv"), pool_hashes
    )
    combined_search = [*old_search, *third["search"]]
    combined_classification = [*old_classification, *third["classification"]]
    combined_clustering = [*old_clustering, *third["clustering_summary"]]
    if (
        len(combined_search) != EXPECTED_POOL_TOTAL * 3
        or len(combined_classification) != EXPECTED_POOL_TOTAL * 3 * 5
        or len(combined_clustering) != EXPECTED_POOL_TOTAL * 3
    ):
        raise RuntimeError("H6 combined three-seed evaluation cohort is incomplete")
    pairwise: list[dict[str, Any]] = []
    classification_stability: list[dict[str, Any]] = []
    clustering_stability: list[dict[str, Any]] = []
    objectives: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        pairwise.append(
            three_seed_pairwise_reliability(
                combined_search, dataset=dataset, task_field="mean_F_cls"
            )
        )
        pairwise.append(
            three_seed_pairwise_reliability(
                combined_search, dataset=dataset, task_field="mean_F_clu"
            )
        )
        classification_stability.extend(
            classification_seed_stability(combined_classification, dataset=dataset)
        )
        clustering_stability.extend(
            clustering_seed_stability(combined_clustering, dataset=dataset)
        )
        objectives.extend(
            joint_normalized_midrank(
                aggregate_three_seed_objectives(combined_search, dataset=dataset)
            )
        )
    write_csv_once(
        result_root / "TC_LPMSO_H6_THREE_SEED_PAIRWISE_RELIABILITY.csv",
        pairwise,
        fieldnames=list(pairwise[0]),
    )
    write_csv_once(
        result_root / "TC_LPMSO_H6_THREE_SEED_CLASSIFICATION_STABILITY.csv",
        classification_stability,
        fieldnames=list(classification_stability[0]),
    )
    write_csv_once(
        result_root / "TC_LPMSO_H6_THREE_SEED_CLUSTERING_STABILITY.csv",
        clustering_stability,
        fieldnames=list(clustering_stability[0]),
    )
    write_csv_once(
        result_root / "TC_LPMSO_H6_THREE_SEED_R3_OBJECTIVES.csv",
        objectives,
        fieldnames=list(objectives[0]),
    )
    reasons = _reliability_reasons(pairwise, classification_stability, clustering_stability)
    if reasons:
        fidelity = [
            {
                "dataset": dataset,
                "task": task,
                "candidate_count": EXPECTED_POOL_COUNTS[dataset],
                "spearman_to_3000_truth": "",
                "kendall_to_3000_truth": "",
                "stage_field": "",
                "truth_field": "",
                "truth_accessed_after_pool_seal": False,
                "status": "NOT_RUN_RELIABILITY_FAIL",
            }
            for dataset in ("acm", "freebase")
            for task in ("classification", "clustering", "joint")
        ]
        write_csv_once(
            result_root / "TC_LPMSO_H6_THREE_SEED_R3_FIDELITY_AUDIT.csv",
            fidelity,
            fieldnames=list(fidelity[0]),
        )
        decision_basis = {
            "status": "TC_LPMSO_H6_THREE_SEED_R3_FAIL",
            "reasons": reasons,
            "reliability_gate_pass": False,
            "truth_accessed": False,
            "r4_started": False,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        }
        decision = envelope(
            "metachorus.v53.m4a-h6.three-seed-r3-gate-decision.v1",
            "TC_LPMSO_H6_THREE_SEED_R3_FAIL",
            decision_basis,
        )
        decision_path = result_root / "TC_LPMSO_H6_THREE_SEED_R3_GATE_DECISION.json"
        write_json_once(decision_path, decision)
        write_failure(
            result_root,
            failure_code="TC_LPMSO_H6_THREE_SEED_R3_FAIL",
            evidence_artifact=decision_path.name,
            evidence_sha256=sha256_file(decision_path),
            counts={"third_seed_completed": EXPECTED_POOL_TOTAL, "r4_started": False},
            scientific_gate_failure=True,
            engineering_retry_permitted=False,
        )
        return decision
    _, pool_seal = validate_pool_seal(result_root, h5_result_root)
    truth_rows, truth_binding = bind_development_truth(
        h2_result_root=h2_result_root,
        result_root=h5_result_root,
    )
    fidelity: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        dataset_objectives = [row for row in objectives if row["dataset"] == dataset]
        for task, stage_field, truth_field in (
            ("classification", "Q_cls", "classification_truth_order_score"),
            ("clustering", "Q_clu", "clustering_truth_order_score"),
            ("joint", "joint_order_score", "joint_truth_order_score"),
        ):
            row = truth_fidelity(
                dataset_objectives,
                truth_rows,
                dataset=dataset,
                stage_field=stage_field,
                truth_field=truth_field,
                task=task,
            )
            row.update(
                {
                    "truth_binding_identity": truth_binding["identity"],
                    "pool_seal_identity": pool_seal["identity"],
                    "truth_accessed_after_reliability_gate": True,
                    "status": "AUDITED",
                }
            )
            fidelity.append(row)
    write_csv_once(
        result_root / "TC_LPMSO_H6_THREE_SEED_R3_FIDELITY_AUDIT.csv",
        fidelity,
        fieldnames=list(fidelity[0]),
    )
    decision_basis = three_seed_gate_decision(
        pairwise_rows=pairwise,
        classification_stability_rows=classification_stability,
        clustering_stability_rows=clustering_stability,
        fidelity_rows=fidelity,
    )
    decision_basis.update(
        {
            "reliability_gate_pass": True,
            "truth_accessed_after_reliability_gate": True,
            "truth_binding_identity": truth_binding["identity"],
            "pool_seal_identity": pool_seal["identity"],
            "h5_locked_report_access": False,
            "s_eval_started": False,
        }
    )
    decision = envelope(
        "metachorus.v53.m4a-h6.three-seed-r3-gate-decision.v1",
        str(decision_basis["status"]),
        decision_basis,
    )
    decision_path = result_root / "TC_LPMSO_H6_THREE_SEED_R3_GATE_DECISION.json"
    write_json_once(decision_path, decision)
    if decision["status"] != "TC_LPMSO_H6_THREE_SEED_R3_PASS":
        write_failure(
            result_root,
            failure_code="TC_LPMSO_H6_THREE_SEED_R3_FAIL",
            evidence_artifact=decision_path.name,
            evidence_sha256=sha256_file(decision_path),
            counts={"third_seed_completed": EXPECTED_POOL_TOTAL, "r4_started": False},
            scientific_gate_failure=True,
            engineering_retry_permitted=False,
        )
        return decision
    ordered: list[dict[str, Any]] = []
    promotion: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        dataset_ordered = pareto_order([row for row in objectives if row["dataset"] == dataset])
        ordered.extend(dataset_ordered)
        promotion.extend(r4_promotion(dataset_ordered, dataset=dataset))
    pareto_path = result_root / "TC_LPMSO_H6_R3_PARETO_FRONTS.csv"
    promotion_path = result_root / "TC_LPMSO_H6_R4_PROMOTION_TABLE.csv"
    write_csv_once(pareto_path, ordered, fieldnames=list(ordered[0]))
    write_csv_once(promotion_path, promotion, fieldnames=list(promotion[0]))
    seal_basis = {
        "three_seed_gate_identity": decision["identity"],
        "r3_pareto_sha256": sha256_file(pareto_path),
        "promotion_table_sha256": sha256_file(promotion_path),
        "acm_promotion_slots": 2,
        "freebase_promotion_slots": 10,
        "controls_outside_slots": True,
        "promotion_row_count": len(promotion),
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    write_json_once(
        result_root / "TC_LPMSO_H6_R4_PROMOTION_SEAL.json",
        envelope(
            "metachorus.v53.m4a-h6.r4-promotion-seal.v1",
            "SEALED_BEFORE_R4_TRAINING",
            seal_basis,
        ),
    )
    build_r4_plan(result_root, h5_result_root)
    return decision


def evaluate_r4_and_decide(
    *,
    result_root: Path,
    h5_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    plan_path = result_root / "TC_LPMSO_H6_R4_RUN_PLAN.csv"
    close_stage_registry(stage="R4", plan_path=plan_path, result_root=result_root)
    metrics = run_stage_evaluation(
        stage="R4",
        plan_path=plan_path,
        result_root=result_root,
        h5_result_root=h5_result_root,
        private_mount_root=private_mount_root,
        workers=workers,
    )
    objectives: list[dict[str, Any]] = []
    ordered: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        dataset_objectives = joint_normalized_midrank(
            aggregate_three_seed_objectives(metrics["search"], dataset=dataset)
        )
        objectives.extend(dataset_objectives)
        ordered.extend(pareto_order(dataset_objectives))
    pareto_path = result_root / "TC_LPMSO_H6_R4_PARETO_FRONTS.csv"
    write_csv_once(pareto_path, ordered, fieldnames=list(ordered[0]))
    search_by_key = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in metrics["search"]
    }
    safety_rows: list[dict[str, Any]] = []
    safety_by_hash: dict[str, dict[str, Any]] = {}
    for row in objectives:
        dataset = str(row["dataset"])
        candidate_hash = str(row["candidate_set_hash"])
        manual_hash = CONTROLS[dataset]["manual"][1]
        candidate = {
            seed: {
                "F_cls": float(search_by_key[(dataset, candidate_hash, seed)]["mean_F_cls"]),
                "F_clu": float(search_by_key[(dataset, candidate_hash, seed)]["mean_F_clu"]),
            }
            for seed in ALL_DEVELOPMENT_MODEL_SEEDS
        }
        manual = {
            seed: {
                "F_cls": float(search_by_key[(dataset, manual_hash, seed)]["mean_F_cls"]),
                "F_clu": float(search_by_key[(dataset, manual_hash, seed)]["mean_F_clu"]),
            }
            for seed in ALL_DEVELOPMENT_MODEL_SEEDS
        }
        safety = manual_safety(candidate, manual)
        safety_by_hash[candidate_hash] = safety
        safety_rows.append(
            {
                "dataset": dataset,
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": candidate_hash,
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "manual_candidate_set_hash": manual_hash,
                **safety,
                "complete_set_only": True,
                "taskwise_set_splicing": False,
            }
        )
    write_csv_once(
        result_root / "TC_LPMSO_H6_MANUAL_AUTOMATIC_COMPARISON.csv",
        safety_rows,
        fieldnames=list(safety_rows[0]),
    )
    selections = [
        choose_single_r4_set(ordered, safety_by_hash, dataset=dataset)
        for dataset in ("acm", "freebase")
    ]
    selection_basis = {
        "selections": selections,
        "one_complete_set_per_dataset": len(selections) == 2,
        "taskwise_set_splicing": False,
        "manual_safety_required": True,
        "truth_used_for_r4_selection": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    selection = envelope(
        "metachorus.v53.m4a-h6.r4-development-selection.v1",
        "PASS" if all(row["status"] == "SELECTED" for row in selections) else "FAIL",
        selection_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H6_R4_DEVELOPMENT_SELECTION.json", selection)
    registry_rows = [
        {
            "dataset": row["dataset"],
            "candidate_set_id": row.get("candidate_set_id", ""),
            "candidate_set_hash": row.get("candidate_set_hash", ""),
            "candidate_set_ordinal": row.get("candidate_set_ordinal", ""),
            "selected_manual": row.get("selected_manual", ""),
            "complete_set_only": row.get("complete_set_only", True),
            "taskwise_set_splicing": False,
            "status": row["status"],
        }
        for row in selections
    ]
    write_csv_once(
        result_root / "TC_LPMSO_H6_SELECTED_PATH_SET_REGISTRY.csv",
        registry_rows,
        fieldnames=list(registry_rows[0]),
    )
    if any(row["status"] != "SELECTED" for row in selections):
        scientific_status = "TC_LPMSO_H6_DEVELOPMENT_FAIL"
    elif any(bool(row["selected_manual"]) for row in selections):
        scientific_status = "TC_LPMSO_H6_SELECTED_MANUAL"
    elif any(float(row["manual_safety"]["mean_joint_delta"]) > 0.0 for row in selections):
        scientific_status = "TC_LPMSO_H6_DEVELOPMENT_PASS"
    else:
        scientific_status = "TC_LPMSO_H6_DEVELOPMENT_NONINFERIOR"
    decision_basis = {
        "scientific_status": scientific_status,
        "r4_selection_identity": selection["identity"],
        "selected_datasets": [row["dataset"] for row in selections if row["status"] == "SELECTED"],
        "one_complete_set_per_dataset": all(row["status"] == "SELECTED" for row in selections),
        "taskwise_set_splicing": False,
        "manual_safety_pass": all(row["status"] == "SELECTED" for row in selections),
        "truth_used_for_r4_selection": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    decision = envelope(
        "metachorus.v53.m4a-h6.development-decision.v1",
        scientific_status,
        decision_basis,
    )
    decision_path = result_root / "TC_LPMSO_H6_DEVELOPMENT_DECISION.json"
    write_json_once(decision_path, decision)
    write_text_once(
        result_root / "TC_LPMSO_H6_DEVELOPMENT_DECISION.md",
        'Stage status is recorded in the structured output.\n'
        f"- 状态：`{scientific_status}`\n"
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n',
    )
    if scientific_status in {
        "TC_LPMSO_H6_DEVELOPMENT_PASS",
        "TC_LPMSO_H6_DEVELOPMENT_NONINFERIOR",
    }:
        write_text_once(
            result_root / "TC_LPMSO_H6_S_EVAL_CONFIRMATION_PLAN_PENDING.md",
            'Stage status is recorded in the structured output.\n',
        )
        pending_rows = [
            {
                "dataset": dataset,
                "model_seed": seed,
                "status": "PENDING_USER_APPROVAL",
                "started": False,
            }
            for dataset in ("acm", "freebase")
            for seed in FUTURE_S_EVAL_MODEL_SEEDS
        ]
        write_csv_once(
            result_root / "TC_LPMSO_H6_S_EVAL_CONFIRMATION_MATRIX_PENDING.csv",
            pending_rows,
            fieldnames=list(pending_rows[0]),
        )
        write_json_once(
            result_root / "TC_LPMSO_H6_S_EVAL_CONFIRMATION_BUDGET_PENDING.json",
            envelope(
                "metachorus.v53.m4a-h6.s-eval-budget-pending.v1",
                "PENDING_USER_APPROVAL",
                {
                    "seed_count": len(FUTURE_S_EVAL_MODEL_SEEDS),
                    "dataset_count": 2,
                    "s_eval_started": False,
                },
            ),
        )
        write_json_once(
            result_root / "TC_LPMSO_H6_S_EVAL_APPROVAL_STATUS.json",
            envelope(
                "metachorus.v53.m4a-h6.s-eval-approval-status.v1",
                "NOT_APPROVED_NOT_STARTED",
                {"user_approval_received": False, "s_eval_started": False},
            ),
        )
    if scientific_status == "TC_LPMSO_H6_DEVELOPMENT_FAIL":
        write_failure(
            result_root,
            failure_code="TC_LPMSO_H6_R4_DEVELOPMENT_FAIL",
            evidence_artifact=decision_path.name,
            evidence_sha256=sha256_file(decision_path),
            counts={"r4_run_count": len(read_csv(plan_path)), "selected_dataset_count": 0},
            scientific_gate_failure=True,
            engineering_retry_permitted=False,
        )
    return decision


def authorize_third_seed(*, result_root: Path, h5_result_root: Path) -> dict[str, Any]:
    """Close canaries, then publish the sealed 85-run fan-out plan."""

    close_canary(result_root=result_root)
    return build_third_seed_plan(result_root, h5_result_root)
