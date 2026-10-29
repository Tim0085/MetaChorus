"""H5 execution shards, stage registries, gates, and promotion state machine."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import (
    atomic_write_csv,
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
    utc_now,
)

from .constants import CONTROLS, H2_RESULT_RELATIVE, SEARCH_MODEL_SEEDS, THIRD_DEVELOPMENT_MODEL_SEED
from .continuation import run_to_fidelity
from .failure import write_scientific_failure
from .planning import build_r3_plan, build_r4_plan, build_third_seed_plan
from .protocol import (
    aggregate_classification_by_seed,
    aggregate_stage_objectives,
    choose_single_r4_set,
    classification_bridge_decision,
    classification_stability,
    clustering_model_reliability,
    joint_normalized_midrank,
    manual_safety,
    pareto_order,
    r3_audit_pool,
    r3_gate_decision,
    r4_promotion,
    three_seed_pairwise_reliability,
    truth_fidelity,
)
from .stage_evaluation import run_stage_evaluation
from .truth import bind_development_truth


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _append_log(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv_once(
    path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]
) -> None:
    if not rows:
        raise RuntimeError(f"refusing to publish empty H5 stage registry: {path.name}")
    if path.is_file():
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        buffer.seek(0)
        expected = [dict(row) for row in csv.DictReader(buffer)]
        if _read_csv(path) != expected:
            raise RuntimeError(f"append-only H5 stage registry conflict: {path.name}")
        return
    atomic_write_csv(path, rows, fieldnames=fieldnames)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if read_json(path) != dict(value):
            raise RuntimeError(f"append-only H5 JSON conflict: {path}")
        return
    atomic_write_json(path, dict(value))


def _stable_shard(run_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(shard_count)


def run_plan_shard(
    *,
    plan_path: Path,
    stage: str,
    shard_index: int,
    shard_count: int,
    repo_root: Path,
    cache_root: Path,
    h1_result_root: Path,
    h4_result_root: Path,
    h5_result_root: Path,
    enforce_target_gpu: bool = True,
    retry_failed: bool = False,
) -> dict[str, Any]:
    if not (0 <= int(shard_index) < int(shard_count) <= 2):
        raise ValueError("H5 permits one or two run-level shards only")
    rows = [
        row
        for row in _read_csv(plan_path)
        if row["stage"] == stage and _stable_shard(row["physical_run_id"], shard_count) == shard_index
    ]
    completed = 0
    reused = 0
    failures: list[str] = []
    failed_root = h5_result_root / "failed_attempts" / stage.lower()
    failed_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["execution_action"] == "EVALUATION_ONLY_REUSE":
            reused += 1
            continue
        working_row = dict(row)
        run_id = str(row["physical_run_id"])
        planned_attempt_index = int(row.get("attempt_index", 0))
        stage_receipt = failed_root / f"{run_id}__attempt_{planned_attempt_index:02d}.json"
        resolution_path = failed_root / f"{run_id}__retry_resolution.json"
        all_failed_receipts = [
            path
            for path in (h5_result_root / "failed_attempts").rglob(
                f"{run_id}__attempt_*.json"
            )
            if read_json(path).get("status") == "FAILED_PRESERVED"
        ]
        if resolution_path.is_file():
            resolution = read_json(resolution_path)
            basis = resolution.get("basis", {})
            if (
                resolution.get("status") != "ENGINEERING_RETRY_RECOVERED"
                or basis.get("physical_run_id") != run_id
                or basis.get("stage") != stage
                or int(basis.get("failed_attempt_index", -1)) != planned_attempt_index
                or int(basis.get("successful_attempt_index", -1))
                != planned_attempt_index + 1
                or resolution.get("identity") != canonical_json_hash(basis)
            ):
                raise RuntimeError(f"invalid H5 retry resolution: {resolution_path}")
            working_row["attempt_index"] = int(basis["successful_attempt_index"])
        elif stage_receipt.is_file():
            if not retry_failed:
                raise RuntimeError(
                    f"{run_id} has a preserved failed attempt; rerun explicitly with retry_failed=True"
                )
            if len(all_failed_receipts) != 1:
                raise RuntimeError(
                    f"engineering retry limit reached for {run_id}: {len(all_failed_receipts)} failures"
                )
            working_row["attempt_index"] = planned_attempt_index + 1
        elif retry_failed and all_failed_receipts:
            raise RuntimeError(
                f"retry request for {run_id} does not match this stage/attempt lineage"
            )
        parent_source = working_row["parent_source"]
        parent_root = (
            h1_result_root
            if parent_source in {"H1_R1", "H1_CONFIG_ONLY"}
            else h4_result_root
            if parent_source == "H4_R2"
            else h5_result_root
        )
        parent_attempt = parent_root / working_row["parent_attempt_relative_path"]
        try:
            completed_payload = run_to_fidelity(
                plan_row=working_row,
                repo_root=repo_root,
                cache_root=cache_root,
                h5_result_root=h5_result_root,
                parent_attempt_root=parent_attempt,
                endpoint_update=int(row["endpoint_update"]),
                enforce_target_gpu=enforce_target_gpu,
            )
            if int(working_row["attempt_index"]) != planned_attempt_index:
                resolution_basis = {
                    "physical_run_id": run_id,
                    "stage": stage,
                    "failed_attempt_index": planned_attempt_index,
                    "successful_attempt_index": int(working_row["attempt_index"]),
                    "failed_attempt_receipt_sha256": sha256_file(stage_receipt),
                    "successful_stage_marker_identity": completed_payload["identity"],
                    "scientific_metrics_observed_before_retry": False,
                    "engineering_retry_count": 1,
                }
                _write_json_once(
                    resolution_path,
                    {
                        "schema_version": "metachorus.v53.m4a-h5.retry-resolution.v1",
                        "status": "ENGINEERING_RETRY_RECOVERED",
                        "basis": resolution_basis,
                        "identity": canonical_json_hash(resolution_basis),
                    },
                )
            completed += 1
        except Exception as exc:
            attempt_index = int(working_row.get("attempt_index", 0))
            receipt = failed_root / f"{run_id}__attempt_{attempt_index:02d}.json"
            if receipt.exists():
                raise RuntimeError(f"failed-attempt receipt already exists: {receipt}") from exc
            basis = {
                "physical_run_id": run_id,
                "stage": stage,
                "attempt_index": attempt_index,
                "failure_kind": type(exc).__name__,
                "error_message": str(exc),
                "scientific_metrics_observed": False,
                "engineering_retry_limit": 1,
                "status": "FAILED_PRESERVED",
            }
            atomic_write_json(
                receipt,
                {
                    "schema_version": "metachorus.v53.m4a-h5.failed-attempt.v1",
                    **basis,
                    "identity": canonical_json_hash(basis),
                },
            )
            failures.append(run_id)
    summary = {
        "stage": stage,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "planned_on_shard": len(rows),
        "completed_training": completed,
        "reused_without_training": reused,
        "failed": len(failures),
        "failed_run_ids": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    if failures:
        raise RuntimeError(f"H5 {stage} shard failures: {failures[:3]}")
    return summary


def close_stage_registry(
    *, stage: str, plan_path: Path, result_root: Path, h4_result_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(plan_path)
    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for row in rows:
        endpoint = int(row["endpoint_update"])
        if row["execution_action"] == "EVALUATION_ONLY_REUSE":
            checkpoint = h4_result_root / row["parent_checkpoint_relative_path"]
            representation = h4_result_root / row["parent_representation_relative_path"]
            run_payload = {
                "optimizer_steps": endpoint,
                "optimizer_steps_this_invocation": 0,
                "model_total_seconds": 0.0,
                "model_seconds_this_invocation": 0.0,
                "peak_gpu_allocated_bytes": 0,
                "peak_gpu_reserved_bytes": 0,
                "continuation_mode": "EXACT_H4_R2_REUSE",
                "status": "PASS",
            }
            checkpoint_sha = row["parent_checkpoint_sha256"]
            representation_sha = row["parent_representation_sha256"]
            checkpoint_relative = row["parent_checkpoint_relative_path"]
            representation_relative = row["parent_representation_relative_path"]
            source_root = "H4_PARENT_IMMUTABLE"
        else:
            run_parent = result_root / "runs" / row["physical_run_id"]
            completed_attempts = [
                attempt
                for attempt in sorted(run_parent.glob("attempt_*"))
                if (attempt / f"H5_STAGE_{endpoint:06d}_COMPLETE.json").is_file()
            ]
            if len(completed_attempts) != 1:
                raise RuntimeError(
                    f"H5 stage requires exactly one completed attempt for "
                    f"{row['physical_run_id']}@{endpoint}, observed {len(completed_attempts)}"
                )
            attempt = completed_attempts[0]
            successful_attempt_index = int(attempt.name.rsplit("_", 1)[-1])
            marker = read_json(attempt / f"H5_STAGE_{endpoint:06d}_COMPLETE.json")
            if marker.get("status") != "PASS":
                raise RuntimeError(f"H5 stage marker not PASS: {row['physical_run_id']}")
            checkpoint = attempt / "checkpoints" / f"update_{endpoint:06d}.pt"
            representation = attempt / "representations" / f"update_{endpoint:06d}.npy"
            run_payload = marker
            checkpoint_sha = marker["checkpoint_sha256"]
            representation_sha = marker["representation_sha256"]
            checkpoint_relative = str(checkpoint.relative_to(result_root)).replace("\\", "/")
            representation_relative = str(representation.relative_to(result_root)).replace("\\", "/")
            source_root = "H5_RESULT_ROOT"
        if not checkpoint.is_file() or not representation.is_file():
            raise RuntimeError(f"H5 stage artifacts missing: {row['physical_run_id']}")
        if sha256_file(checkpoint) != checkpoint_sha or sha256_file(representation) != representation_sha:
            raise RuntimeError(f"H5 stage artifact hash mismatch: {row['physical_run_id']}")
        common = {
            "physical_run_id": row["physical_run_id"],
            "dataset": row["dataset"],
            "candidate_set_id": row["candidate_set_id"],
            "candidate_set_hash": row["candidate_set_hash"],
            "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
            "model_seed": int(row["model_seed"]),
            "endpoint_update": endpoint,
            "successful_attempt_index": (
                0 if row["execution_action"] == "EVALUATION_ONLY_REUSE" else successful_attempt_index
            ),
        }
        run_rows.append(
            {
                **common,
                "continuation_mode": run_payload["continuation_mode"],
                "optimizer_steps": int(run_payload["optimizer_steps"]),
                "optimizer_steps_this_invocation": int(run_payload["optimizer_steps_this_invocation"]),
                "model_total_seconds": float(run_payload["model_total_seconds"]),
                "model_seconds_this_invocation": float(run_payload["model_seconds_this_invocation"]),
                "peak_gpu_allocated_bytes": int(run_payload["peak_gpu_allocated_bytes"]),
                "peak_gpu_reserved_bytes": int(run_payload["peak_gpu_reserved_bytes"]),
                "training_reexecuted": row["execution_action"] != "EVALUATION_ONLY_REUSE",
                "status": "PASS",
            }
        )
        checkpoint_rows.append(
            {
                **common,
                "source_root": source_root,
                "checkpoint_relative_path": checkpoint_relative,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "representation_relative_path": representation_relative,
                "representation_sha256": representation_sha,
                "representation_size_bytes": representation.stat().st_size,
                "strict_state_complete": True,
                "status": "PASS",
            }
        )
    prefix = "TC_LPMSO_H5_THIRD_SEED" if stage == "THIRD_SEED_R3_AUDIT" else f"TC_LPMSO_H5_{stage}"
    run_registry_path = result_root / f"{prefix}_RUN_REGISTRY.csv"
    checkpoint_registry_path = result_root / f"{prefix}_CHECKPOINT_REGISTRY.csv"
    _write_csv_once(
        run_registry_path, run_rows, fieldnames=list(run_rows[0])
    )
    _write_csv_once(
        checkpoint_registry_path,
        checkpoint_rows,
        fieldnames=list(checkpoint_rows[0]),
    )
    plan_seal_path = result_root / f"{prefix}_PLAN_SEAL.json"
    if not plan_seal_path.is_file():
        raise RuntimeError(f"missing H5 stage plan seal: {plan_seal_path.name}")
    seal_basis = {
        "stage": stage,
        "endpoint_update": int(rows[0]["endpoint_update"]),
        "plan_sha256": sha256_file(plan_path),
        "plan_seal_sha256": sha256_file(plan_seal_path),
        "run_registry_sha256": sha256_file(run_registry_path),
        "checkpoint_registry_sha256": sha256_file(checkpoint_registry_path),
        "run_count": len(run_rows),
        "checkpoint_count": len(checkpoint_rows),
        "all_status_pass": all(row["status"] == "PASS" for row in run_rows)
        and all(row["status"] == "PASS" for row in checkpoint_rows),
    }
    _write_json_once(
        result_root / f"{prefix}_REGISTRY_SEAL.json",
        {
            "schema_version": "metachorus.v53.m4a-h5.stage-registry-seal.v1",
            "status": "SEALED_BEFORE_STAGE_EVALUATION",
            "basis": seal_basis,
            "identity": canonical_json_hash(seal_basis),
        },
    )
    return run_rows, checkpoint_rows


def _objectives_from_metrics(metrics: Mapping[str, Sequence[Mapping[str, Any]]]):
    by_seed: list[dict[str, Any]] = []
    objectives: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        dataset_by_seed = aggregate_classification_by_seed(metrics["classification"], dataset)
        by_seed.extend(dataset_by_seed)
        objectives.extend(
            aggregate_stage_objectives(
                dataset_by_seed,
                metrics["clustering_summary"],
                dataset=dataset,
                model_seeds=SEARCH_MODEL_SEEDS,
            )
        )
    return by_seed, objectives


def evaluate_r2(
    *,
    result_root: Path,
    h2_result_root: Path,
    h4_result_root: Path,
    h1_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    close_stage_registry(
        stage="R2",
        plan_path=result_root / "TC_LPMSO_H5_R2_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
    )
    metrics = run_stage_evaluation(
        stage="R2",
        plan_path=result_root / "TC_LPMSO_H5_R2_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
        private_mount_root=private_mount_root,
        workers=workers,
    )
    truth_rows, _ = bind_development_truth(h2_result_root=h2_result_root, result_root=result_root)
    by_seed, objectives = _objectives_from_metrics(metrics)
    stability = [
        classification_stability(metrics["classification"], dataset=dataset, fidelity=1000)
        for dataset in ("acm", "freebase")
    ]
    atomic_write_csv(
        result_root / "TC_LPMSO_H5_R2_CLASSIFICATION_STABILITY.csv",
        stability,
        fieldnames=list(stability[0]),
    )
    fidelity: list[dict[str, Any]] = []
    clustering_diagnostics: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        fidelity.append(
            truth_fidelity(
                objectives,
                truth_rows,
                dataset=dataset,
                stage_field="Q_cls",
                truth_field="classification_truth_order_score",
                task="classification",
            )
        )
        clustering_diagnostics.append(
            truth_fidelity(
                objectives,
                truth_rows,
                dataset=dataset,
                stage_field="Q_clu",
                truth_field="clustering_truth_order_score",
                task="clustering_diagnostic_only",
            )
        )
        joint = joint_normalized_midrank([row for row in objectives if row["dataset"] == dataset])
        clustering_diagnostics.append(
            truth_fidelity(
                joint,
                truth_rows,
                dataset=dataset,
                stage_field="joint_order_score",
                truth_field="joint_truth_order_score",
                task="joint_diagnostic_only",
            )
        )
    audit_rows = fidelity + clustering_diagnostics
    for row in audit_rows:
        row["used_for_r2_gate"] = row["task"] == "classification"
    atomic_write_csv(
        result_root / "TC_LPMSO_H5_R2_FIDELITY_AUDIT.csv",
        audit_rows,
        fieldnames=list(audit_rows[0]),
    )
    decision_basis = classification_bridge_decision(stability, fidelity)
    decision_basis["r2_identity_count"] = len(metrics["search"])
    decision_basis["clustering_diagnostic_rows"] = len(clustering_diagnostics)
    decision = {
        "schema_version": "metachorus.v53.m4a-h5.r2-bridge-decision.v1",
        **decision_basis,
        "identity": canonical_json_hash(decision_basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_R2_BRIDGE_DECISION.json", decision)
    seal_basis = {
        "decision_identity": decision["identity"],
        "run_plan_sha256": sha256_file(result_root / "TC_LPMSO_H5_R2_RUN_PLAN.csv"),
        "run_registry_sha256": sha256_file(result_root / "TC_LPMSO_H5_R2_RUN_REGISTRY.csv"),
        "search_metrics_sha256": sha256_file(result_root / "TC_LPMSO_H5_R2_SEARCH_METRICS.csv"),
        "r2_no_pruning": True,
        "all_sets_continue_if_pass": True,
        "r3_started": False,
    }
    seal = {
        "schema_version": "metachorus.v53.m4a-h5.r2-bridge-seal.v1",
        "status": "SEALED_BEFORE_R3" if decision["status"] == "H5_R2_BRIDGE_PASS" else "NOT_ELIGIBLE_R2_BRIDGE_FAIL",
        "basis": seal_basis,
        "identity": canonical_json_hash(seal_basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_R2_BRIDGE_SEAL.json", seal)
    if decision["status"] == "H5_R2_BRIDGE_PASS":
        build_r3_plan(result_root=result_root, h4_result_root=h4_result_root)
    else:
        write_scientific_failure(
            result_root=result_root,
            failure_code=decision["status"],
            decision_path=result_root / "TC_LPMSO_H5_R2_BRIDGE_DECISION.json",
            counts={"r2_identity_count": len(metrics["search"]), "r3_started": False},
        )
    return decision


def evaluate_r3(
    *,
    result_root: Path,
    h2_result_root: Path,
    h4_result_root: Path,
    h1_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    close_stage_registry(
        stage="R3",
        plan_path=result_root / "TC_LPMSO_H5_R3_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
    )
    metrics = run_stage_evaluation(
        stage="R3",
        plan_path=result_root / "TC_LPMSO_H5_R3_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
        private_mount_root=private_mount_root,
        workers=workers,
    )
    truth_rows, _ = bind_development_truth(h2_result_root=h2_result_root, result_root=result_root)
    by_seed, objectives = _objectives_from_metrics(metrics)
    cls_stability = [
        classification_stability(metrics["classification"], dataset=dataset, fidelity=2000)
        for dataset in ("acm", "freebase")
    ]
    clu_stability = [
        clustering_model_reliability(metrics["clustering_summary"], dataset=dataset)
        for dataset in ("acm", "freebase")
    ]
    atomic_write_csv(result_root / "TC_LPMSO_H5_R3_CLASSIFICATION_STABILITY.csv", cls_stability, fieldnames=list(cls_stability[0]))
    atomic_write_csv(result_root / "TC_LPMSO_H5_R3_CLUSTERING_STABILITY.csv", clu_stability, fieldnames=list(clu_stability[0]))
    cls_fidelity = []
    clu_fidelity = []
    joint_fidelity = []
    joint_rows: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        dataset_objectives = [row for row in objectives if row["dataset"] == dataset]
        cls_fidelity.append(
            truth_fidelity(dataset_objectives, truth_rows, dataset=dataset, stage_field="Q_cls", truth_field="classification_truth_order_score", task="classification")
        )
        clu_fidelity.append(
            truth_fidelity(dataset_objectives, truth_rows, dataset=dataset, stage_field="Q_clu", truth_field="clustering_truth_order_score", task="clustering")
        )
        joint = joint_normalized_midrank(dataset_objectives)
        joint_rows.extend(joint)
        joint_fidelity.append(
            truth_fidelity(joint, truth_rows, dataset=dataset, stage_field="joint_order_score", truth_field="joint_truth_order_score", task="joint")
        )
    fidelity = cls_fidelity + clu_fidelity + joint_fidelity
    atomic_write_csv(result_root / "TC_LPMSO_H5_R3_FIDELITY_AUDIT.csv", fidelity, fieldnames=list(fidelity[0]))
    decision_basis = r3_gate_decision(
        classification_stability_rows=cls_stability,
        clustering_stability_rows=clu_stability,
        classification_fidelity_rows=cls_fidelity,
        clustering_fidelity_rows=clu_fidelity,
        joint_fidelity_rows=joint_fidelity,
    )
    decision = {
        "schema_version": "metachorus.v53.m4a-h5.r3-gate-decision.v1",
        **decision_basis,
        "identity": canonical_json_hash(decision_basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_R3_GATE_DECISION.json", decision)
    if decision["status"] != "TC_LPMSO_H5_R3_GATE_PASS":
        write_scientific_failure(
            result_root=result_root,
            failure_code=decision["status"],
            decision_path=result_root / "TC_LPMSO_H5_R3_GATE_DECISION.json",
            counts={"r3_identity_count": len(metrics["search"]), "third_seed_started": False},
        )
        return decision
    ordered: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        ordered.extend(pareto_order([row for row in objectives if row["dataset"] == dataset]))
    atomic_write_csv(result_root / "TC_LPMSO_H5_R3_TWO_SEED_PARETO.csv", ordered, fieldnames=list(ordered[0]))
    pool = r3_audit_pool(ordered, "acm") + r3_audit_pool(ordered, "freebase")
    atomic_write_csv(result_root / "TC_LPMSO_H5_R3_AUDIT_POOL_TABLE.csv", pool, fieldnames=list(pool[0]))
    seal_basis = {
        "pareto_sha256": sha256_file(result_root / "TC_LPMSO_H5_R3_TWO_SEED_PARETO.csv"),
        "pool_sha256": sha256_file(result_root / "TC_LPMSO_H5_R3_AUDIT_POOL_TABLE.csv"),
        "acm_unique_count": sum(row["dataset"] == "acm" for row in pool),
        "freebase_unique_count": sum(row["dataset"] == "freebase" for row in pool),
        "third_seed_started": False,
    }
    seal = {
        "schema_version": "metachorus.v53.m4a-h5.r3-audit-pool-seal.v1",
        "status": "SEALED_BEFORE_THIRD_SEED",
        "basis": seal_basis,
        "identity": canonical_json_hash(seal_basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_R3_AUDIT_POOL_SEAL.json", seal)
    build_third_seed_plan(result_root=result_root, h1_result_root=h1_result_root)
    return decision


def evaluate_third_seed(
    *,
    result_root: Path,
    h4_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    close_stage_registry(
        stage="THIRD_SEED_R3_AUDIT",
        plan_path=result_root / "TC_LPMSO_H5_THIRD_SEED_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
    )
    third = run_stage_evaluation(
        stage="THIRD_SEED_R3_AUDIT",
        plan_path=result_root / "TC_LPMSO_H5_THIRD_SEED_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
        private_mount_root=private_mount_root,
        workers=workers,
    )
    r3 = _read_csv(result_root / "TC_LPMSO_H5_R3_SEARCH_METRICS.csv")
    combined = [dict(row) for row in r3] + [dict(row) for row in third["search"]]
    reliability = []
    for dataset in ("acm", "freebase"):
        reliability.append(three_seed_pairwise_reliability(combined, dataset=dataset, task_field="mean_F_cls"))
        reliability.append(three_seed_pairwise_reliability(combined, dataset=dataset, task_field="mean_F_clu"))
    atomic_write_csv(result_root / "TC_LPMSO_H5_THREE_SEED_RELIABILITY.csv", reliability, fieldnames=list(reliability[0]))
    reliable = all(row["status"] == "RELIABLE" for row in reliability)
    decision = {
        "schema_version": "metachorus.v53.m4a-h5.three-seed-decision.v1",
        "status": "H5_THREE_SEED_RELIABILITY_PASS" if reliable else "H5_THREE_SEED_RELIABILITY_FAIL",
        "r4_started": False,
        "basis": {"rows": reliability},
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_THREE_SEED_DECISION.json", decision)
    if not reliable:
        write_scientific_failure(
            result_root=result_root,
            failure_code=decision["status"],
            decision_path=result_root / "TC_LPMSO_H5_THREE_SEED_DECISION.json",
            counts={"third_seed_run_count": len(third["search"]), "r4_started": False},
        )
        return decision
    objectives: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        cls = [
            {
                "dataset": row["dataset"], "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"], "candidate_set_ordinal": row["candidate_set_ordinal"],
                "model_seed": row["model_seed"], "mean_F_cls": row["mean_F_cls"],
            }
            for row in combined if row["dataset"] == dataset
        ]
        clu = [
            {
                "dataset": row["dataset"], "candidate_set_hash": row["candidate_set_hash"],
                "model_seed": row["model_seed"], "mean_F_clu": row["mean_F_clu"],
                "assignment_stability_mean": row["assignment_stability_mean"],
            }
            for row in combined if row["dataset"] == dataset
        ]
        objectives.extend(
            aggregate_stage_objectives(
                cls, clu, dataset=dataset,
                model_seeds=(*SEARCH_MODEL_SEEDS, THIRD_DEVELOPMENT_MODEL_SEED),
            )
        )
    ordered = []
    for dataset in ("acm", "freebase"):
        ordered.extend(pareto_order([row for row in objectives if row["dataset"] == dataset]))
    promotion = r4_promotion(ordered, "acm") + r4_promotion(ordered, "freebase")
    atomic_write_csv(result_root / "TC_LPMSO_H5_R4_PROMOTION_TABLE.csv", promotion, fieldnames=list(promotion[0]))
    seal_basis = {
        "promotion_sha256": sha256_file(result_root / "TC_LPMSO_H5_R4_PROMOTION_TABLE.csv"),
        "three_seed_reliability_sha256": sha256_file(result_root / "TC_LPMSO_H5_THREE_SEED_RELIABILITY.csv"),
        "acm_unique_set_count": sum(row["dataset"] == "acm" for row in promotion),
        "freebase_unique_set_count": sum(row["dataset"] == "freebase" for row in promotion),
        "r4_started": False,
    }
    seal = {
        "schema_version": "metachorus.v53.m4a-h5.r4-promotion-seal.v1",
        "status": "SEALED_BEFORE_R4",
        "basis": seal_basis,
        "identity": canonical_json_hash(seal_basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_R4_PROMOTION_SEAL.json", seal)
    build_r4_plan(result_root=result_root)
    return decision


def evaluate_r4(
    *,
    result_root: Path,
    h4_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    close_stage_registry(
        stage="R4",
        plan_path=result_root / "TC_LPMSO_H5_R4_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
    )
    metrics = run_stage_evaluation(
        stage="R4",
        plan_path=result_root / "TC_LPMSO_H5_R4_RUN_PLAN.csv",
        result_root=result_root,
        h4_result_root=h4_result_root,
        private_mount_root=private_mount_root,
        workers=workers,
    )
    objectives: list[dict[str, Any]] = []
    ordered: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        by_seed = aggregate_classification_by_seed(metrics["classification"], dataset)
        dataset_objectives = aggregate_stage_objectives(
            by_seed,
            metrics["clustering_summary"],
            dataset=dataset,
            model_seeds=(*SEARCH_MODEL_SEEDS, THIRD_DEVELOPMENT_MODEL_SEED),
        )
        objectives.extend(dataset_objectives)
        ordered.extend(pareto_order(dataset_objectives))
    atomic_write_csv(result_root / "TC_LPMSO_H5_R4_PARETO_FRONTS.csv", ordered, fieldnames=list(ordered[0]))
    search_by_key = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in metrics["search"]
    }
    comparisons: list[dict[str, Any]] = []
    safety_by_dataset_hash: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in ("acm", "freebase"):
        manual_hash = CONTROLS[dataset]["manual"][1]
        manual = {
            seed: search_by_key[(dataset, manual_hash, seed)]
            for seed in (*SEARCH_MODEL_SEEDS, THIRD_DEVELOPMENT_MODEL_SEED)
        }
        for row in [item for item in ordered if item["dataset"] == dataset]:
            candidate_hash = str(row["candidate_set_hash"])
            candidate = {
                seed: search_by_key[(dataset, candidate_hash, seed)]
                for seed in (*SEARCH_MODEL_SEEDS, THIRD_DEVELOPMENT_MODEL_SEED)
            }
            safety = manual_safety(candidate, manual)
            safety_by_dataset_hash[(dataset, candidate_hash)] = safety
            comparisons.append(
                {
                    "dataset": dataset,
                    "candidate_set_id": row["candidate_set_id"],
                    "candidate_set_hash": candidate_hash,
                    "candidate_set_ordinal": row["candidate_set_ordinal"],
                    **safety,
                }
            )
    atomic_write_csv(result_root / "TC_LPMSO_H5_MANUAL_AUTOMATIC_COMPARISON.csv", comparisons, fieldnames=list(comparisons[0]))
    selections = []
    for dataset in ("acm", "freebase"):
        safety = {
            candidate_hash: value
            for (candidate_dataset, candidate_hash), value in safety_by_dataset_hash.items()
            if candidate_dataset == dataset
        }
        selections.append(choose_single_r4_set(ordered, safety, dataset=dataset))
    if any(row["status"] != "SELECTED" for row in selections):
        status = "TC_LPMSO_H5_DEVELOPMENT_FAIL"
    elif any(row["selected_manual"] for row in selections):
        status = "TC_LPMSO_H5_SELECTED_MANUAL"
    elif any(float(row["manual_safety"]["mean_joint_delta"]) > 0 for row in selections):
        status = "TC_LPMSO_H5_DEVELOPMENT_PASS"
    else:
        status = "TC_LPMSO_H5_DEVELOPMENT_NONINFERIOR"
    selection_payload = {
        "schema_version": "metachorus.v53.m4a-h5.r4-development-selection.v1",
        "status": status,
        "selections": selections,
        "single_complete_set_per_dataset": True,
        "taskwise_set_splicing": False,
        "h5_locked_report_access": False,
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_R4_DEVELOPMENT_SELECTION.json", selection_payload)
    atomic_write_json(
        result_root / "TC_LPMSO_H5_SELECTED_PATH_SET_REGISTRY.json",
        {
            "schema_version": "metachorus.v53.m4a-h5.selected-path-set-registry.v1",
            "status": status,
            "datasets": selections,
        },
    )
    decision = {
        "schema_version": "metachorus.v53.m4a-h5.development-decision.v1",
        "status": status,
        "s_eval_started": False,
        "h5_locked_report_access": False,
        "selections": selections,
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_DEVELOPMENT_DECISION.json", decision)
    if status == "TC_LPMSO_H5_DEVELOPMENT_FAIL":
        write_scientific_failure(
            result_root=result_root,
            failure_code=status,
            decision_path=result_root / "TC_LPMSO_H5_DEVELOPMENT_DECISION.json",
            counts={"r4_identity_count": len(metrics["search"]), "s_eval_started": False},
        )
    return decision
