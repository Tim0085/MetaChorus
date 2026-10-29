"""Stage closure, frozen selection, and preliminary evaluation for M7F-H3."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash
from metachorus_v53.m6f_m7f_fast.evaluation import (
    PreliminaryLabelSession,
    evaluate_preliminary_representation,
    evaluate_search_representation,
    open_search_mount,
)
from metachorus_v53.m6f_m7f_fast.protocol import METRICS

from .common import (
    envelope,
    read_csv,
    read_json,
    sha256_file,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    ALL_SETS_IDENTITY,
    CACHE_IDENTITY,
    CURRENT_AUTOMATIC_STATUS,
    DATASET,
    EVALUATION_MODEL_SEEDS,
    LABEL_ROLE_MANIFEST_IDENTITY,
    LEGAL_SET_COUNT,
    MANUAL_SET_HASH,
    MANUAL_SET_ID,
    PATH_LIBRARY_IDENTITY,
    PRELIM_ENDPOINT,
    R1_MODEL_SEEDS,
    R1_POOL_SIZE,
    R3_SHORTLIST_SIZE,
    SEARCH_ENDPOINTS,
    SEARCH_MODEL_SEEDS,
)
from .planning import (
    build_continuation_plan,
    build_preliminary_plan,
    canonical_plan_row,
    validate_plan,
)
from .protocol import (
    close_search_selection,
    derive_r1_pool,
    derive_r3_shortlist,
    first_pareto_front,
    preliminary_decision,
)
from .training import ENGINE_STAGE, completed_marker, validate_engine_marker


METRIC_NUMERIC_FIELDS = (
    "candidate_set_ordinal",
    "model_seed",
    "endpoint_update",
    "model_total_seconds",
    "peak_gpu_reserved_bytes",
    *METRICS,
    "assignment_stability_mean",
)


def canonical_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    for field in ("candidate_set_ordinal", "model_seed", "endpoint_update", "peak_gpu_reserved_bytes"):
        if field in value:
            value[field] = int(value[field])
    for field in ("model_total_seconds", *METRICS, "assignment_stability_mean"):
        if field in value:
            value[field] = float(value[field])
    for field in ("prelim_locked_report_access", "labels_mounted_during_training"):
        if field in value and not isinstance(value[field], bool):
            token = str(value[field]).casefold()
            if token not in {"true", "false"}:
                raise ValueError(f"invalid metric boolean: {field}")
            value[field] = token == "true"
    return value


def _governed_result_file(result: Path, relative: str) -> Path:
    if "\\" in relative:
        raise ValueError("non-portable M7F-H3 result path")
    path = result.joinpath(*Path(relative).parts)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(result)
    except ValueError as error:
        raise ValueError("M7F-H3 artifact escapes result root") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("M7F-H3 artifact is not a regular file")
    return resolved


def _load_representation(result: Path, engine_basis: Mapping[str, Any]) -> np.ndarray:
    path = _governed_result_file(result, str(engine_basis["representation_relative_path"]))
    if sha256_file(path) != engine_basis.get("representation_sha256"):
        raise ValueError("M7F-H3 representation hash differs from completion marker")
    values = np.load(path, allow_pickle=False)
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("M7F-H3 representation is not a finite matrix")
    return values


def _engine_marker(
    result: Path, row: Mapping[str, Any], logical_stage: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    complete = completed_marker(result, row, logical_stage)
    if complete is None:
        raise RuntimeError(f"M7F-H3 run is incomplete: {row['physical_run_id']} {logical_stage}")
    h3_marker = complete[1]
    h3_basis = h3_marker["basis"]
    engine_path = _governed_result_file(result, str(h3_basis["engine_marker_relative_path"]))
    if sha256_file(engine_path) != h3_basis.get("engine_marker_sha256"):
        raise ValueError("M7F-H3 engine marker hash changed")
    engine_marker = read_json(engine_path)
    engine_basis = validate_engine_marker(
        engine_marker,
        row,
        logical_stage=logical_stage,
        expected_attempt_index=int(h3_basis["attempt_index"]),
    )
    return h3_marker, h3_basis, engine_basis


def stage_registry(
    *,
    result_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    logical_stage: str,
) -> list[dict[str, Any]]:
    result = Path(result_root).resolve()
    plan = validate_plan(rows, seal, logical_stage=logical_stage)
    output: list[dict[str, Any]] = []
    for row in plan:
        h3_marker, h3_basis, engine = _engine_marker(result, row, logical_stage)
        output.append(
            {
                "logical_stage": logical_stage,
                "physical_run_id": row["physical_run_id"],
                "dataset": DATASET,
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "model_seed": int(row["model_seed"]),
                "attempt_index": int(engine["attempt_index"]),
                "continuation_from_update": int(engine["continuation_from_update"]),
                "endpoint_update": int(engine["endpoint_update"]),
                "optimizer_steps": int(engine["optimizer_steps"]),
                "checkpoint_relative_path": engine["checkpoint_relative_path"],
                "checkpoint_sha256": engine["checkpoint_sha256"],
                "representation_relative_path": engine["representation_relative_path"],
                "representation_sha256": engine["representation_sha256"],
                "model_total_seconds": float(engine["model_total_seconds"]),
                "peak_gpu_reserved_bytes": int(engine["peak_gpu_reserved_bytes"]),
                "gpu_uuid": h3_basis["gpu_uuid"],
                "gpu_pci_bus_id": h3_basis["gpu_pci_bus_id"],
                "runtime_domain": h3_basis["runtime_domain"],
                "worker_index": int(h3_basis["worker_index"]),
                "logical_role": row.get("logical_role", ""),
                "preferred_gpu_slot": row.get("preferred_gpu_slot", ""),
                "invocation_started_at": h3_basis["invocation_started_at"],
                "invocation_ended_at": h3_basis["invocation_ended_at"],
                "invocation_wall_seconds": float(h3_basis["invocation_wall_seconds"]),
                "parent_endpoint": int(h3_basis["parent_endpoint"]),
                "child_endpoint": int(h3_basis["child_endpoint"]),
                "incremental_optimizer_steps": int(h3_basis["incremental_optimizer_steps"]),
                "engineering_retry_of_attempt": h3_basis.get("engineering_retry_of_attempt", ""),
                "engineering_retry_replayed_from_zero": bool(
                    h3_basis.get("engineering_retry_replayed_from_zero", False)
                ),
                "plan_seal_identity": row["plan_seal_identity"],
                "completion_marker_identity": h3_marker["identity"],
                "labels_mounted_during_training": False,
                "downstream_evaluations_during_training": 0,
                "status": "PASS",
            }
        )
    return output


def close_canaries(
    *,
    result_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    registry = stage_registry(result_root=result, rows=rows, seal=seal, logical_stage="CANARY")
    if len(registry) != 4 or len({row["physical_run_id"] for row in registry}) != 4:
        raise RuntimeError("M7F-H3 canary matrix is not exactly four runs")
    audits = []
    for row in validate_plan(rows, seal, logical_stage="CANARY"):
        _h3, _h3_basis, engine = _engine_marker(result, row, "CANARY")
        audit = engine.get("canary_engineering_audit", {})
        if (
            audit.get("status") != "PASS"
            or audit.get("checkpoint_write_pass") is not True
            or audit.get("checkpoint_read_pass") is not True
            or audit.get("checkpoint_rng_roundtrip") is not True
            or audit.get("optimizer_step_pass") is not True
            or audit.get("labels_mounted") is not False
        ):
            raise RuntimeError("M7F-H3 engineering canary did not prove strict checkpoint restoration")
        audits.append(audit)
    basis = {
        "plan_seal_identity": seal["identity"],
        "run_count": 4,
        "registry_identity": canonical_json_hash(registry),
        "engineering_audit_identities": [row["identity"] for row in audits],
        "all_status_pass": True,
        "scientific_results_allowed": False,
        "formal_runs_restart_from_step_zero": True,
        "labels_mounted_during_training": False,
    }
    value = envelope("metachorus.v53.m7f-h3.canary-result.v1", "PASS", basis)
    write_csv_once(result / "M7F_H3_DBLP_CANARY_RUN_REGISTRY.csv", registry)
    write_json_once(result / "M7F_H3_DBLP_CANARY_RESULTS.json", value)
    return value


def evaluate_search_stage(
    *,
    result_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    logical_stage: str,
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if logical_stage not in {"R1", "R2", "R3", "R4"}:
        raise ValueError("M7F-H3 search evaluator requires R1..R4")
    result = Path(result_root).resolve()
    plan = validate_plan(rows, seal, logical_stage=logical_stage)
    search_mount = open_search_mount(
        role_manifest_path=role_manifest_path,
        role_seal_path=role_seal_path,
        private_mount_audit_path=private_mount_audit_path,
        private_search_mount_root=private_search_mount_root,
        dataset=DATASET,
    )
    metrics: list[dict[str, Any]] = []
    for row in plan:
        h3_marker, h3_basis, engine = _engine_marker(result, row, logical_stage)
        evaluated = evaluate_search_representation(
            _load_representation(result, engine), dataset=DATASET, search_mount=search_mount
        )
        if evaluated.get("status") != "PASS" or evaluated.get("prelim_locked_report_access") is not False:
            raise RuntimeError("M7F-H3 search evaluator crossed the locked-report boundary")
        metrics.append(
            {
                "logical_stage": logical_stage,
                "physical_run_id": row["physical_run_id"],
                "dataset": DATASET,
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "model_seed": int(row["model_seed"]),
                "endpoint_update": int(row["endpoint_update"]),
                "completion_marker_identity": h3_marker["identity"],
                "representation_sha256": engine["representation_sha256"],
                "gpu_uuid": h3_basis["gpu_uuid"],
                "model_total_seconds": float(engine["model_total_seconds"]),
                "peak_gpu_reserved_bytes": int(engine["peak_gpu_reserved_bytes"]),
                **evaluated["metrics"],
                "assignment_stability_mean": float(evaluated["assignment_stability_mean"]),
                "metric_role": "GLOBAL_SEARCH_POOL_ONLY",
                "prelim_locked_report_access": False,
                "labels_mounted_during_training": False,
                "status": "PASS",
            }
        )
    expected = len(plan)
    if len(metrics) != expected or len({row["physical_run_id"] for row in metrics}) != expected:
        raise RuntimeError("M7F-H3 search metric matrix is incomplete")
    metric_seal = envelope(
        "metachorus.v53.m7f-h3.search-metrics-seal.v1",
        "PASS",
        {
            "logical_stage": logical_stage,
            "dataset": DATASET,
            "endpoint_update": int(plan[0]["endpoint_update"]) if plan else SEARCH_ENDPOINTS[int(logical_stage[1]) - 1],
            "row_count": len(metrics),
            "rows_identity": canonical_json_hash(metrics),
            "plan_seal_identity": seal["identity"],
            "search_model_seeds": list(R1_MODEL_SEEDS if logical_stage == "R1" else SEARCH_MODEL_SEEDS),
            "metric_role": "GLOBAL_SEARCH_POOL_ONLY",
            "prelim_locked_report_access": False,
            "labels_mounted_during_training": False,
        },
    )
    write_csv_once(result / f"DBLP_H3_{logical_stage}_RUN_REGISTRY.csv", stage_registry(result_root=result, rows=plan, seal=seal, logical_stage=logical_stage))
    write_csv_once(result / f"DBLP_H3_{logical_stage}_METRICS.csv", metrics)
    write_json_once(result / f"DBLP_H3_{logical_stage}_METRICS_SEAL.json", metric_seal)
    return metrics, metric_seal


def _write_plan(result: Path, stage: str, rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]) -> None:
    write_csv_once(result / f"DBLP_H3_{stage}_RUN_PLAN.csv", list(rows))
    write_json_once(result / f"DBLP_H3_{stage}_RUN_PLAN_SEAL.json", dict(seal))


def advance_search(
    *,
    result_root: str | Path,
    cache: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    logical_stage: str,
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    metrics, metric_seal = evaluate_search_stage(
        result_root=result,
        rows=rows,
        seal=seal,
        logical_stage=logical_stage,
        role_manifest_path=role_manifest_path,
        role_seal_path=role_seal_path,
        private_mount_audit_path=private_mount_audit_path,
        private_search_mount_root=private_search_mount_root,
    )
    plan = validate_plan(rows, seal, logical_stage=logical_stage)
    candidate_by_hash = {
        str(candidate["candidate_set_hash"]): dict(candidate)
        for candidate in cache["all_sets"]["sets"]
    }
    current_auto = {"status": CURRENT_AUTOMATIC_STATUS}
    if logical_stage == "R1":
        pool = derive_r1_pool(metrics, list(candidate_by_hash.values()))
        selected_rows = list(pool["selected"])
        if len(selected_rows) != R1_POOL_SIZE:
            raise RuntimeError("M7F-H3 R1 pool is not exactly 24 sets")
        selection_basis = {
            **dict(pool["seal"]["basis"]),
            "r1_metrics_seal_identity": metric_seal["identity"],
            "current_automatic_status": CURRENT_AUTOMATIC_STATUS,
            "sealed_before_r2_training": True,
        }
        selection_seal = envelope(
            "metachorus.v53.m7f-h3.r1-pool-seal.v1",
            "SEALED_AFTER_R1_BEFORE_R2",
            selection_basis,
        )
        write_csv_once(result / "DBLP_H3_R1_POOL_COMPONENTS.csv", list(pool["components"]))
        write_csv_once(result / "DBLP_H3_R1_POOL.csv", selected_rows)
        write_json_once(result / "DBLP_H3_R1_POOL_SEAL.json", selection_seal)
        next_rows, next_seal = build_continuation_plan(
            prior_rows=plan,
            selected_candidate_hashes={str(row["candidate_set_hash"]) for row in selected_rows},
            result_root=result,
            next_endpoint=1000,
            prior_seal_identity=seal["identity"],
            selection_seal=selection_seal,
            cache=cache,
        )
        _write_plan(result, "R2", next_rows, next_seal)
        return {"status": "PASS", "next_stage": "R2", "selected_set_count": len(selected_rows), "next_row_count": len(next_rows)}
    if logical_stage == "R2":
        selected_hashes = {str(row["candidate_set_hash"]) for row in plan}
        no_pruning_seal = envelope(
            "metachorus.v53.m7f-h3.r2-no-pruning-seal.v1",
            "SEALED_AFTER_R2_BEFORE_R3",
            {
                "dataset": DATASET,
                "r2_metrics_seal_identity": metric_seal["identity"],
                "selected_candidate_hashes": sorted(selected_hashes),
                "selected_candidate_set_count": len(selected_hashes),
                "r2_pruning_applied": False,
                "sealed_before_r3_training": True,
            },
        )
        write_json_once(result / "DBLP_H3_R2_NO_PRUNING_SEAL.json", no_pruning_seal)
        next_rows, next_seal = build_continuation_plan(
            prior_rows=plan,
            selected_candidate_hashes=selected_hashes,
            result_root=result,
            next_endpoint=2000,
            prior_seal_identity=seal["identity"],
            selection_seal=no_pruning_seal,
            cache=cache,
        )
        _write_plan(result, "R3", next_rows, next_seal)
        return {"status": "PASS", "next_stage": "R3", "selected_set_count": len(selected_hashes), "next_row_count": len(next_rows)}
    if logical_stage == "R3":
        candidate_rows = [candidate_by_hash[digest] for digest in sorted({str(row["candidate_set_hash"]) for row in plan})]
        shortlist = derive_r3_shortlist(metrics, candidate_rows)
        shortlist_rows = list(shortlist["shortlist"])
        if len(shortlist_rows) != R3_SHORTLIST_SIZE:
            raise RuntimeError("M7F-H3 R3 shortlist is not exactly eight sets")
        shortlist_basis = {
            **dict(shortlist["seal"]["basis"]),
            "r3_metrics_seal_identity": metric_seal["identity"],
            "sealed_before_r4_training": True,
        }
        shortlist_seal = envelope(
            "metachorus.v53.m7f-h3.r3-shortlist-seal.v1",
            "SEALED_AFTER_R3_BEFORE_R4",
            shortlist_basis,
        )
        write_csv_once(result / "DBLP_H3_R3_OBJECTIVES.csv", list(shortlist["objectives"]))
        write_csv_once(result / "DBLP_H3_R3_SHORTLIST_COMPONENTS.csv", list(shortlist["components"]))
        write_csv_once(result / "DBLP_H3_R3_SHORTLIST.csv", shortlist_rows)
        write_json_once(result / "DBLP_H3_R3_SHORTLIST_SEAL.json", shortlist_seal)
        next_rows, next_seal = build_continuation_plan(
            prior_rows=plan,
            selected_candidate_hashes={str(row["candidate_set_hash"]) for row in shortlist_rows},
            result_root=result,
            next_endpoint=3000,
            prior_seal_identity=seal["identity"],
            selection_seal=shortlist_seal,
            cache=cache,
        )
        _write_plan(result, "R4", next_rows, next_seal)
        return {"status": "PASS", "next_stage": "R4", "shortlist_set_count": len(shortlist_rows), "next_row_count": len(next_rows)}
    if logical_stage != "R4":
        raise ValueError("unsupported M7F-H3 search transition")
    candidate_rows = [candidate_by_hash[digest] for digest in sorted({str(row["candidate_set_hash"]) for row in plan})]
    closed = close_search_selection(
        dataset=DATASET,
        metric_rows=metrics,
        candidate_rows=candidate_rows,
        manual_set=cache["manual"],
        current_automatic=current_auto,
    )
    basis = closed["basis"]
    selected = dict(basis["selected_set"])
    selection_status = (
        "DBLP_BUDGETED_SELECTED_MANUAL"
        if closed["status"] == "MANUAL_SELECTED"
        else "DBLP_BUDGETED_NEW_SET_SELECTED"
    )
    safety = list(basis["manual_safety_rows"])
    robust = list(basis["objective_rows"])
    first_front_hashes = {
        str(row["candidate_set_hash"])
        for row in first_pareto_front(
            [row for row in robust if row.get("manual_safety_status") == "PASS"]
        )
    }
    robust = [
        {**row, "on_first_robust_pareto_front": str(row["candidate_set_hash"]) in first_front_hashes}
        for row in robust
    ]
    selection_basis = {
        "dataset": DATASET,
        "selection_status": selection_status,
        "selected_set": selected,
        "manual_set": dict(cache["manual"]),
        "r4_metrics_seal_identity": metric_seal["identity"],
        "selection_identity": basis["selection_identity"],
        "safe_set_count": int(basis["safe_set_count"]),
        "first_robust_pareto_front_count": len(first_front_hashes),
        "no_second_choice": True,
        "algorithm_retuning": False,
        "evaluation_seeds_started": False,
        "prelim_locked_report_access": False,
    }
    selection_seal = envelope(
        "metachorus.v53.m7f-h3.selection-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        selection_basis,
    )
    write_csv_once(result / "DBLP_H3_R4_MANUAL_SAFETY.csv", safety)
    write_csv_once(result / "DBLP_H3_R4_PARETO.csv", robust)
    write_json_once(result / "DBLP_H3_SELECTED_SET.json", envelope("metachorus.v53.m7f-h3.selected-set.v1", selection_status, {"selected_set": selected, "selection_seal_identity": selection_seal["identity"]}))
    write_json_once(result / "DBLP_H3_SELECTION_SEAL.json", selection_seal)
    write_text_once(
        result / "DBLP_H3_SELECTION_DECISION.md",
        "# DBLP H3 Budgeted Frozen Selection\n\n"
        f"- status: `{selection_status}`\n"
        f"- selected_set_id: `{selected['candidate_set_id']}`\n"
        f"- selected_set_hash: `{selected['candidate_set_hash']}`\n"
        f"- path_ids: `{json.dumps(selected['path_ids'], ensure_ascii=False)}`\n"
        "- second_choice_used: `false`\n"
        "- algorithm_retuning: `false`\n",
    )
    prelim_rows, prelim_seal = build_preliminary_plan(cache, selected)
    _write_plan(result, "PRELIMINARY", prelim_rows, prelim_seal)
    return {
        "status": selection_status,
        "selected_set": selected,
        "selection_seal_identity": selection_seal["identity"],
        "preliminary_row_count": len(prelim_rows),
    }


def seal_preliminary_training_registry(
    *,
    result_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    selection_seal: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve()
    selection_basis = selection_seal.get("basis", {})
    if (
        selection_seal.get("status") != "SEALED_BEFORE_PRELIMINARY_EVALUATION"
        or selection_seal.get("identity") != canonical_json_hash(selection_basis)
        or selection_basis.get("evaluation_seeds_started") is not False
        or selection_basis.get("no_second_choice") is not True
    ):
        raise ValueError("M7F-H3 preliminary cannot start before the frozen selection seal")
    plan = validate_plan(rows, seal, logical_stage="PRELIMINARY")
    registry = stage_registry(result_root=result, rows=plan, seal=seal, logical_stage="PRELIMINARY") if plan else []
    selected_is_manual = selection_basis["selected_set"]["candidate_set_hash"] == MANUAL_SET_HASH
    if (selected_is_manual and len(registry) != 3) or (not selected_is_manual and len(registry) != 6):
        raise RuntimeError("M7F-H3 preliminary physical deduplication matrix is invalid")
    if selected_is_manual:
        if (
            {str(row.get("logical_role")) for row in registry} != {"MANUAL"}
            or {int(row["model_seed"]) for row in registry} != set(EVALUATION_MODEL_SEEDS)
        ):
            raise RuntimeError("M7F-H3 deduplicated Manual preliminary registry is invalid")
    else:
        if (
            {str(row.get("logical_role")) for row in registry} != {"BUDGETED_SELECTED_SET", "MANUAL"}
            or {int(row["model_seed"]) for row in registry} != set(EVALUATION_MODEL_SEEDS)
        ):
            raise RuntimeError("M7F-H3 preliminary registry is not the paired two-arm matrix")
        for seed in EVALUATION_MODEL_SEEDS:
            pair = [row for row in registry if int(row["model_seed"]) == int(seed)]
            if (
                len(pair) != 2
                or {str(row["logical_role"]) for row in pair} != {"BUDGETED_SELECTED_SET", "MANUAL"}
                or len({str(row["gpu_uuid"]) for row in pair}) != 1
            ):
                raise RuntimeError("M7F-H3 selected/Manual pair did not run on the same GPU UUID")
    if any(
        int(row["continuation_from_update"]) != 0
        or int(row["endpoint_update"]) != PRELIM_ENDPOINT
        or int(row["optimizer_steps"]) != PRELIM_ENDPOINT
        for row in registry
    ):
        raise RuntimeError("M7F-H3 preliminary runs must be fresh step0-to-3000 identities")
    write_csv_once(
        result / "DBLP_H3_PRELIM_TRAINING_REGISTRY.csv",
        registry,
    )
    registry_basis = {
        "selection_seal_identity": selection_seal["identity"],
        "preliminary_plan_seal_identity": seal["identity"],
        "active_datasets": [DATASET],
        "physical_run_count": len(registry),
        "registry_identity": canonical_json_hash(registry),
        "selected_is_manual": selected_is_manual,
        "manual_physical_run_deduplicated": selected_is_manual,
        "all_unique_sets_trained_fresh_from_zero": True,
        "same_gpu_uuid_within_evaluation_seed": True,
        "all_status_pass": True,
        "labels_mounted": False,
        "preliminary_evaluation_started": False,
    }
    registry_seal = envelope(
        "metachorus.v53.m7f-h3.preliminary-training-registry-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        registry_basis,
    )
    write_json_once(result / "DBLP_H3_PRELIM_TRAINING_REGISTRY_SEAL.json", registry_seal)
    return registry, registry_seal


def _summary_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for role in ("BUDGETED_SELECTED_SET", "MANUAL"):
        cohort = [row for row in metrics if row["logical_role"] == role]
        if len(cohort) != 3 or {int(row["model_seed"]) for row in cohort} != set(EVALUATION_MODEL_SEEDS):
            raise ValueError("M7F-H3 preliminary summary requires paired three-seed arms")
        output.append(
            {
                "dataset": DATASET,
                "logical_role": role,
                "candidate_set_id": cohort[0]["candidate_set_id"],
                "candidate_set_hash": cohort[0]["candidate_set_hash"],
                "model_seed_count": 3,
                **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in cohort) for metric in METRICS},
                **{f"sample_sd_{metric}": statistics.stdev(float(row[metric]) for row in cohort) for metric in METRICS},
                "evidence_level": "3_SEED_PRELIMINARY_NONCONFIRMATORY",
                "confirmatory_claim_allowed": False,
                "status": "PASS",
            }
        )
    return output


def _update_five_dataset_table(
    *,
    corrected_table_path: str | Path,
    selected_summary: Mapping[str, Any],
    selection_status: str,
    decision_status: str,
    final_method_role: str,
    result: Path,
) -> list[dict[str, Any]]:
    rows = read_csv(corrected_table_path)
    if (
        len(rows) != 5
        or {str(row.get("dataset", "")).casefold() for row in rows}
        != {"acm", "freebase", "imdb", "dblp", "aminer"}
    ):
        raise ValueError("M7F-H3 corrected five-dataset table is incomplete")
    output: list[dict[str, Any]] = []
    for row in rows:
        value: dict[str, Any] = dict(row)
        dataset = str(value["dataset"]).casefold()
        value.setdefault("execution_status", value.get("status", ""))
        value.setdefault("selection_status", "PARENT_FROZEN")
        value.setdefault("preliminary_decision", "PARENT_FROZEN")
        if dataset == DATASET:
            value.update(
                {
                    "method_role": final_method_role,
                    **{metric: selected_summary[f"mean_{metric}"] for metric in METRICS},
                    "model_seed_count": 3,
                    "evidence_level": "3_SEED_PRELIMINARY_NONCONFIRMATORY",
                    "confirmatory_claim_allowed": False,
                    "status": decision_status,
                    "execution_status": "DBLP_H3_FULL_SEARCH_COMPLETE",
                    "selection_status": selection_status,
                    "preliminary_decision": decision_status,
                }
            )
        output.append(value)
    write_csv_once(result / "FIVE_DATASET_STATUS_TABLE_AFTER_DBLP.csv", output)
    lines = [
        "# Five-Dataset Status After DBLP H3",
        "",
        "| Dataset | Execution | Selection | Preliminary | Evidence | Confirmatory |",
        "|---|---|---|---|---|---|",
    ]
    for row in output:
        lines.append(
            f"| {row['dataset']} | {row['execution_status']} | {row['selection_status']} | "
            f"{row['preliminary_decision']} | {row['evidence_level']} | {str(row['confirmatory_claim_allowed']).lower()} |"
        )
    write_text_once(result / "FIVE_DATASET_STATUS_TABLE_AFTER_DBLP.md", "\n".join(lines) + "\n")
    return output


def evaluate_preliminary(
    *,
    result_root: str | Path,
    corrected_five_dataset_table_path: str | Path,
    role_manifest_path: str | Path,
    private_search_mount_root: str | Path,
    label_source: str | Path,
    label_source_root: str | Path,
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    selection_seal = read_json(result / "DBLP_H3_SELECTION_SEAL.json")
    prelim_rows = read_csv(result / "DBLP_H3_PRELIMINARY_RUN_PLAN.csv")
    prelim_seal = read_json(result / "DBLP_H3_PRELIMINARY_RUN_PLAN_SEAL.json")
    registry, registry_seal = seal_preliminary_training_registry(
        result_root=result,
        rows=prelim_rows,
        seal=prelim_seal,
        selection_seal=selection_seal,
    )
    selected = selection_seal["basis"]["selected_set"]
    selected_is_manual = selected["candidate_set_hash"] == MANUAL_SET_HASH
    session = PreliminaryLabelSession(
            role_manifest_path=role_manifest_path,
            private_search_mount_root=private_search_mount_root,
            label_sources={DATASET: label_source},
            label_source_roots={DATASET: label_source_root},
            training_registry_seal=registry_seal,
    )
    roles = session.open_dataset(DATASET)
    plan_by_id = {row["physical_run_id"]: canonical_plan_row(row) for row in prelim_rows}
    evaluated_rows: list[dict[str, Any]] = []
    for registry_row in registry:
        row = plan_by_id[registry_row["physical_run_id"]]
        _h3, _h3_basis, engine = _engine_marker(result, row, "PRELIMINARY")
        evaluated = evaluate_preliminary_representation(
            _load_representation(result, engine), dataset=DATASET, roles=roles
        )
        evaluated_rows.append(
            {
                "logical_role": row["logical_role"],
                "physical_run_id": row["physical_run_id"],
                "dataset": DATASET,
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "model_seed": int(row["model_seed"]),
                "endpoint_update": PRELIM_ENDPOINT,
                **evaluated["metrics"],
                "classification_fit_role": evaluated["classification_fit_role"],
                "classification_metric_role": evaluated["classification_metric_role"],
                "clustering_fit_role": evaluated["clustering_fit_role"],
                "clustering_metric_role": evaluated["clustering_metric_role"],
                "assignment_stability_mean": evaluated["assignment_stability_mean"],
                "gpu_uuid": registry_row["gpu_uuid"],
                "manual_reused": False,
                "physical_run_deduplicated_with_manual": selected_is_manual,
                "confirmatory_claim_allowed": False,
                "status": "PASS",
            }
        )
    manual_metrics = [row for row in evaluated_rows if row["logical_role"] == "MANUAL"]
    selected_metrics = (
        [
            {
                **row,
                "logical_role": "BUDGETED_SELECTED_SET",
                "physical_run_deduplicated_with_manual": True,
            }
            for row in manual_metrics
        ]
        if selected_is_manual
        else [row for row in evaluated_rows if row["logical_role"] == "BUDGETED_SELECTED_SET"]
    )
    access_count = session.access_count
    if access_count != 1 or session.per_dataset_access_count[DATASET] != 1:
        raise RuntimeError("M7F-H3 preliminary label source must be opened exactly once")
    all_metrics = selected_metrics + manual_metrics
    summaries = _summary_rows(all_metrics)
    selected_by_seed = {int(row["model_seed"]): row for row in selected_metrics}
    manual_by_seed = {int(row["model_seed"]): row for row in manual_metrics}
    comparisons = []
    for seed in EVALUATION_MODEL_SEEDS:
        selected_gpu_uuid = str(selected_by_seed[seed].get("gpu_uuid", "PARENT_NOT_RECORDED"))
        manual_gpu_uuid = str(manual_by_seed[seed].get("gpu_uuid", "PARENT_NOT_RECORDED"))
        same_gpu = selected_is_manual or (selected_gpu_uuid == manual_gpu_uuid and selected_gpu_uuid not in {"", "PARENT_NOT_RECORDED"})
        if not same_gpu:
            raise RuntimeError("M7F-H3 final selected/Manual comparison is not GPU-paired")
        comparisons.append(
            {
                "dataset": DATASET,
                "model_seed": int(seed),
                "selected_set_id": selected["candidate_set_id"],
                "selected_set_hash": selected["candidate_set_hash"],
                "manual_set_id": MANUAL_SET_ID,
                "manual_set_hash": MANUAL_SET_HASH,
                **{f"delta_{metric}": float(selected_by_seed[seed][metric]) - float(manual_by_seed[seed][metric]) for metric in METRICS},
                "paired_by_same_evaluation_seed": True,
                "selected_gpu_uuid": selected_gpu_uuid,
                "manual_gpu_uuid": manual_gpu_uuid,
                "same_gpu_uuid": same_gpu,
                "status": "PASS",
            }
        )
    raw_decision = preliminary_decision(
        [{**row, "logical_role": "FINAL_FROZEN_METHOD"} for row in selected_metrics],
        manual_metrics,
    )
    if selected_is_manual:
        decision_status = "DBLP_BUDGETED_SELECTED_MANUAL"
    else:
        decision_status = {
            "PROMISING": "DBLP_BUDGETED_PRELIMINARY_PROMISING",
            "NONINFERIOR": "DBLP_BUDGETED_PRELIMINARY_NONINFERIOR",
            "UNSAFE": "DBLP_BUDGETED_PRELIMINARY_UNSAFE",
            "INDETERMINATE": "DBLP_BUDGETED_PRELIMINARY_UNSAFE",
        }[raw_decision["status"]]
    selection_status = selection_seal["basis"]["selection_status"]
    unsafe = decision_status == "DBLP_BUDGETED_PRELIMINARY_UNSAFE"
    final_set = (
        {
            "candidate_set_id": MANUAL_SET_ID,
            "candidate_set_hash": MANUAL_SET_HASH,
            "candidate_set_ordinal": next(
                int(row["candidate_set_ordinal"])
                for row in prelim_rows
                if str(row["candidate_set_hash"]) == MANUAL_SET_HASH
            ),
            "path_ids": next(
                list(row["path_ids"])
                for row in prelim_rows
                if str(row["candidate_set_hash"]) == MANUAL_SET_HASH
            ),
        }
        if unsafe or selected_is_manual
        else dict(selected)
    )
    final_method_role = "MANUAL" if final_set["candidate_set_hash"] == MANUAL_SET_HASH else "BUDGETED_SELECTED_SET"
    decision_basis = {
        "dataset": DATASET,
        "selection_status": selection_status,
        "search_selected_set": selected,
        "final_set": final_set,
        "final_method_role": final_method_role,
        "manual_set_id": MANUAL_SET_ID,
        "manual_set_hash": MANUAL_SET_HASH,
        "preliminary_decision": decision_status,
        "raw_frozen_decision": raw_decision,
        "selected_is_manual": selected_is_manual,
        "manual_physical_run_deduplicated": selected_is_manual,
        "unsafe_search_selection_fell_back_to_manual": unsafe,
        "selected_metric_rows_identity": canonical_json_hash(selected_metrics),
        "manual_metric_rows_identity": canonical_json_hash(manual_metrics),
        "locked_label_source_access_count": access_count,
        "no_second_choice": True,
        "search_reopened": False,
        "ten_seed_expansion_started": False,
        "confirmatory_claim_allowed": False,
    }
    decision = envelope("metachorus.v53.m7f-h3.final-decision.v1", decision_status, decision_basis)
    write_csv_once(result / "DBLP_H3_PRELIMINARY_METRICS_BY_SEED.csv", all_metrics)
    write_csv_once(result / "DBLP_H3_PRELIMINARY_SUMMARY.csv", summaries)
    write_csv_once(result / "DBLP_H3_SELECTED_VS_MANUAL.csv", comparisons)
    write_json_once(result / "DBLP_H3_FINAL_DECISION.json", decision)
    write_text_once(
        result / "DBLP_H3_FINAL_DECISION.md",
        "# DBLP H3 Three-Seed Preliminary Decision\n\n"
        f"- decision: `{decision_status}`\n"
        f"- selection_status: `{selection_status}`\n"
        f"- selected_set_id: `{selected['candidate_set_id']}`\n"
        f"- mean_delta_F_cls: `{raw_decision['mean_delta_F_cls']}`\n"
        f"- mean_delta_F_clu: `{raw_decision['mean_delta_F_clu']}`\n"
        f"- mean_delta_F_joint: `{raw_decision['mean_delta_F_joint']}`\n"
        f"- joint_non_worse_count: `{raw_decision['joint_non_worse_count']}/3`\n"
        "- ten_seed_expansion_started: `false`\n"
        "- second_choice_used: `false`\n",
    )
    write_csv_once(
        result / "DBLP_H3_FINAL_SET_REGISTRY.csv",
        [
            {
                "dataset": DATASET,
                "candidate_set_id": final_set["candidate_set_id"],
                "candidate_set_hash": final_set["candidate_set_hash"],
                "candidate_set_ordinal": int(final_set["candidate_set_ordinal"]),
                "path_ids": list(final_set["path_ids"]),
                "final_method_role": final_method_role,
                "search_selected_set_id": selected["candidate_set_id"],
                "search_selected_set_hash": selected["candidate_set_hash"],
                "selection_status": selection_status,
                "preliminary_decision": decision_status,
                "selection_seal_identity": selection_seal["identity"],
                "final_decision_identity": decision["identity"],
                "no_second_choice": True,
                "status": "PASS",
            }
        ],
    )
    selected_summary = next(
        row
        for row in summaries
        if row["logical_role"] == ("MANUAL" if final_method_role == "MANUAL" else "BUDGETED_SELECTED_SET")
    )
    _update_five_dataset_table(
        corrected_table_path=corrected_five_dataset_table_path,
        selected_summary=selected_summary,
        selection_status=selection_status,
        decision_status=decision_status,
        final_method_role=final_method_role,
        result=result,
    )
    expansion_pending = decision_status in {
        "DBLP_BUDGETED_PRELIMINARY_PROMISING",
        "DBLP_BUDGETED_PRELIMINARY_NONINFERIOR",
    }
    write_text_once(
        result / "DBLP_10SEED_CONFIRMATION_PLAN_PENDING.md",
        "# DBLP 10-Seed Confirmation Pending\n\n"
        f"- approval_status: `{'PENDING_USER' if expansion_pending else 'NOT_AUTHORIZED_BY_THREE_SEED_DECISION'}`\n"
        "- started: `false`\n"
        "- frozen_set_only: `true`\n",
    )
    write_csv_once(
        result / "DBLP_10SEED_CONFIRMATION_MATRIX_PENDING.csv",
        [
            {
                "dataset": DATASET,
                "selected_set_id": selected["candidate_set_id"],
                "selected_set_hash": selected["candidate_set_hash"],
                "status": "PENDING_USER" if expansion_pending else "NOT_AUTHORIZED",
                "training_started": False,
            }
        ],
    )
    approval = envelope(
        "metachorus.v53.m7f-h3.ten-seed-approval.v1",
        "PENDING_USER" if expansion_pending else "NOT_AUTHORIZED",
        {
            "preliminary_decision": decision_status,
            "final_decision_identity": decision["identity"],
            "ten_seed_expansion_started": False,
            "user_approval_received": False,
        },
    )
    write_json_once(result / "DBLP_10SEED_APPROVAL_STATUS.json", approval)
    write_json_once(
        result / "DBLP_H3_PRELIMINARY_ACCESS_AUDIT.json",
        envelope(
            "metachorus.v53.m7f-h3.preliminary-access-audit.v1",
            "PASS",
            {
                "training_registry_seal_identity": registry_seal["identity"],
                "label_source_access_count": access_count,
                "manual_physical_run_deduplicated": selected_is_manual,
                "all_unique_sets_evaluated_in_one_label_session": True,
                "raw_labels_persisted": False,
                "labels_mounted_during_training": False,
            },
        ),
    )
    return decision


__all__ = [
    "canonical_metric_row",
    "stage_registry",
    "close_canaries",
    "evaluate_search_stage",
    "advance_search",
    "seal_preliminary_training_registry",
    "evaluate_preliminary",
]
