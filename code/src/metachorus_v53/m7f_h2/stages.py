"""Stage closure, frozen selection, and preliminary evaluation for M7F-H2."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import zipfile
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
from metachorus_v53.m6f_m7f_fast.protocol import (
    METRICS,
    close_search_selection,
    derive_r1_high_recall_pool,
    derive_robust_shortlist,
    first_pareto_front,
    preliminary_decision,
)
from metachorus_v53.m6f_m7f_fast.training import validate_run_marker

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
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    PATH_LIBRARY_IDENTITY,
    PRELIM_ENDPOINT,
    SEARCH_ENDPOINTS,
    SEARCH_MODEL_SEEDS,
)
from .planning import (
    build_continuation_plan,
    build_preliminary_plan,
    canonical_plan_row,
    validate_plan,
)
from .training import ENGINE_STAGE, completed_marker


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
        raise ValueError("non-portable H2 result path")
    path = result.joinpath(*Path(relative).parts)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(result)
    except ValueError as error:
        raise ValueError("H2 artifact escapes result root") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("H2 artifact is not a regular file")
    return resolved


def _load_representation(result: Path, engine_basis: Mapping[str, Any]) -> np.ndarray:
    path = _governed_result_file(result, str(engine_basis["representation_relative_path"]))
    if sha256_file(path) != engine_basis.get("representation_sha256"):
        raise ValueError("H2 representation hash differs from completion marker")
    values = np.load(path, allow_pickle=False)
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("H2 representation is not a finite matrix")
    return values


def _engine_marker(
    result: Path, row: Mapping[str, Any], logical_stage: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    complete = completed_marker(result, row, logical_stage)
    if complete is None:
        raise RuntimeError(f"H2 run is incomplete: {row['physical_run_id']} {logical_stage}")
    h2_marker = complete[1]
    h2_basis = h2_marker["basis"]
    engine_path = _governed_result_file(result, str(h2_basis["engine_marker_relative_path"]))
    if sha256_file(engine_path) != h2_basis.get("engine_marker_sha256"):
        raise ValueError("H2 engine marker hash changed")
    engine_marker = read_json(engine_path)
    engine_basis = validate_run_marker(
        engine_marker,
        row,
        stage=ENGINE_STAGE[logical_stage],
        endpoint=int(row["endpoint_update"]),
    )
    return h2_marker, h2_basis, engine_basis


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
        h2_marker, h2_basis, engine = _engine_marker(result, row, logical_stage)
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
                "gpu_uuid": h2_basis["gpu_uuid"],
                "gpu_pci_bus_id": h2_basis["gpu_pci_bus_id"],
                "runtime_domain": h2_basis["runtime_domain"],
                "worker_index": int(h2_basis["worker_index"]),
                "logical_role": row.get("logical_role", ""),
                "preferred_gpu_slot": row.get("preferred_gpu_slot", ""),
                "started_at": h2_basis["started_at"],
                "ended_at": h2_basis["ended_at"],
                "plan_seal_identity": row["plan_seal_identity"],
                "completion_marker_identity": h2_marker["identity"],
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
        raise RuntimeError("H2 canary matrix is not exactly four runs")
    audits = []
    for row in validate_plan(rows, seal, logical_stage="CANARY"):
        _h2, _h2_basis, engine = _engine_marker(result, row, "CANARY")
        audit = engine.get("canary_engineering_audit", {})
        if (
            audit.get("status") != "PASS"
            or audit.get("checkpoint_write_pass") is not True
            or audit.get("checkpoint_read_pass") is not True
            or audit.get("checkpoint_rng_roundtrip") is not True
            or audit.get("optimizer_step_pass") is not True
            or audit.get("labels_mounted") is not False
        ):
            raise RuntimeError("H2 engineering canary did not prove strict checkpoint restoration")
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
    value = envelope("metachorus.v53.m7f-h2.canary-result.v1", "PASS", basis)
    write_csv_once(result / "AMINER_H2_CANARY_RUN_REGISTRY.csv", registry)
    write_json_once(result / "AMINER_H2_CANARY_RESULT.json", value)
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
        raise ValueError("H2 search evaluator requires R1..R4")
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
        h2_marker, h2_basis, engine = _engine_marker(result, row, logical_stage)
        evaluated = evaluate_search_representation(
            _load_representation(result, engine), dataset=DATASET, search_mount=search_mount
        )
        if evaluated.get("status") != "PASS" or evaluated.get("prelim_locked_report_access") is not False:
            raise RuntimeError("H2 search evaluator crossed the locked-report boundary")
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
                "completion_marker_identity": h2_marker["identity"],
                "representation_sha256": engine["representation_sha256"],
                "gpu_uuid": h2_basis["gpu_uuid"],
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
        raise RuntimeError("H2 search metric matrix is incomplete")
    metric_seal = envelope(
        "metachorus.v53.m7f-h2.search-metrics-seal.v1",
        "PASS",
        {
            "logical_stage": logical_stage,
            "dataset": DATASET,
            "endpoint_update": int(plan[0]["endpoint_update"]) if plan else SEARCH_ENDPOINTS[int(logical_stage[1]) - 1],
            "row_count": len(metrics),
            "rows_identity": canonical_json_hash(metrics),
            "plan_seal_identity": seal["identity"],
            "search_model_seeds": list(SEARCH_MODEL_SEEDS),
            "metric_role": "GLOBAL_SEARCH_POOL_ONLY",
            "prelim_locked_report_access": False,
            "labels_mounted_during_training": False,
        },
    )
    write_csv_once(result / f"AMINER_H2_{logical_stage}_RUN_REGISTRY.csv", stage_registry(result_root=result, rows=plan, seal=seal, logical_stage=logical_stage))
    write_csv_once(result / f"AMINER_H2_{logical_stage}_METRICS_BY_SEED.csv", metrics)
    write_json_once(result / f"AMINER_H2_{logical_stage}_METRICS_SEAL.json", metric_seal)
    if logical_stage == "R1":
        summary = []
        for digest in sorted({row["candidate_set_hash"] for row in metrics}):
            cohort = [row for row in metrics if row["candidate_set_hash"] == digest]
            if len(cohort) != 3:
                raise RuntimeError("H2 R1 summary lacks three seeds")
            summary.append(
                {
                    "dataset": DATASET,
                    "candidate_set_id": cohort[0]["candidate_set_id"],
                    "candidate_set_hash": digest,
                    "candidate_set_ordinal": int(cohort[0]["candidate_set_ordinal"]),
                    "model_seed_count": 3,
                    **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in cohort) for metric in METRICS},
                    **{f"sample_sd_{metric}": statistics.stdev(float(row[metric]) for row in cohort) for metric in METRICS},
                    "status": "PASS",
                }
            )
        write_csv_once(result / "AMINER_H2_R1_SUMMARY.csv", summary)
    return metrics, metric_seal


def _write_plan(result: Path, stage: str, rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]) -> None:
    write_csv_once(result / f"AMINER_H2_{stage}_RUN_PLAN.csv", list(rows))
    write_json_once(result / f"AMINER_H2_{stage}_RUN_PLAN_SEAL.json", dict(seal))


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
        pool = derive_r1_high_recall_pool(
            metrics,
            list(candidate_by_hash.values()),
            manual_candidate_set_hash=MANUAL_SET_HASH,
            current_automatic_candidate_set_hash=None,
        )
        selected_rows = list(pool["selected"])
        if len(selected_rows) > 57:
            raise RuntimeError("H2 frozen high-recall implementation exceeded 57 unique sets")
        selection_seal = envelope(
            "metachorus.v53.m7f-h2.r1-selection-seal.v1",
            "SEALED_AFTER_R1_BEFORE_R2",
            {
                "dataset": DATASET,
                "r1_metrics_seal_identity": metric_seal["identity"],
                "selected_row_count": len(selected_rows),
                "selected_rows_identity": canonical_json_hash(selected_rows),
                "selection_identity": pool["basis"]["selection_identity"],
                "selected_candidate_hashes": sorted(str(row["candidate_set_hash"]) for row in selected_rows),
                "selected_candidate_set_count": len(selected_rows),
                "manual_included": any(row["candidate_set_hash"] == MANUAL_SET_HASH for row in selected_rows),
                "current_automatic_status": CURRENT_AUTOMATIC_STATUS,
                "sealed_before_r2_training": True,
            },
        )
        write_csv_once(result / "AMINER_H2_R1_HIGH_RECALL_POOL.csv", selected_rows)
        write_json_once(result / "AMINER_H2_R1_SELECTION_SEAL.json", selection_seal)
        next_rows, next_seal = build_continuation_plan(
            prior_rows=plan,
            selected_candidate_hashes={str(row["candidate_set_hash"]) for row in selected_rows},
            result_root=result,
            next_endpoint=1000,
            prior_seal_identity=seal["identity"],
            selection_seal=selection_seal,
        )
        _write_plan(result, "R2", next_rows, next_seal)
        return {"status": "PASS", "next_stage": "R2", "selected_set_count": len(selected_rows), "next_row_count": len(next_rows)}
    if logical_stage == "R2":
        selected_hashes = {str(row["candidate_set_hash"]) for row in plan}
        no_pruning_seal = envelope(
            "metachorus.v53.m7f-h2.r2-no-pruning-seal.v1",
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
        write_json_once(result / "AMINER_H2_R2_NO_PRUNING_SEAL.json", no_pruning_seal)
        next_rows, next_seal = build_continuation_plan(
            prior_rows=plan,
            selected_candidate_hashes=selected_hashes,
            result_root=result,
            next_endpoint=2000,
            prior_seal_identity=seal["identity"],
            selection_seal=no_pruning_seal,
        )
        _write_plan(result, "R3", next_rows, next_seal)
        return {"status": "PASS", "next_stage": "R3", "selected_set_count": len(selected_hashes), "next_row_count": len(next_rows)}
    if logical_stage == "R3":
        candidate_rows = [candidate_by_hash[digest] for digest in sorted({str(row["candidate_set_hash"]) for row in plan})]
        shortlist = derive_robust_shortlist(
            metrics,
            candidate_rows,
            manual_candidate_set_hash=MANUAL_SET_HASH,
            current_automatic_candidate_set_hash=None,
        )
        shortlist_rows = list(shortlist["shortlist"])
        shortlist_seal = envelope(
            "metachorus.v53.m7f-h2.r3-shortlist-seal.v1",
            "SEALED_AFTER_R3_BEFORE_R4",
            {
                "dataset": DATASET,
                "r3_metrics_seal_identity": metric_seal["identity"],
                "shortlist_row_count": len(shortlist_rows),
                "shortlist_rows_identity": canonical_json_hash(shortlist_rows),
                "shortlist_identity": shortlist["shortlist_identity"],
                "selected_candidate_hashes": sorted(str(row["candidate_set_hash"]) for row in shortlist_rows),
                "selected_candidate_set_count": len(shortlist_rows),
                "manual_included": any(row["candidate_set_hash"] == MANUAL_SET_HASH for row in shortlist_rows),
                "sealed_before_r4_training": True,
            },
        )
        write_csv_once(result / "AMINER_H2_R3_ROBUST_SHORTLIST.csv", shortlist_rows)
        write_json_once(result / "AMINER_H2_R3_SHORTLIST_SEAL.json", shortlist_seal)
        next_rows, next_seal = build_continuation_plan(
            prior_rows=plan,
            selected_candidate_hashes={str(row["candidate_set_hash"]) for row in shortlist_rows},
            result_root=result,
            next_endpoint=3000,
            prior_seal_identity=seal["identity"],
            selection_seal=shortlist_seal,
        )
        _write_plan(result, "R4", next_rows, next_seal)
        return {"status": "PASS", "next_stage": "R4", "shortlist_set_count": len(shortlist_rows), "next_row_count": len(next_rows)}
    if logical_stage != "R4":
        raise ValueError("unsupported H2 search transition")
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
    selection_status = "AMINER_H2_MANUAL_SELECTED" if closed["status"] == "MANUAL_SELECTED" else "AMINER_H2_NEW_SET_SELECTED"
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
        "metachorus.v53.m7f-h2.selection-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        selection_basis,
    )
    write_csv_once(result / "AMINER_H2_R4_MANUAL_SAFETY.csv", safety)
    write_csv_once(result / "AMINER_H2_R4_ROBUST_PARETO.csv", robust)
    write_json_once(result / "AMINER_H2_SELECTED_SET.json", envelope("metachorus.v53.m7f-h2.selected-set.v1", selection_status, {"selected_set": selected, "selection_seal_identity": selection_seal["identity"]}))
    write_json_once(result / "AMINER_H2_SELECTION_SEAL.json", selection_seal)
    write_text_once(
        result / "AMINER_H2_SELECTION_DECISION.md",
        "# AMiner H2 Frozen Selection\n\n"
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


def _zip_csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    if name not in archive.namelist():
        raise ValueError(f"H2 parent package lacks {name}")
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))


def _zip_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    if name not in archive.namelist():
        raise ValueError(f"H2 parent package lacks {name}")
    value = json.loads(archive.read(name).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"H2 parent member is not an object: {name}")
    return value


def audit_manual_reuse(
    *,
    parent_package: str | Path,
    method_freeze_path: str | Path,
    result_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent = Path(parent_package).resolve(strict=True)
    if (
        parent.is_symlink()
        or not parent.is_file()
        or parent.name != PARENT_PACKAGE_FILENAME
        or parent.stat().st_size != PARENT_PACKAGE_SIZE_BYTES
        or sha256_file(parent) != PARENT_PACKAGE_SHA256
    ):
        raise ValueError("H2 parent package identity changed")
    method = read_json(method_freeze_path)
    if (
        method.get("status") != "PASS"
        or method.get("identity") != canonical_json_hash(method.get("basis", {}))
        or method.get("basis", {}).get("training_core_unchanged_from_parent") is not True
        or method.get("basis", {}).get("trainer") != "O0-CodeFaithful"
        or method.get("basis", {}).get("view_pipeline") != "S0"
    ):
        raise ValueError("H2 method freeze cannot authorize Manual reuse")
    with zipfile.ZipFile(parent) as archive:
        if archive.testzip() is not None:
            raise ValueError("H2 parent ZIP CRC failed")
        metric_rows = [
            row
            for row in _zip_csv(archive, "DBLP_AMINER_MANUAL_METRICS_BY_SEED.csv")
            if str(row.get("dataset", "")).casefold() == DATASET
        ]
        registry_rows = [
            row
            for row in _zip_csv(archive, "M7F_H1_RUN_REGISTRY_MINIMAL.csv")
            if str(row.get("dataset", "")).casefold() == DATASET
            and str(row.get("stage", "")) == "MANUAL_BASELINE"
        ]
        git_identity = _zip_json(archive, "M7F_H1_GIT_AND_CONFIG_IDENTITY.json")
    git_basis = git_identity.get("basis", {})
    if (
        git_identity.get("status") != "PASS"
        or git_identity.get("identity") != canonical_json_hash(git_basis)
        or git_basis.get("source_commit") != PARENT_SOURCE_COMMIT
        or git_basis.get("source_tree") != PARENT_SOURCE_TREE
        or git_basis.get("cache_identities", {}).get(DATASET) != CACHE_IDENTITY
        or git_basis.get("path_library_identities", {}).get(DATASET) != PATH_LIBRARY_IDENTITY
        or git_basis.get("all_sets_identities", {}).get(DATASET) != ALL_SETS_IDENTITY
        or git_basis.get("public_seed_role_identities", {}).get("FAST_LABEL_ROLE_MANIFEST.json")
        != LABEL_ROLE_MANIFEST_IDENTITY
        or git_basis.get("labels_mounted_during_training") is not False
        or git_basis.get("algorithm_retuning") is not False
    ):
        raise ValueError("H2 Manual parent governance identity is not exact")
    if (
        len(metric_rows) != 3
        or len(registry_rows) != 3
        or {int(row["model_seed"]) for row in metric_rows} != set(EVALUATION_MODEL_SEEDS)
        or {int(row["model_seed"]) for row in registry_rows} != set(EVALUATION_MODEL_SEEDS)
    ):
        raise ValueError("H2 Manual reuse requires exactly three evaluation seeds")
    registry_by_seed = {int(row["model_seed"]): row for row in registry_rows}
    audits: list[dict[str, Any]] = []
    normalized_metrics: list[dict[str, Any]] = []
    for row in sorted(metric_rows, key=lambda value: EVALUATION_MODEL_SEEDS.index(int(value["model_seed"]))):
        seed = int(row["model_seed"])
        registry = registry_by_seed[seed]
        metrics = {metric: float(row[metric]) for metric in METRICS}
        formula_pass = math.isclose(metrics["F_cls"], (metrics["Micro_F1"] + metrics["Macro_F1"]) / 2.0, abs_tol=1e-12) and math.isclose(
            metrics["F_clu"], (metrics["NMI"] + metrics["ARI"]) / 2.0, abs_tol=1e-12
        ) and math.isclose(metrics["F_joint"], (metrics["F_cls"] + metrics["F_clu"]) / 2.0, abs_tol=1e-12)
        exact = (
            row.get("candidate_set_id") == MANUAL_SET_ID
            and row.get("candidate_set_hash") == MANUAL_SET_HASH
            and int(row.get("update", row.get("endpoint_update", -1))) == PRELIM_ENDPOINT
            and row.get("classification_fit_role") == "GLOBAL_SEARCH_POOL"
            and row.get("classification_metric_role") == "PRELIM_LOCKED_REPORT"
            and row.get("clustering_fit_role") == "ALL_TARGET_NODES_UNLABELED"
            and row.get("clustering_metric_role") == "PRELIM_LOCKED_REPORT"
            and str(row.get("labels_mounted_during_training", "")).casefold() == "false"
            and str(row.get("raw_labels_persisted", "")).casefold() == "false"
            and row.get("status") == "PASS"
            and registry.get("candidate_set_id") == MANUAL_SET_ID
            and registry.get("candidate_set_hash") == MANUAL_SET_HASH
            and int(registry.get("continuation_from_update", -1)) == 0
            and int(registry.get("optimizer_steps", -1)) == PRELIM_ENDPOINT
            and int(registry.get("endpoint_update", -1)) == PRELIM_ENDPOINT
            and str(registry.get("labels_mounted", "")).casefold() == "false"
            and registry.get("status") == "PASS"
            and formula_pass
        )
        if not exact:
            raise ValueError(f"H2 Manual seed {seed} is not exactly reusable")
        audits.append(
            {
                "dataset": DATASET,
                "model_seed": seed,
                "physical_run_id": row["physical_run_id"],
                "candidate_set_id": MANUAL_SET_ID,
                "candidate_set_hash": MANUAL_SET_HASH,
                "dataset_cache_path_set_identity_match": True,
                "o0_s0_identity_match": True,
                "step0_to_3000_match": True,
                "label_roles_match": True,
                "evaluator_protocol_match": True,
                "training_core_hash_match": True,
                "parent_gpu_uuid_available": False,
                "eligible_for_h2_same_gpu_pairing": False,
                "final_pairing_action_if_selected_nonmanual": "RERUN_MANUAL_ON_H2_PAIRED_GPU",
                "metric_formula_pass": True,
                "parent_package_sha256": PARENT_PACKAGE_SHA256,
                "status": "PASS",
            }
        )
        normalized_metrics.append(
            {
                "logical_role": "MANUAL",
                "physical_run_id": row["physical_run_id"],
                "dataset": DATASET,
                "candidate_set_id": MANUAL_SET_ID,
                "candidate_set_hash": MANUAL_SET_HASH,
                "model_seed": seed,
                "endpoint_update": PRELIM_ENDPOINT,
                **metrics,
                "classification_fit_role": "GLOBAL_SEARCH_POOL",
                "classification_metric_role": "PRELIM_LOCKED_REPORT",
                "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
                "clustering_metric_role": "PRELIM_LOCKED_REPORT",
                "manual_reused": True,
                "confirmatory_claim_allowed": False,
                "status": "PASS",
            }
        )
    write_csv_once(Path(result_root).resolve() / "AMINER_H2_MANUAL_REUSE_AUDIT.csv", audits)
    write_csv_once(Path(result_root).resolve() / "AMINER_H2_MANUAL_METRICS_REUSED.csv", normalized_metrics)
    return audits, normalized_metrics


def seal_preliminary_training_registry(
    *,
    result_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    selection_seal: Mapping[str, Any],
    manual_reuse_audit_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve()
    selection_basis = selection_seal.get("basis", {})
    if (
        selection_seal.get("status") != "SEALED_BEFORE_PRELIMINARY_EVALUATION"
        or selection_seal.get("identity") != canonical_json_hash(selection_basis)
        or selection_basis.get("evaluation_seeds_started") is not False
        or selection_basis.get("no_second_choice") is not True
        or len(manual_reuse_audit_rows) != 3
        or any(row.get("status") != "PASS" for row in manual_reuse_audit_rows)
    ):
        raise ValueError("H2 preliminary cannot start before selection and Manual reuse seals")
    plan = validate_plan(rows, seal, logical_stage="PRELIMINARY")
    registry = stage_registry(result_root=result, rows=plan, seal=seal, logical_stage="PRELIMINARY") if plan else []
    selected_is_manual = selection_basis["selected_set"]["candidate_set_hash"] == MANUAL_SET_HASH
    if (selected_is_manual and registry) or (not selected_is_manual and len(registry) != 6):
        raise RuntimeError("H2 preliminary physical deduplication matrix is invalid")
    if not selected_is_manual:
        if (
            {str(row.get("logical_role")) for row in registry} != {"FINAL_FROZEN_METHOD", "MANUAL"}
            or {int(row["model_seed"]) for row in registry} != set(EVALUATION_MODEL_SEEDS)
        ):
            raise RuntimeError("H2 preliminary registry is not the paired two-arm matrix")
        for seed in EVALUATION_MODEL_SEEDS:
            pair = [row for row in registry if int(row["model_seed"]) == int(seed)]
            if (
                len(pair) != 2
                or {str(row["logical_role"]) for row in pair} != {"FINAL_FROZEN_METHOD", "MANUAL"}
                or len({str(row["gpu_uuid"]) for row in pair}) != 1
            ):
                raise RuntimeError("H2 selected/Manual pair did not run on the same GPU UUID")
    write_csv_once(
        result / "AMINER_H2_PRELIM_TRAINING_REGISTRY.csv",
        registry if registry else [{"status": "MANUAL_EXACT_REUSE", "physical_run_count": 0}],
    )
    registry_basis = {
        "selection_seal_identity": selection_seal["identity"],
        "preliminary_plan_seal_identity": seal["identity"],
        "active_datasets": [DATASET],
        "physical_run_count": len(registry),
        "registry_identity": canonical_json_hash(registry),
        "manual_reuse_identity": canonical_json_hash(list(manual_reuse_audit_rows)),
        "selected_is_manual": selected_is_manual,
        "manual_parent_reused_for_final_pairing": selected_is_manual,
        "manual_rerun_for_same_gpu_pairing": not selected_is_manual,
        "same_gpu_uuid_within_evaluation_seed": True,
        "all_status_pass": True,
        "labels_mounted": False,
        "preliminary_evaluation_started": False,
    }
    registry_seal = envelope(
        "metachorus.v53.m7f-h2.preliminary-training-registry-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        registry_basis,
    )
    write_json_once(result / "AMINER_H2_PRELIM_TRAINING_REGISTRY_SEAL.json", registry_seal)
    return registry, registry_seal


def _summary_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for role in ("FINAL_FROZEN_METHOD", "MANUAL"):
        cohort = [row for row in metrics if row["logical_role"] == role]
        if len(cohort) != 3 or {int(row["model_seed"]) for row in cohort} != set(EVALUATION_MODEL_SEEDS):
            raise ValueError("H2 preliminary summary requires paired three-seed arms")
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
    parent_package: Path,
    selected_summary: Mapping[str, Any],
    selection_status: str,
    decision_status: str,
    result: Path,
) -> list[dict[str, Any]]:
    with zipfile.ZipFile(parent_package) as archive:
        rows = _zip_csv(archive, "UPDATED_FIVE_DATASET_STATUS_TABLE.csv")
    if {str(row.get("dataset", "")).casefold() for row in rows} != {"acm", "freebase", "imdb", "dblp", "aminer"}:
        raise ValueError("H2 parent five-dataset table is incomplete")
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
                    "method_role": "FINAL_FROZEN_METHOD" if selection_status == "AMINER_H2_NEW_SET_SELECTED" else "MANUAL",
                    **{metric: selected_summary[f"mean_{metric}"] for metric in METRICS},
                    "model_seed_count": 3,
                    "evidence_level": "3_SEED_PRELIMINARY_NONCONFIRMATORY",
                    "confirmatory_claim_allowed": False,
                    "status": decision_status,
                    "execution_status": "AMINER_H2_FULL_SEARCH_COMPLETE",
                    "selection_status": selection_status,
                    "preliminary_decision": decision_status,
                }
            )
        output.append(value)
    write_csv_once(result / "UPDATED_FIVE_DATASET_STATUS_TABLE.csv", output)
    lines = [
        "# Updated Five-Dataset Status",
        "",
        "| Dataset | Execution | Selection | Preliminary | Evidence | Confirmatory |",
        "|---|---|---|---|---|---|",
    ]
    for row in output:
        lines.append(
            f"| {row['dataset']} | {row['execution_status']} | {row['selection_status']} | "
            f"{row['preliminary_decision']} | {row['evidence_level']} | {str(row['confirmatory_claim_allowed']).lower()} |"
        )
    write_text_once(result / "UPDATED_FIVE_DATASET_STATUS_TABLE.md", "\n".join(lines) + "\n")
    return output


def evaluate_preliminary(
    *,
    result_root: str | Path,
    parent_package: str | Path,
    method_freeze_path: str | Path,
    role_manifest_path: str | Path,
    private_search_mount_root: str | Path,
    label_source: str | Path,
    label_source_root: str | Path,
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    parent = Path(parent_package).resolve(strict=True)
    audits, parent_manual_metrics = audit_manual_reuse(
        parent_package=parent,
        method_freeze_path=method_freeze_path,
        result_root=result,
    )
    selection_seal = read_json(result / "AMINER_H2_SELECTION_SEAL.json")
    prelim_rows = read_csv(result / "AMINER_H2_PRELIMINARY_RUN_PLAN.csv")
    prelim_seal = read_json(result / "AMINER_H2_PRELIMINARY_RUN_PLAN_SEAL.json")
    registry, registry_seal = seal_preliminary_training_registry(
        result_root=result,
        rows=prelim_rows,
        seal=prelim_seal,
        selection_seal=selection_seal,
        manual_reuse_audit_rows=audits,
    )
    selected = selection_seal["basis"]["selected_set"]
    selected_is_manual = selected["candidate_set_hash"] == MANUAL_SET_HASH
    selected_metrics: list[dict[str, Any]] = []
    manual_metrics: list[dict[str, Any]] = []
    access_count = 0
    if selected_is_manual:
        manual_metrics = parent_manual_metrics
        selected_metrics = [
            {
                **row,
                "logical_role": "FINAL_FROZEN_METHOD",
                "manual_reused": True,
                "physical_run_deduplicated_with_manual": True,
            }
            for row in manual_metrics
        ]
    else:
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
            _h2, _h2_basis, engine = _engine_marker(result, row, "PRELIMINARY")
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
                    "physical_run_deduplicated_with_manual": False,
                    "confirmatory_claim_allowed": False,
                    "status": "PASS",
                }
            )
        selected_metrics = [row for row in evaluated_rows if row["logical_role"] == "FINAL_FROZEN_METHOD"]
        manual_metrics = [row for row in evaluated_rows if row["logical_role"] == "MANUAL"]
        access_count = session.access_count
        if access_count != 1 or session.per_dataset_access_count[DATASET] != 1:
            raise RuntimeError("H2 preliminary label source must be opened exactly once")
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
            raise RuntimeError("H2 final selected/Manual comparison is not GPU-paired")
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
    raw_decision = preliminary_decision(selected_metrics, manual_metrics)
    if selected_is_manual:
        decision_status = "AMINER_FAST_SELECTED_MANUAL"
    else:
        decision_status = {
            "PROMISING": "AMINER_FAST_PROMISING",
            "NONINFERIOR": "AMINER_FAST_NONINFERIOR",
            "UNSAFE": "AMINER_FAST_UNSAFE",
            "INDETERMINATE": "AMINER_FAST_UNSAFE",
        }[raw_decision["status"]]
    selection_status = selection_seal["basis"]["selection_status"]
    decision_basis = {
        "dataset": DATASET,
        "selection_status": selection_status,
        "selected_set": selected,
        "manual_set_id": MANUAL_SET_ID,
        "manual_set_hash": MANUAL_SET_HASH,
        "preliminary_decision": decision_status,
        "raw_frozen_decision": raw_decision,
        "selected_is_manual": selected_is_manual,
        "manual_reused": selected_is_manual,
        "manual_rerun_for_same_gpu_pairing": not selected_is_manual,
        "manual_reuse_audit_identity": canonical_json_hash(audits),
        "selected_metric_rows_identity": canonical_json_hash(selected_metrics),
        "manual_metric_rows_identity": canonical_json_hash(manual_metrics),
        "locked_label_source_access_count": access_count,
        "no_second_choice": True,
        "search_reopened": False,
        "ten_seed_expansion_started": False,
        "confirmatory_claim_allowed": False,
    }
    decision = envelope("metachorus.v53.m7f-h2.final-decision.v1", decision_status, decision_basis)
    write_csv_once(result / "AMINER_H2_PRELIM_METRICS_BY_SEED.csv", all_metrics)
    write_csv_once(result / "AMINER_H2_PRELIM_SUMMARY.csv", summaries)
    write_csv_once(result / "AMINER_H2_SELECTED_VS_MANUAL.csv", comparisons)
    write_json_once(result / "AMINER_H2_FINAL_DECISION.json", decision)
    write_text_once(
        result / "AMINER_H2_FINAL_DECISION.md",
        "# AMiner H2 Three-Seed Preliminary Decision\n\n"
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
        result / "AMINER_H2_FINAL_SET_REGISTRY.csv",
        [
            {
                "dataset": DATASET,
                "candidate_set_id": selected["candidate_set_id"],
                "candidate_set_hash": selected["candidate_set_hash"],
                "candidate_set_ordinal": int(selected["candidate_set_ordinal"]),
                "path_ids": list(selected["path_ids"]),
                "selection_status": selection_status,
                "preliminary_decision": decision_status,
                "selection_seal_identity": selection_seal["identity"],
                "final_decision_identity": decision["identity"],
                "no_second_choice": True,
                "status": "PASS",
            }
        ],
    )
    selected_summary = next(row for row in summaries if row["logical_role"] == "FINAL_FROZEN_METHOD")
    _update_five_dataset_table(
        parent_package=parent,
        selected_summary=selected_summary,
        selection_status=selection_status,
        decision_status=decision_status,
        result=result,
    )
    expansion_pending = decision_status in {"AMINER_FAST_PROMISING", "AMINER_FAST_NONINFERIOR"}
    write_text_once(
        result / "AMINER_10SEED_EXPANSION_PLAN_PENDING.md",
        "# AMiner 10-Seed Expansion\n\n"
        f"- approval_status: `{'PENDING_USER_APPROVAL' if expansion_pending else 'NOT_AUTHORIZED_BY_THREE_SEED_DECISION'}`\n"
        "- started: `false`\n"
        "- frozen_set_only: `true`\n",
    )
    write_csv_once(
        result / "AMINER_10SEED_EXPANSION_MATRIX_PENDING.csv",
        [
            {
                "dataset": DATASET,
                "selected_set_id": selected["candidate_set_id"],
                "selected_set_hash": selected["candidate_set_hash"],
                "status": "PENDING_USER_APPROVAL" if expansion_pending else "NOT_AUTHORIZED",
                "training_started": False,
            }
        ],
    )
    approval = envelope(
        "metachorus.v53.m7f-h2.ten-seed-approval.v1",
        "PENDING_USER_APPROVAL" if expansion_pending else "NOT_AUTHORIZED",
        {
            "preliminary_decision": decision_status,
            "final_decision_identity": decision["identity"],
            "ten_seed_expansion_started": False,
            "user_approval_received": False,
        },
    )
    write_json_once(result / "AMINER_10SEED_APPROVAL_STATUS.json", approval)
    write_json_once(
        result / "AMINER_H2_PRELIMINARY_ACCESS_AUDIT.json",
        envelope(
            "metachorus.v53.m7f-h2.preliminary-access-audit.v1",
            "PASS",
            {
                "training_registry_seal_identity": registry_seal["identity"],
                "label_source_access_count": access_count,
                "manual_metrics_reused_without_label_reaccess": selected_is_manual,
                "manual_rerun_evaluated_in_same_label_session": not selected_is_manual,
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
    "audit_manual_reuse",
    "seal_preliminary_training_registry",
    "evaluate_preliminary",
]
