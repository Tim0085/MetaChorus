"""Locked DBLP evaluation, paired statistics, and final freeze for M7F-H4."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.m4a_h8.constants import FROZEN_PATH_SETS
from metachorus_v53.m4a_h8.protocol import paired_interval as h8_paired_interval
from metachorus_v53.m6f_m7f_fast.evaluation import (
    PreliminaryLabelSession,
    evaluate_preliminary_representation,
)
from metachorus_v53.m6f_m7f_fast.protocol import METRICS
from metachorus_v53.m7f_h1.constants import IMDB_MANUAL_SET, MANUAL_SET_IDENTITIES

from .common import (
    canonical_json_hash,
    envelope,
    read_csv,
    read_json,
    sha256_file,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    CONFIRMATION_ENDPOINT,
    CONFIRMATION_MODEL_SEEDS,
    DATASET,
    H8_BOOTSTRAP_NAMESPACE,
    H8_BOOTSTRAP_RESAMPLES,
    H8_STATISTICS_FUNCTION,
    H8_STATISTICS_RELATIVE_PATH,
    H8_STATISTICS_SHA256,
    KMEANS_SEEDS,
    MANUAL_PATH_IDS,
    MANUAL_SET_HASH,
    MANUAL_SET_ID,
    MANUAL_SET_ORDINAL,
    SELECTED_PATH_IDS,
    SELECTED_SET_HASH,
    SELECTED_SET_ID,
    SELECTED_SET_ORDINAL,
)
from .planning import canonical_plan_row, validate_plan
from .training import completed_marker, validate_completion_marker


COMPARISON = "DBLP_H4_SELECTED_VS_MANUAL"


def _governed_file(result: Path, relative: str) -> Path:
    if "\\" in relative:
        raise ValueError("governed relative path uses a backslash")
    candidate = result / Path(*relative.split("/"))
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("governed evidence is not a regular file")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(result)
    except ValueError as exc:
        raise ValueError("governed evidence escapes the result root") from exc
    return resolved


def _load_representation(result: Path, basis: Mapping[str, Any]) -> np.ndarray:
    path = _governed_file(result, str(basis["representation_relative_path"]))
    if sha256_file(path) != str(basis["representation_sha256"]):
        raise ValueError("M7F-H4 representation hash changed")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[0] != 4057 or not np.isfinite(values).all():
        raise ValueError("M7F-H4 representation shape/finiteness is invalid")
    return values


def close_canaries(
    *, result_root: str | Path, rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    plan = validate_plan(rows, seal, logical_stage="CANARY")
    registry = []
    for row in plan:
        completed = completed_marker(result, row, "CANARY")
        if completed is None:
            raise RuntimeError("M7F-H4 canary matrix is incomplete")
        basis = validate_completion_marker(completed[1], row, logical_stage="CANARY")
        audit = basis.get("canary_engineering_audit", {})
        if (
            audit.get("status") != "PASS"
            or audit.get("checkpoint_write_pass") is not True
            or audit.get("checkpoint_read_pass") is not True
            or audit.get("checkpoint_state_roundtrip") is not True
            or audit.get("labels_mounted") is not False
            or int(audit.get("downstream_evaluations_during_training", -1)) != 0
        ):
            raise RuntimeError("M7F-H4 canary checkpoint/label boundary failed")
        registry.append(basis)
    if len(registry) != 4 or len({row["gpu_uuid"] for row in registry}) != 4:
        raise RuntimeError("M7F-H4 canaries did not cover four unique GPUs")
    result_value = envelope(
        "metachorus.v53.m7f-h4.canary-results.v1",
        "PASS",
        {
            "canary_plan_seal_identity": seal["identity"],
            "run_count": 4,
            "run_registry_identity": canonical_json_hash(registry),
            "gpu_uuids": sorted(str(row["gpu_uuid"]) for row in registry),
            "all_checkpoint_roundtrip_pass": True,
            "all_labels_mounted": False,
            "all_downstream_evaluations_during_training": 0,
            "scientific_results_allowed": False,
            "formal_confirmation_restarts_from_zero": True,
        },
    )
    write_csv_once(result / "DBLP_H4_CANARY_RUN_REGISTRY.csv", registry)
    write_json_once(result / "DBLP_H4_CANARY_RESULTS.json", result_value)
    return result_value


def seal_confirmation_training_registry(
    *, result_root: str | Path, rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve()
    plan = validate_plan(rows, seal, logical_stage="CONFIRMATION")
    registry: list[dict[str, Any]] = []
    for row in plan:
        completed = completed_marker(result, row, "CONFIRMATION")
        if completed is None:
            raise RuntimeError("M7F-H4 confirmation matrix is incomplete")
        basis = validate_completion_marker(completed[1], row, logical_stage="CONFIRMATION")
        registry.append(
            {
                **basis,
                "completion_marker_relative_path": completed[0].relative_to(result).as_posix(),
                "completion_marker_sha256": sha256_file(completed[0]),
                "status": "PASS",
            }
        )
    if len(registry) != 20 or len({row["physical_run_id"] for row in registry}) != 20:
        raise RuntimeError("M7F-H4 training registry is not the exact 20-run matrix")
    for seed in CONFIRMATION_MODEL_SEEDS:
        pair = [row for row in registry if int(row["model_seed"]) == int(seed)]
        if (
            len(pair) != 2
            or {row["logical_role"] for row in pair} != {"DBLP_SELECTED", "DBLP_MANUAL"}
            or len({row["gpu_uuid"] for row in pair}) != 1
            or len({row["gpu_pci_bus_id"] for row in pair}) != 1
            or len({row["runtime_domain"] for row in pair}) != 1
            or len({row["resolved_run_config_identity"] for row in pair}) != 2
        ):
            raise RuntimeError("M7F-H4 selected/Manual pair crossed GPU or runtime domain")
    seal_basis = {
        "confirmation_matrix_seal_identity": seal["identity"],
        "active_datasets": [DATASET],
        "physical_run_count": len(registry),
        "model_seed_count": len(CONFIRMATION_MODEL_SEEDS),
        "registry_identity": canonical_json_hash(registry),
        "same_model_seed_paired": True,
        "same_gpu_uuid_within_model_seed": True,
        "same_gpu_pci_bus_id_within_model_seed": True,
        "same_runtime_domain_within_model_seed": True,
        "all_unique_runs_trained_fresh_from_zero": True,
        "all_status_pass": True,
        "labels_mounted": False,
        "preliminary_evaluation_started": False,
    }
    registry_seal = envelope(
        "metachorus.v53.m7f-h4.confirmation-training-registry-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        seal_basis,
    )
    write_csv_once(result / "DBLP_H4_RUN_REGISTRY_MINIMAL.csv", registry)
    write_json_once(result / "DBLP_H4_CONFIRMATION_TRAINING_REGISTRY_SEAL.json", registry_seal)
    return registry, registry_seal


def paired_delta_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = {
        int(row["model_seed"]): row for row in metrics if row["logical_role"] == "DBLP_SELECTED"
    }
    manual = {
        int(row["model_seed"]): row for row in metrics if row["logical_role"] == "DBLP_MANUAL"
    }
    if set(selected) != set(CONFIRMATION_MODEL_SEEDS) or set(manual) != set(CONFIRMATION_MODEL_SEEDS):
        raise ValueError("M7F-H4 paired delta matrix lacks exact ten-seed arms")
    rows: list[dict[str, Any]] = []
    for seed in CONFIRMATION_MODEL_SEEDS:
        left = selected[int(seed)]
        right = manual[int(seed)]
        if (
            left["gpu_uuid"] != right["gpu_uuid"]
            or left["gpu_pci_bus_id"] != right["gpu_pci_bus_id"]
            or left["runtime_domain"] != right["runtime_domain"]
        ):
            raise ValueError("M7F-H4 paired delta rows are not same-domain observations")
        for metric in METRICS:
            candidate_value = float(left[metric])
            baseline_value = float(right[metric])
            rows.append(
                {
                    "comparison": COMPARISON,
                    "dataset": DATASET,
                    "candidate_role": "DBLP_SELECTED",
                    "baseline_role": "DBLP_MANUAL",
                    "model_seed": int(seed),
                    "metric": metric,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "delta": candidate_value - baseline_value,
                    "gpu_uuid": left["gpu_uuid"],
                    "gpu_pci_bus_id": left["gpu_pci_bus_id"],
                    "runtime_domain": left["runtime_domain"],
                    "paired_by_same_model_seed": True,
                    "paired_by_same_gpu_uuid": True,
                    "paired_by_same_runtime_domain": True,
                    "status": "PASS",
                }
            )
    return rows


def confirmed_improvement_gate(
    interval_rows: Sequence[Mapping[str, Any]], delta_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    intervals = {str(row["metric"]): row for row in interval_rows}
    per_seed = {
        metric: [float(row["delta"]) for row in delta_rows if row["metric"] == metric]
        for metric in ("F_cls", "F_clu")
    }
    checks = {
        "bootstrap_joint_lower_gt_zero": float(intervals["F_joint"]["bootstrap_lower_95"]) > 0.0,
        "bootstrap_clu_lower_gt_zero": float(intervals["F_clu"]["bootstrap_lower_95"]) > 0.0,
        "bootstrap_cls_lower_at_least_minus_0_01": float(intervals["F_cls"]["bootstrap_lower_95"]) >= -0.01,
        "joint_non_worse_at_least_7_of_10": int(intervals["F_joint"]["non_worse_count"]) >= 7,
        "clustering_non_worse_at_least_7_of_10": int(intervals["F_clu"]["non_worse_count"]) >= 7,
        "every_seed_cls_delta_gt_minus_0_02": min(per_seed["F_cls"]) > -0.02,
        "every_seed_clu_delta_gt_minus_0_02": min(per_seed["F_clu"]) > -0.02,
    }
    return {"gate": "DBLP_H4_CONFIRMED_IMPROVEMENT", "checks": checks, "passed": all(checks.values())}


def confirmed_noninferior_gate(
    interval_rows: Sequence[Mapping[str, Any]], delta_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    intervals = {str(row["metric"]): row for row in interval_rows}
    per_seed = {
        metric: [float(row["delta"]) for row in delta_rows if row["metric"] == metric]
        for metric in ("F_cls", "F_clu")
    }
    checks = {
        "bootstrap_cls_lower_at_least_minus_0_01": float(intervals["F_cls"]["bootstrap_lower_95"]) >= -0.01,
        "bootstrap_clu_lower_at_least_minus_0_01": float(intervals["F_clu"]["bootstrap_lower_95"]) >= -0.01,
        "bootstrap_joint_lower_at_least_minus_0_005": float(intervals["F_joint"]["bootstrap_lower_95"]) >= -0.005,
        "joint_non_worse_at_least_6_of_10": int(intervals["F_joint"]["non_worse_count"]) >= 6,
        "every_seed_cls_delta_gt_minus_0_02": min(per_seed["F_cls"]) > -0.02,
        "every_seed_clu_delta_gt_minus_0_02": min(per_seed["F_clu"]) > -0.02,
    }
    return {"gate": "DBLP_H4_CONFIRMED_NONINFERIOR", "checks": checks, "passed": all(checks.values())}


def decision_status(
    interval_rows: Sequence[Mapping[str, Any]], delta_rows: Sequence[Mapping[str, Any]]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    improvement = confirmed_improvement_gate(interval_rows, delta_rows)
    noninferior = confirmed_noninferior_gate(interval_rows, delta_rows)
    if improvement["passed"]:
        status = "DBLP_H4_CONFIRMED_IMPROVEMENT"
    elif noninferior["passed"]:
        status = "DBLP_H4_CONFIRMED_NONINFERIOR"
    else:
        status = "DBLP_H4_FALLBACK_MANUAL"
    return status, improvement, noninferior


def _summary(cohort: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(cohort) != 10 or {int(row["model_seed"]) for row in cohort} != set(CONFIRMATION_MODEL_SEEDS):
        raise ValueError("M7F-H4 role summary requires exact ten seeds")
    return {
        **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in cohort) for metric in METRICS},
        **{f"sample_sd_{metric}": statistics.stdev(float(row[metric]) for row in cohort) for metric in METRICS},
        "assignment_stability_mean": statistics.mean(float(row["assignment_stability"]) for row in cohort),
        "model_seed_count": 10,
    }


def _h8_final_set(dataset: str) -> dict[str, Any]:
    matches = [row for row in FROZEN_PATH_SETS if row["dataset"] == dataset and row["role"] == "H7_SELECTED"]
    if len(matches) != 1:
        raise ValueError(f"missing H8 frozen set identity: {dataset}")
    return dict(matches[0])


def _final_five_dataset_table(
    *, pre_rows: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], status: str
) -> list[dict[str, Any]]:
    pre = {str(row["dataset"]): dict(row) for row in pre_rows}
    if set(pre) != {"acm", "freebase", "imdb", "dblp", "aminer"}:
        raise ValueError("M7F-H4 pre-confirmation table is incomplete")
    final_role = "DBLP_SELECTED" if status in {"DBLP_H4_CONFIRMED_IMPROVEMENT", "DBLP_H4_CONFIRMED_NONINFERIOR"} else "DBLP_MANUAL"
    summary = _summary([row for row in metrics if row["logical_role"] == final_role])
    dblp_set = {
        "candidate_set_id": SELECTED_SET_ID,
        "candidate_set_hash": SELECTED_SET_HASH,
        "final_method_role": "BUDGETED_SELECTED_SET_CONFIRMED",
    } if final_role == "DBLP_SELECTED" else {
        "candidate_set_id": MANUAL_SET_ID,
        "candidate_set_hash": MANUAL_SET_HASH,
        "final_method_role": "MANUAL_FALLBACK",
    }
    acm_set = _h8_final_set("acm")
    freebase_set = _h8_final_set("freebase")
    identities = {
        "acm": ("MANUAL", acm_set["candidate_set_id"], acm_set["candidate_set_hash"]),
        "freebase": ("CONFIRMED_NEW_SET", freebase_set["candidate_set_id"], freebase_set["candidate_set_hash"]),
        "imdb": ("MANUAL_FALLBACK", IMDB_MANUAL_SET["candidate_set_id"], IMDB_MANUAL_SET["candidate_set_hash"]),
        "aminer": ("MANUAL_FALLBACK", MANUAL_SET_IDENTITIES["aminer"]["candidate_set_id"], MANUAL_SET_IDENTITIES["aminer"]["candidate_set_hash"]),
    }
    rows: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase", "imdb", "dblp", "aminer"):
        source = pre[dataset]
        if dataset == "dblp":
            rows.append(
                {
                    "dataset": dataset,
                    "final_method_role": dblp_set["final_method_role"],
                    "final_set_id": dblp_set["candidate_set_id"],
                    "final_set_hash": dblp_set["candidate_set_hash"],
                    **{metric: summary[f"mean_{metric}"] for metric in METRICS},
                    "model_seed_count": 10,
                    "search_evidence": "BUDGET_LIMITED_ONE_SEED_FULL_SPACE_SCREEN_PLUS_THREE_SEED_REFINEMENT",
                    "confirmation_evidence": "10_SEED_PAIRED_PROTOCOL_FROZEN_INTERNAL_REPORT_SUBSET",
                    "execution_status": "DBLP_H4_CONFIRMATION_COMPLETE",
                    "selection_status": "DBLP_BUDGETED_NEW_SET_SELECTED",
                    "final_freeze_status": status,
                    "claim_boundary": "NO_FULL_THREE_SEED_SEARCH_SPACE_OPTIMUM_CLAIM",
                }
            )
            continue
        role, set_id, set_hash = identities[dataset]
        confirmation = {
            "acm": "10_SEED_LOCKED_CONFIRMATION",
            "freebase": "10_SEED_LOCKED_CONFIRMATION",
            "imdb": "3_SEED_PRELIMINARY_MANUAL_FIXED_END_3000",
            "aminer": "3_SEED_SAME_GPU_PAIRED_MANUAL_FALLBACK_NONCONFIRMATORY",
        }[dataset]
        rows.append(
            {
                "dataset": dataset,
                "final_method_role": role,
                "final_set_id": set_id,
                "final_set_hash": set_hash,
                **{metric: source.get(metric, "") for metric in METRICS},
                "model_seed_count": source.get("model_seed_count", ""),
                "search_evidence": source.get("evidence_level", "PARENT_FROZEN"),
                "confirmation_evidence": confirmation,
                "execution_status": source.get("execution_status", source.get("status", "")),
                "selection_status": source.get("selection_status", source.get("search_selection_status", "PARENT_FROZEN")),
                "final_freeze_status": source.get("final_freeze_status", "PARENT_FROZEN"),
                "claim_boundary": "PARENT_FROZEN" if dataset in {"acm", "freebase"} else "NONCONFIRMATORY_INTERNAL_REPORT",
            }
        )
    return rows


def _markdown_table(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    fields = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def evaluate_locked_confirmation(
    *,
    result_root: str | Path,
    role_manifest_path: str | Path,
    seed_manifest_path: str | Path,
    private_search_mount_root: str | Path,
    label_source: str | Path,
    label_source_root: str | Path,
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    rows = read_csv(result / "DBLP_H4_CONFIRMATION_MATRIX.csv")
    seal = read_json(result / "DBLP_H4_CONFIRMATION_MATRIX_SEAL.json")
    registry, registry_seal = seal_confirmation_training_registry(
        result_root=result, rows=rows, seal=seal
    )
    seed_manifest = read_json(seed_manifest_path)
    seed_basis = seed_manifest.get("basis", {})
    if (
        seed_manifest.get("identity") != "ae8677f2675ebab870fa5ee7796cca3021a0635579fe0193e88fcfb3881addc6"
        or seed_manifest.get("identity") != canonical_json_hash(seed_basis)
        or tuple(map(int, seed_basis.get("preliminary_kmeans_seeds", []))) != tuple(KMEANS_SEEDS)
        or len(KMEANS_SEEDS) != 20
    ):
        raise ValueError("M7F-H4 KMeans seed manifest identity changed")
    session = PreliminaryLabelSession(
        role_manifest_path=role_manifest_path,
        private_search_mount_root=private_search_mount_root,
        label_sources={DATASET: label_source},
        label_source_roots={DATASET: label_source_root},
        training_registry_seal=registry_seal,
    )
    roles = session.open_dataset(DATASET)
    plan_by_id = {str(row["physical_run_id"]): canonical_plan_row(row) for row in rows}
    metrics: list[dict[str, Any]] = []
    classification: list[dict[str, Any]] = []
    clustering_long: list[dict[str, Any]] = []
    clustering_summary: list[dict[str, Any]] = []
    for registry_row in registry:
        plan_row = plan_by_id[str(registry_row["physical_run_id"])]
        evaluated = evaluate_preliminary_representation(
            _load_representation(result, registry_row), dataset=DATASET, roles=roles
        )
        metric_row = {
            "dataset": DATASET,
            "logical_role": plan_row["logical_role"],
            "physical_run_id": plan_row["physical_run_id"],
            "set_id": plan_row["candidate_set_id"],
            "set_hash": plan_row["candidate_set_hash"],
            "candidate_set_id": plan_row["candidate_set_id"],
            "candidate_set_hash": plan_row["candidate_set_hash"],
            "model_seed": int(plan_row["model_seed"]),
            **evaluated["metrics"],
            "assignment_stability": evaluated["assignment_stability_mean"],
            "assignment_stability_p10": evaluated["assignment_stability_p10"],
            "GPU_UUID": registry_row["gpu_uuid"],
            "gpu_uuid": registry_row["gpu_uuid"],
            "gpu_pci_bus_id": registry_row["gpu_pci_bus_id"],
            "runtime_domain": registry_row["runtime_domain"],
            "training_seconds": float(registry_row["invocation_wall_seconds"]),
            "classification_fit_role": evaluated["classification_fit_role"],
            "classification_metric_role": evaluated["classification_metric_role"],
            "clustering_fit_role": evaluated["clustering_fit_role"],
            "clustering_metric_role": evaluated["clustering_metric_role"],
            "labels_mounted_during_training": False,
            "status": "PASS",
        }
        metrics.append(metric_row)
        classification.append(
            {
                key: metric_row[key]
                for key in (
                    "dataset", "logical_role", "physical_run_id", "set_id", "set_hash",
                    "model_seed", "Micro_F1", "Macro_F1", "F_cls", "classification_fit_role",
                    "classification_metric_role", "gpu_uuid", "runtime_domain", "status",
                )
            }
        )
        clustering_summary.append(
            {
                key: metric_row[key]
                for key in (
                    "dataset", "logical_role", "physical_run_id", "set_id", "set_hash",
                    "model_seed", "NMI", "ARI", "F_clu", "assignment_stability",
                    "assignment_stability_p10", "clustering_fit_role", "clustering_metric_role",
                    "gpu_uuid", "runtime_domain", "status",
                )
            }
        )
        for repeat in evaluated["clustering_repeats"]:
            clustering_long.append(
                {
                    "dataset": DATASET,
                    "logical_role": plan_row["logical_role"],
                    "physical_run_id": plan_row["physical_run_id"],
                    "set_hash": plan_row["candidate_set_hash"],
                    "model_seed": int(plan_row["model_seed"]),
                    **repeat,
                }
            )
    if session.access_count != 1 or session.per_dataset_access_count[DATASET] != 1:
        raise RuntimeError("M7F-H4 DBLP label source must be opened exactly once")
    if len(metrics) != 20 or len(clustering_long) != 20 * len(KMEANS_SEEDS):
        raise RuntimeError("M7F-H4 locked metric matrix is incomplete")
    deltas = paired_delta_rows(metrics)
    intervals = []
    bootstrap_manifests = []
    for metric in METRICS:
        interval, manifest = h8_paired_interval(deltas, comparison=COMPARISON, metric=metric)
        intervals.append(interval)
        bootstrap_manifests.append(manifest)
    status, improvement, noninferior = decision_status(intervals, deltas)
    final_selected = status in {"DBLP_H4_CONFIRMED_IMPROVEMENT", "DBLP_H4_CONFIRMED_NONINFERIOR"}
    final_set = {
        "candidate_set_id": SELECTED_SET_ID if final_selected else MANUAL_SET_ID,
        "candidate_set_hash": SELECTED_SET_HASH if final_selected else MANUAL_SET_HASH,
        "candidate_set_ordinal": SELECTED_SET_ORDINAL if final_selected else MANUAL_SET_ORDINAL,
        "path_ids": list(SELECTED_PATH_IDS if final_selected else MANUAL_PATH_IDS),
        "final_method_role": "BUDGETED_SELECTED_SET_CONFIRMED" if final_selected else "MANUAL_FALLBACK",
    }
    stats_identity = envelope(
        "metachorus.v53.m7f-h4.statistics-identity.v1",
        "PASS",
        {
            "module_path": H8_STATISTICS_RELATIVE_PATH,
            "function_name": H8_STATISTICS_FUNCTION,
            "source_sha256": H8_STATISTICS_SHA256,
            "bootstrap_method": "PAIRED_MODEL_SEED_INDEX_PERCENTILE_MEAN_LINEAR_QUANTILE",
            "confidence_level": 0.95,
            "resample_count": H8_BOOTSTRAP_RESAMPLES,
            "seed_namespace": H8_BOOTSTRAP_NAMESPACE,
            "paired_sampling_unit": "model_seed",
            "arms_resampled_independently": False,
            "comparison": COMPARISON,
            "model_seeds": list(CONFIRMATION_MODEL_SEEDS),
            "delta_rows_identity": canonical_json_hash(deltas),
            "interval_rows_identity": canonical_json_hash(intervals),
            "bootstrap_seed_manifests": bootstrap_manifests,
            "statistics_implementation_rewritten": False,
        },
    )
    decision_basis = {
        "dataset": DATASET,
        "selected_set_id": SELECTED_SET_ID,
        "selected_set_hash": SELECTED_SET_HASH,
        "manual_set_id": MANUAL_SET_ID,
        "manual_set_hash": MANUAL_SET_HASH,
        "confirmation_matrix_seal_identity": seal["identity"],
        "training_registry_seal_identity": registry_seal["identity"],
        "statistics_identity": stats_identity["identity"],
        "model_seed_count": 10,
        "paired_run_count": 20,
        "improvement_gate": improvement,
        "noninferior_gate": noninferior,
        "final_set": final_set,
        "fallback_used": not final_selected,
        "search_evidence": "BUDGET_LIMITED_ONE_SEED_FULL_SPACE_SCREEN_PLUS_THREE_SEED_REFINEMENT",
        "confirmation_evidence": "10_SEED_PAIRED_PROTOCOL_FROZEN_INTERNAL_REPORT_SUBSET",
        "full_three_seed_search_space_optimum_claim": False,
        "no_second_choice": True,
        "further_search": False,
        "additional_model_seeds": False,
        "algorithm_retuning": False,
        "confirmatory_claim_allowed": True,
    }
    decision = envelope("metachorus.v53.m7f-h4.confirmation-decision.v1", status, decision_basis)
    method_freeze = envelope(
        "metachorus.v53.m7f-h4.final-method-freeze.v1",
        "FROZEN_AFTER_10SEED_PAIRED_CONFIRMATION",
        {
            "confirmation_decision_identity": decision["identity"],
            "final_set": final_set,
            "model_seed_count": 10,
            "search_reopened": False,
            "second_choice_used": False,
            "further_search": False,
            "additional_model_seeds": False,
            "algorithm_retuning": False,
        },
    )
    pre_rows = read_csv(result / "FIVE_DATASET_STATUS_TABLE_PRE_DBLP_CONFIRMATION.csv")
    five_rows = _final_five_dataset_table(pre_rows=pre_rows, metrics=metrics, status=status)
    write_csv_once(result / "DBLP_H4_LOCKED_CLASSIFICATION_METRICS.csv", classification)
    write_csv_once(result / "DBLP_H4_LOCKED_CLUSTERING_METRICS_LONG.csv", clustering_long)
    write_csv_once(result / "DBLP_H4_LOCKED_CLUSTERING_METRICS_SUMMARY.csv", clustering_summary)
    write_csv_once(result / "DBLP_H4_LOCKED_METRICS_BY_MODEL_SEED.csv", metrics)
    write_csv_once(result / "DBLP_H4_PAIRWISE_DELTAS.csv", deltas)
    write_csv_once(result / "DBLP_H4_PAIRED_INTERVALS.csv", intervals)
    write_json_once(result / "DBLP_H4_STATISTICS_IDENTITY.json", stats_identity)
    write_json_once(result / "DBLP_H4_CONFIRMATION_DECISION.json", decision)
    write_text_once(
        result / "DBLP_H4_CONFIRMATION_DECISION.md",
        "# DBLP H4 Ten-Seed Paired Confirmation\n\n"
        f"- status: `{status}`\n"
        f"- final_set_id: `{final_set['candidate_set_id']}`\n"
        f"- final_set_hash: `{final_set['candidate_set_hash']}`\n"
        f"- fallback_used: `{str(not final_selected).lower()}`\n"
        "- full_three_seed_search_space_optimum_claim: `false`\n"
        "- further_search: `false`\n"
        "- additional_model_seeds: `false`\n"
        "- algorithm_retuning: `false`\n",
    )
    write_csv_once(
        result / "DBLP_H4_FINAL_SET_REGISTRY.csv",
        [{**final_set, "dataset": DATASET, "decision_status": status, "decision_identity": decision["identity"]}],
    )
    write_json_once(result / "DBLP_H4_FINAL_METHOD_FREEZE.json", method_freeze)
    write_csv_once(result / "FIVE_DATASET_FINAL_STATUS_TABLE.csv", five_rows)
    write_text_once(
        result / "FIVE_DATASET_FINAL_STATUS_TABLE.md",
        _markdown_table(five_rows, "Five-Dataset Final Frozen Status"),
    )
    write_json_once(
        result / "DBLP_H4_LOCKED_EVALUATION_ACCESS_AUDIT.json",
        envelope(
            "metachorus.v53.m7f-h4.locked-evaluation-access-audit.v1",
            "PASS",
            {
                "training_registry_seal_identity": registry_seal["identity"],
                "label_source_access_count": session.access_count,
                "dataset_access_count": session.per_dataset_access_count[DATASET],
                "classification_fit_role": "GLOBAL_SEARCH_POOL",
                "classification_metric_role": "PRELIM_LOCKED_REPORT",
                "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
                "clustering_metric_role": "PRELIM_LOCKED_REPORT",
                "raw_labels_persisted": False,
            },
        ),
    )
    return decision


__all__ = [
    "COMPARISON",
    "close_canaries",
    "seal_confirmation_training_registry",
    "paired_delta_rows",
    "confirmed_improvement_gate",
    "confirmed_noninferior_gate",
    "decision_status",
    "evaluate_locked_confirmation",
]
