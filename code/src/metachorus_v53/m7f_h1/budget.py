"""Measured DBLP/AMiner budget gate derived only from engineering canaries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import canonical_json_hash, envelope, write_csv_once, write_json_once
from .constants import (
    BUDGET_CANARY_UPDATES,
    BUDGET_MODEL_SEEDS,
    MANUAL_DATASETS,
    RESOURCE_LIMITS,
)


CANARY_ROLES = ("MANUAL", "BUDGET_SHORTEST", "BUDGET_LONGEST")


def _lpt_makespan(durations: Sequence[float], gpu_count: int = 2) -> float:
    if gpu_count != 2 or any(value < 0 or not math.isfinite(value) for value in durations):
        raise ValueError("budget makespan requires finite non-negative two-GPU work")
    loads = [0.0, 0.0]
    for duration in sorted(map(float, durations), reverse=True):
        slot = min(range(2), key=lambda index: (loads[index], index))
        loads[slot] += duration
    return max(loads, default=0.0)


def _p90(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("measured budget requires finite non-negative observations")
    return float(np.percentile(np.asarray(values, dtype=np.float64), 90))


def _validate_canaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        dataset = str(row.get("dataset", ""))
        role = str(row.get("path_set_role", ""))
        if (
            dataset not in MANUAL_DATASETS
            or role not in CANARY_ROLES
            or int(row.get("model_seed", -1)) not in BUDGET_MODEL_SEEDS
            or int(row.get("endpoint_update", BUDGET_CANARY_UPDATES))
            != BUDGET_CANARY_UPDATES
            or row.get("status") != "PASS"
            or row.get("labels_mounted") is not False
            or row.get("budget_canary_not_scientific_selection") is not True
            or row.get("nan_or_inf_observed", False) is not False
        ):
            raise ValueError("budget canary row violates its engineering-only contract")
        value = {
            **row,
            "model_total_seconds": float(row["model_total_seconds"]),
            "checkpoint_overhead_seconds": float(
                row.get("checkpoint_overhead_seconds", 0.0)
            ),
            "evaluation_overhead_seconds": float(
                row.get("evaluation_overhead_seconds", 0.0)
            ),
            "run_output_bytes": int(row["run_output_bytes"]),
        }
        if value["model_total_seconds"] <= 0 or value["run_output_bytes"] <= 0:
            raise ValueError("budget canary lacks positive time/storage observations")
        normalized.append(value)
    expected = {
        (dataset, role, seed)
        for dataset in MANUAL_DATASETS
        for role in CANARY_ROLES
        for seed in BUDGET_MODEL_SEEDS
    }
    observed = {
        (row["dataset"], row["path_set_role"], int(row["model_seed"]))
        for row in normalized
    }
    if observed != expected or len(normalized) != len(expected):
        raise ValueError("budget canary matrix must be exact 2×3×2=12")
    return normalized


def measured_budget(
    canary_rows: Sequence[Mapping[str, Any]],
    caches: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _validate_canaries(canary_rows)
    if set(caches) != set(MANUAL_DATASETS):
        raise ValueError("measured budget requires exactly DBLP and AMiner caches")
    reports: list[dict[str, Any]] = []
    authorization_records: list[dict[str, Any]] = []
    reserve = 1.0 + float(RESOURCE_LIMITS["retry_reserve_fraction"])
    for dataset in MANUAL_DATASETS:
        cache = caches[dataset]
        if cache.get("dataset") != dataset:
            raise ValueError("budget cache dataset mismatch")
        observed = [row for row in rows if row["dataset"] == dataset]
        p90_seconds_per_update = _p90(
            [row["model_total_seconds"] / BUDGET_CANARY_UPDATES for row in observed]
        )
        p90_checkpoint_overhead = _p90(
            [row["checkpoint_overhead_seconds"] for row in observed]
        )
        p90_evaluation_overhead = _p90(
            [row["evaluation_overhead_seconds"] for row in observed]
        )
        p90_output_bytes = int(
            math.ceil(_p90([float(row["run_output_bytes"]) for row in observed]))
        )
        candidate_count = len(cache["all_sets"]["sets"])
        high_recall_count = min(
            candidate_count,
            math.ceil(0.40 * candidate_count)
            + math.ceil(0.20 * candidate_count)
            + 1,
        )
        stage_specs = (
            ("R1_0000_TO_0300", candidate_count * 3, 300),
            ("R2_0300_TO_1000", high_recall_count * 3, 700),
            ("R3_1000_TO_2000", high_recall_count * 3, 1000),
            ("R4_2000_TO_3000", high_recall_count * 3, 1000),
            
            
            
            
            ("PRELIMINARY_FRESH_0000_TO_3000", 6, 3000),
        )
        stage_gpu_seconds: dict[str, float] = {}
        stage_critical_seconds: dict[str, float] = {}
        total_gpu_seconds = 0.0
        total_critical_seconds = 0.0
        overhead = p90_checkpoint_overhead + p90_evaluation_overhead
        for stage, run_count, updates in stage_specs:
            per_run = updates * p90_seconds_per_update + overhead
            durations = [per_run] * run_count
            gpu_seconds = sum(durations)
            critical_seconds = _lpt_makespan(durations)
            stage_gpu_seconds[stage] = gpu_seconds
            stage_critical_seconds[stage] = critical_seconds
            total_gpu_seconds += gpu_seconds
            total_critical_seconds += critical_seconds
        estimated_gpu_hours = total_gpu_seconds * reserve / 3600.0
        estimated_critical_hours = total_critical_seconds * reserve / 3600.0
        cache_bytes = int(
            cache.get(
                "cache_size_bytes",
                cache.get("manifest_size_bytes", 0),
            )
        )
        if cache_bytes <= 0:
            manifest = cache.get("manifest", {})
            files = manifest.get("files", {})
            members = files.values() if isinstance(files, Mapping) else files
            cache_bytes = sum(
                int(row.get("size_bytes", 0))
                for row in members
            )
        if cache_bytes <= 0:
            raise ValueError("budget cache lacks observed byte size")
        capture_count = candidate_count * 3 + high_recall_count * 3 * 3 + 6
        estimated_storage_bytes = int(
            math.ceil((cache_bytes + p90_output_bytes * capture_count) * reserve)
        )
        reasons: list[str] = []
        if estimated_gpu_hours > float(RESOURCE_LIMITS["hard_gpu_hours"]):
            reasons.append("GPU_HOURS")
        if estimated_critical_hours > float(
            RESOURCE_LIMITS["hard_critical_path_hours"]
        ):
            reasons.append("CRITICAL_PATH")
        if estimated_storage_bytes > int(
            RESOURCE_LIMITS["hard_remote_storage_bytes"]
        ):
            reasons.append("REMOTE_STORAGE")
        path_resolution_pass = all(
            bool(row.get("path_resolution_pass", True)) for row in observed
        )
        labels_gate_pass = all(row.get("labels_mounted") is False for row in observed)
        finite_gate_pass = all(row.get("nan_or_inf_observed", False) is False for row in observed)
        if not path_resolution_pass:
            reasons.append("PATH_RESOLUTION")
        if not labels_gate_pass:
            reasons.append("LABEL_LEAKAGE")
        if not finite_gate_pass:
            reasons.append("NAN_OR_INF")
        status = "FULL_SEARCH_AUTHORIZED" if not reasons else "MEASURED_BUDGET_BLOCKED"
        reports.append(
            {
                "dataset": dataset,
                "canary_run_count": len(observed),
                "candidate_count": candidate_count,
                "high_recall_upper_count": high_recall_count,
                "p90_seconds_per_update": p90_seconds_per_update,
                "p90_checkpoint_overhead_seconds": p90_checkpoint_overhead,
                "p90_evaluation_overhead_seconds": p90_evaluation_overhead,
                "retry_reserve_fraction": RESOURCE_LIMITS[
                    "retry_reserve_fraction"
                ],
                "estimated_gpu_hours_with_reserve": estimated_gpu_hours,
                "estimated_two_gpu_critical_path_hours_with_reserve": estimated_critical_hours,
                "estimated_remote_storage_bytes_with_reserve": estimated_storage_bytes,
                "hard_gpu_hours": RESOURCE_LIMITS["hard_gpu_hours"],
                "hard_critical_path_hours": RESOURCE_LIMITS[
                    "hard_critical_path_hours"
                ],
                "hard_remote_storage_bytes": RESOURCE_LIMITS[
                    "hard_remote_storage_bytes"
                ],
                "stage_gpu_seconds": stage_gpu_seconds,
                "stage_critical_seconds": stage_critical_seconds,
                "path_resolution_pass": path_resolution_pass,
                "labels_gate_pass": labels_gate_pass,
                "finite_gate_pass": finite_gate_pass,
                "hard_limit_reasons": reasons,
                "status": status,
            }
        )
        authorization_records.append(
            {
                "dataset": dataset,
                "status": status,
                "authorized": status == "FULL_SEARCH_AUTHORIZED",
                "hard_limit_reasons": reasons,
                "candidate_library_reduced": False,
                "algorithm_changed": False,
                "budget_canaries_used_for_scientific_selection": False,
            }
        )
    basis = {
        "measured_budget_rows_identity": canonical_json_hash(reports),
        "records": authorization_records,
        "active_datasets": [
            row["dataset"] for row in authorization_records if row["authorized"]
        ],
        "blocked_datasets": [
            row["dataset"] for row in authorization_records if not row["authorized"]
        ],
        "per_dataset_gate": True,
        "candidate_library_reduced": False,
        "algorithm_changed": False,
        "budget_canaries_used_for_scientific_selection": False,
    }
    authorization = envelope(
        "metachorus.v53.m7f-h1.search-authorization.v1",
        "PASS",
        basis,
    )
    return reports, authorization


def write_budget_outputs(
    result_root: str | Path,
    *,
    canary_rows: Sequence[Mapping[str, Any]],
    caches: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(result_root)
    normalized = _validate_canaries(canary_rows)
    reports, authorization = measured_budget(normalized, caches)
    write_csv_once(
        root / "DBLP_AMINER_BUDGET_CANARY_RUNS.csv",
        normalized,
        fieldnames=list(normalized[0]),
    )
    write_csv_once(
        root / "DBLP_AMINER_MEASURED_BUDGET.csv",
        reports,
        fieldnames=list(reports[0]),
    )
    write_json_once(root / "DBLP_AMINER_SEARCH_AUTHORIZATION.json", authorization)
    return reports, authorization


__all__ = [
    "CANARY_ROLES",
    "measured_budget",
    "write_budget_outputs",
]
