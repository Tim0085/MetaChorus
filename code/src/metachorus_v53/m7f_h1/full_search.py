"""Connect budget planning to the multi-stage search workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from metachorus_v53.m6f_m7f_fast.constants import (
    DATASET_NAMES as FAST_DATASET_NAMES,
    EVAL_MODEL_SEEDS,
    SEARCH_MODEL_SEEDS,
)
from metachorus_v53.m6f_m7f_fast.planning import (
    build_search_plan,
    load_cache_resolution,
)
from metachorus_v53.m6f_m7f_fast.stages import validate_plan as validate_fast_plan
from metachorus_v53.m6f_m7f_fast.training import (
    validate_run_marker as validate_fast_run_marker,
)

from .common import (
    canonical_json_hash,
    envelope,
    read_csv,
    read_csv_allow_empty,
    read_json,
    safe_relative,
    sha256_file,
    write_csv_once,
    write_json_once,
)
from .constants import MANUAL_DATASETS, RESOURCE_LIMITS


R1_PLAN = "FAST_SEARCH_R1_RUN_PLAN.csv"
R1_SEAL = "FAST_SEARCH_R1_RUN_PLAN_SEAL.json"
BRIDGE_NAME = "M7F_H1_FULL_SEARCH_AUTHORIZATION_BRIDGE.json"
STATUS_NAME = "M7F_H1_FULL_SEARCH_STATUS.json"
FAST_STATUS_NAME = "FAST_DATASET_EXECUTION_STATUS.json"

SEARCH_STAGE_SPECS = (
    (
        "FULL_SEARCH_R1",
        300,
        "FAST_SEARCH_R1_RUN_PLAN.csv",
        "FAST_SEARCH_R1_RUN_PLAN_SEAL.json",
        "SEALED_AFTER_CANARY_BEFORE_R1_GPU_TRAINING",
    ),
    (
        "FULL_SEARCH_R2",
        1000,
        "FAST_SEARCH_RUN_PLAN_001000.csv",
        "FAST_SEARCH_RUN_PLAN_001000_SEAL.json",
        "SEALED_BEFORE_SEARCH_CONTINUATION",
    ),
    (
        "FULL_SEARCH_R3",
        2000,
        "FAST_SEARCH_RUN_PLAN_002000.csv",
        "FAST_SEARCH_RUN_PLAN_002000_SEAL.json",
        "SEALED_BEFORE_SEARCH_CONTINUATION",
    ),
    (
        "FULL_SEARCH_R4",
        3000,
        "FAST_R4_RUN_PLAN.csv",
        "FAST_R4_RUN_PLAN_SEAL.json",
        "SEALED_BEFORE_SEARCH_CONTINUATION",
    ),
)

SELECTION_FIELDS = (
    "dataset",
    "status",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "path_ids",
    "selection_decision_identity",
    "manual_safety_status",
)
METRIC_FIELDS = (
    "dataset",
    "logical_role",
    "availability",
    "physical_run_id",
    "candidate_set_id",
    "candidate_set_hash",
    "model_seed",
    "endpoint_update",
    "Micro_F1",
    "Macro_F1",
    "NMI",
    "ARI",
    "F_cls",
    "F_clu",
    "F_joint",
    "assignment_stability_mean",
    "classification_fit_role",
    "classification_metric_role",
    "clustering_fit_role",
    "clustering_metric_role",
    "confirmatory_claim_allowed",
    "status",
)
SUMMARY_FIELDS = (
    "dataset",
    "logical_role",
    "candidate_set_hash",
    "model_seed_count",
    "Micro_F1_mean",
    "Macro_F1_mean",
    "NMI_mean",
    "ARI_mean",
    "F_cls_mean",
    "F_clu_mean",
    "F_joint_mean",
    "Micro_F1_sample_sd",
    "Macro_F1_sample_sd",
    "NMI_sample_sd",
    "ARI_sample_sd",
    "F_cls_sample_sd",
    "F_clu_sample_sd",
    "F_joint_sample_sd",
    "evidence_level",
    "status",
)
COMPARISON_FIELDS = (
    "dataset",
    "comparison",
    "model_seed",
    "candidate_set_hash",
    "manual_set_hash",
    "delta_Micro_F1",
    "delta_Macro_F1",
    "delta_NMI",
    "delta_ARI",
    "delta_F_cls",
    "delta_F_clu",
    "delta_F_joint",
    "decision_status",
    "confirmatory_claim_allowed",
    "status",
)
RUN_REGISTRY_FIELDS = (
    "run_boundary_id",
    "stage",
    "physical_run_id",
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "model_seed",
    "continuation_from_update",
    "endpoint_update",
    "optimizer_steps",
    "model_total_seconds",
    "marker_relative_path",
    "marker_identity",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "representation_relative_path",
    "representation_sha256",
    "labels_mounted",
    "status",
)
RETENTION_FIELDS = (
    "run_boundary_id",
    "physical_run_id",
    "dataset",
    "stage",
    "artifact_role",
    "update",
    "remote_relative_path",
    "size_bytes",
    "sha256",
    "retention_status",
    "included_in_minimal_package",
    "download_authorized",
)

BUDGET_STAGE_NAMES = (
    "R1_0000_TO_0300",
    "R2_0300_TO_1000",
    "R3_1000_TO_2000",
    "R4_2000_TO_3000",
    "PRELIMINARY_FRESH_0000_TO_3000",
)

SEARCH_METRIC_FIELDS = (
    "physical_run_id",
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "model_seed",
    "endpoint_update",
    "marker_identity",
    "representation_sha256",
    "model_total_seconds",
    "peak_gpu_reserved_bytes",
    "Micro_F1",
    "Macro_F1",
    "NMI",
    "ARI",
    "F_cls",
    "F_clu",
    "F_joint",
    "assignment_stability_mean",
    "prelim_locked_report_access",
    "status",
)


def _strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"invalid boolean in {field}")


def _normalized_budget_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    integer_fields = (
        "canary_run_count",
        "candidate_count",
        "high_recall_upper_count",
        "estimated_remote_storage_bytes_with_reserve",
        "hard_remote_storage_bytes",
    )
    float_fields = (
        "p90_seconds_per_update",
        "p90_checkpoint_overhead_seconds",
        "p90_evaluation_overhead_seconds",
        "retry_reserve_fraction",
        "estimated_gpu_hours_with_reserve",
        "estimated_two_gpu_critical_path_hours_with_reserve",
        "hard_gpu_hours",
        "hard_critical_path_hours",
    )
    for raw in rows:
        row = dict(raw)
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
        for field in (
            "path_resolution_pass",
            "labels_gate_pass",
            "finite_gate_pass",
        ):
            row[field] = _strict_bool(row[field], field=field)
        for field in ("stage_gpu_seconds", "stage_critical_seconds"):
            value = row[field]
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, Mapping):
                raise ValueError(f"invalid measured-budget mapping in {field}")
            row[field] = {str(key): float(item) for key, item in value.items()}
        reasons = row["hard_limit_reasons"]
        if isinstance(reasons, str):
            reasons = json.loads(reasons)
        if not isinstance(reasons, list):
            raise ValueError("invalid measured-budget hard-limit reasons")
        row["hard_limit_reasons"] = list(map(str, reasons))
        normalized.append(row)
    return normalized


def _normalized_search_metric_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recover the exact typed rows used to seal a search-metric CSV."""

    normalized: list[dict[str, Any]] = []
    integer_fields = (
        "candidate_set_ordinal",
        "model_seed",
        "endpoint_update",
        "peak_gpu_reserved_bytes",
    )
    float_fields = (
        "model_total_seconds",
        "Micro_F1",
        "Macro_F1",
        "NMI",
        "ARI",
        "F_cls",
        "F_clu",
        "F_joint",
        "assignment_stability_mean",
    )
    for raw in rows:
        if set(raw) != set(SEARCH_METRIC_FIELDS):
            raise ValueError("full-search metric CSV schema changed")
        row = {field: raw[field] for field in SEARCH_METRIC_FIELDS}
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError("full-search metric CSV contains non-finite values")
        row["prelim_locked_report_access"] = _strict_bool(
            row["prelim_locked_report_access"],
            field="prelim_locked_report_access",
        )
        normalized.append(row)
    return normalized


def normalize_full_search_registry_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a full-search registry across the CSV serialization boundary."""

    output: list[dict[str, Any]] = []
    for raw in rows:
        if set(raw) != set(RUN_REGISTRY_FIELDS):
            raise ValueError("full-search registry schema changed")
        row = {field: raw[field] for field in RUN_REGISTRY_FIELDS}
        for field in (
            "model_seed",
            "continuation_from_update",
            "endpoint_update",
            "optimizer_steps",
        ):
            row[field] = int(row[field])
        row["model_total_seconds"] = float(row["model_total_seconds"])
        row["labels_mounted"] = _strict_bool(
            row["labels_mounted"], field="labels_mounted"
        )
        output.append(row)
    return sorted(
        output,
        key=lambda row: (str(row["stage"]), str(row["run_boundary_id"])),
    )


def normalize_full_search_retention_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize retained-artifact rows across the CSV serialization boundary."""

    output: list[dict[str, Any]] = []
    for raw in rows:
        if set(raw) != set(RETENTION_FIELDS):
            raise ValueError("full-search retention schema changed")
        row = {field: raw[field] for field in RETENTION_FIELDS}
        for field in ("update", "size_bytes"):
            row[field] = int(row[field])
        for field in ("included_in_minimal_package", "download_authorized"):
            row[field] = _strict_bool(row[field], field=field)
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            str(row["stage"]),
            str(row["run_boundary_id"]),
            str(row["artifact_role"]),
        ),
    )


def validate_search_authorization(
    value: Mapping[str, Any],
    *,
    measured_budget_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the exact per-dataset H1 budget authorization envelope."""

    authorization = dict(value)
    basis = authorization.get("basis", {})
    records = list(basis.get("records", []))
    by_dataset = {str(row.get("dataset", "")): dict(row) for row in records}
    active = [str(name) for name in basis.get("active_datasets", [])]
    blocked = [str(name) for name in basis.get("blocked_datasets", [])]
    if (
        authorization.get("schema_version")
        != "metachorus.v53.m7f-h1.search-authorization.v1"
        or authorization.get("status") != "PASS"
        or authorization.get("identity") != canonical_json_hash(basis)
        or len(records) != len(MANUAL_DATASETS)
        or set(by_dataset) != set(MANUAL_DATASETS)
        or len(active) != len(set(active))
        or len(blocked) != len(set(blocked))
        or set(active) & set(blocked)
        or set(active) | set(blocked) != set(MANUAL_DATASETS)
        or any(bool(row.get("authorized")) != (dataset in active) for dataset, row in by_dataset.items())
        or any(
            row.get("status")
            != ("FULL_SEARCH_AUTHORIZED" if bool(row.get("authorized")) else "MEASURED_BUDGET_BLOCKED")
            for row in records
        )
        or basis.get("candidate_library_reduced") is not False
        or basis.get("algorithm_changed") is not False
        or basis.get("budget_canaries_used_for_scientific_selection") is not False
        or basis.get("per_dataset_gate") is not True
    ):
        raise ValueError("DBLP/AMiner search authorization identity is invalid")
    if measured_budget_rows is not None:
        reports = _normalized_budget_rows(measured_budget_rows)
        report_by_dataset = {
            str(row.get("dataset", "")): row for row in reports
        }
        if (
            len(reports) != len(MANUAL_DATASETS)
            or list(report_by_dataset) != list(MANUAL_DATASETS)
            or basis.get("measured_budget_rows_identity")
            != canonical_json_hash(reports)
        ):
            raise ValueError("measured-budget rows are not bound to the authorization")
        for dataset in MANUAL_DATASETS:
            report = report_by_dataset[dataset]
            record = by_dataset[dataset]
            reasons: list[str] = []
            if float(report["estimated_gpu_hours_with_reserve"]) > float(
                RESOURCE_LIMITS["hard_gpu_hours"]
            ):
                reasons.append("GPU_HOURS")
            if float(
                report["estimated_two_gpu_critical_path_hours_with_reserve"]
            ) > float(RESOURCE_LIMITS["hard_critical_path_hours"]):
                reasons.append("CRITICAL_PATH")
            if int(report["estimated_remote_storage_bytes_with_reserve"]) > int(
                RESOURCE_LIMITS["hard_remote_storage_bytes"]
            ):
                reasons.append("REMOTE_STORAGE")
            if report["path_resolution_pass"] is not True:
                reasons.append("PATH_RESOLUTION")
            if report["labels_gate_pass"] is not True:
                reasons.append("LABEL_LEAKAGE")
            if report["finite_gate_pass"] is not True:
                reasons.append("NAN_OR_INF")
            expected_status = (
                "FULL_SEARCH_AUTHORIZED"
                if not reasons
                else "MEASURED_BUDGET_BLOCKED"
            )
            numeric_values = [
                float(report[field])
                for field in (
                    "p90_seconds_per_update",
                    "p90_checkpoint_overhead_seconds",
                    "p90_evaluation_overhead_seconds",
                    "estimated_gpu_hours_with_reserve",
                    "estimated_two_gpu_critical_path_hours_with_reserve",
                )
            ]
            if (
                report.get("status") != expected_status
                or record.get("status") != expected_status
                or bool(record.get("authorized"))
                != (expected_status == "FULL_SEARCH_AUTHORIZED")
                or report["hard_limit_reasons"] != reasons
                or list(record.get("hard_limit_reasons", [])) != reasons
                or int(report["canary_run_count"]) != 6
                or int(report["candidate_count"]) <= 0
                or not 0 < int(report["high_recall_upper_count"]) <= int(report["candidate_count"])
                or set(report["stage_gpu_seconds"]) != set(BUDGET_STAGE_NAMES)
                or set(report["stage_critical_seconds"]) != set(BUDGET_STAGE_NAMES)
                or any(value < 0 or not math.isfinite(value) for value in numeric_values)
                or any(
                    value < 0 or not math.isfinite(value)
                    for value in (
                        *report["stage_gpu_seconds"].values(),
                        *report["stage_critical_seconds"].values(),
                    )
                )
                or float(report["retry_reserve_fraction"])
                != float(RESOURCE_LIMITS["retry_reserve_fraction"])
                or float(report["hard_gpu_hours"])
                != float(RESOURCE_LIMITS["hard_gpu_hours"])
                or float(report["hard_critical_path_hours"])
                != float(RESOURCE_LIMITS["hard_critical_path_hours"])
                or int(report["hard_remote_storage_bytes"])
                != int(RESOURCE_LIMITS["hard_remote_storage_bytes"])
            ):
                raise ValueError("measured-budget authorization semantics changed")
    return authorization, by_dataset


def build_authorized_r1_plan(
    *,
    authorization: Mapping[str, Any],
    caches: Mapping[str, Mapping[str, Any]],
    source_commit: str,
    source_tree: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    """Create the exact active-dataset R1 plan and FAST compatibility status."""

    validated, by_dataset = validate_search_authorization(authorization)
    if set(caches) != set(FAST_DATASET_NAMES):
        raise ValueError("H1 full search requires all three frozen cache identities")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_commit)) or not re.fullmatch(
        r"[0-9a-f]{40}", str(source_tree)
    ):
        raise ValueError("H1 full-search source commit/tree is not canonical")
    active = tuple(
        dataset
        for dataset in FAST_DATASET_NAMES
        if dataset in validated["basis"]["active_datasets"]
    )
    compatibility_records: list[dict[str, Any]] = []
    for dataset in FAST_DATASET_NAMES:
        if dataset in active:
            compatibility_records.append(
                {
                    "dataset": dataset,
                    "status": "M7F_H1_FULL_SEARCH_AUTHORIZED",
                    "reason": "MEASURED_H1_BUDGET_PASS",
                    "h1_authorization_record_identity": canonical_json_hash(by_dataset[dataset]),
                    "compatibility_only": True,
                }
            )
        elif dataset == "imdb":
            compatibility_records.append(
                {
                    "dataset": dataset,
                    "status": "FAST_SEARCH_BUDGET_BLOCKED",
                    "reason": "OUT_OF_SCOPE_H1_IMDB_TRAJECTORY_COMPATIBILITY_ONLY",
                    "h1_authorization_record_identity": "",
                    "compatibility_only": True,
                }
            )
        else:
            compatibility_records.append(
                {
                    "dataset": dataset,
                    "status": "FAST_SEARCH_BUDGET_BLOCKED",
                    "reason": "M7F_H1_MEASURED_BUDGET_BLOCKED",
                    "h1_authorization_record_identity": canonical_json_hash(by_dataset[dataset]),
                    "compatibility_only": True,
                }
            )
    fast_status = envelope(
        "metachorus.v53.m6f-m7f.dataset-execution-status.v1",
        "PARTIAL" if active else "BLOCKED",
        {
            "records": compatibility_records,
            "active_datasets": list(active),
            "blocked_datasets": [name for name in FAST_DATASET_NAMES if name not in active],
            "h1_search_authorization_identity": validated["identity"],
            "compatibility_scope": "M7F_H1_CONDITIONAL_DBLP_AMINER_FULL_SEARCH",
            "imdb_scientific_status_is_governed_by_h1_trajectory_not_this_compatibility_record": True,
        },
    )
    if not active:
        return [], None, fast_status
    draft, draft_seal = build_search_plan(caches)
    rows = [
        {
            **row,
            "stage": "SEARCH",
            "all_sets_identity": str(caches[str(row["dataset"])]["all_sets_identity"]),
            "source_commit": str(source_commit),
            "source_tree": str(source_tree),
            "status": "AUTHORIZED",
            "plan_seal_identity": "",
        }
        for row in draft
        if str(row["dataset"]) in active
    ]
    expected = sum(len(caches[name]["all_sets"]["sets"]) for name in active) * len(SEARCH_MODEL_SEEDS)
    if len(rows) != expected or len({str(row["physical_run_id"]) for row in rows}) != expected:
        raise RuntimeError("H1 authorized R1 is not the exact candidate-by-seed matrix")
    unsealed = [
        {key: value for key, value in row.items() if key != "plan_seal_identity"}
        for row in rows
    ]
    candidate_hashes_by_dataset = {
        dataset: sorted(
            {
                str(row["candidate_set_hash"])
                for row in rows
                if str(row["dataset"]) == dataset
            }
        )
        for dataset in active
    }
    seal = envelope(
        "metachorus.v53.m7f-h1.full-search-r1-plan-seal.v1",
        "SEALED_AFTER_CANARY_BEFORE_R1_GPU_TRAINING",
        {
            "row_count": len(rows),
            "rows_identity": canonical_json_hash(unsealed),
            "draft_plan_seal_identity": draft_seal["identity"],
            "search_authorization_identity": validated["identity"],
            "fast_dataset_execution_status_identity": fast_status["identity"],
            "source_commit": str(source_commit),
            "source_tree": str(source_tree),
            "active_datasets": list(active),
            "blocked_h1_datasets": list(validated["basis"]["blocked_datasets"]),
            "out_of_scope_datasets": ["imdb"],
            "endpoint_update": 300,
            "search_model_seeds": list(SEARCH_MODEL_SEEDS),
            "candidate_count_by_dataset": {
                dataset: len(candidate_hashes_by_dataset[dataset])
                for dataset in active
            },
            "candidate_hashes_identity_by_dataset": {
                dataset: canonical_json_hash(candidate_hashes_by_dataset[dataset])
                for dataset in active
            },
            "candidate_bank_identity_by_dataset": {
                dataset: str(caches[dataset]["all_sets_identity"])
                for dataset in active
            },
            "candidate_library_reduced": False,
            "algorithm_changed": False,
            "labels_mounted": False,
            "prelim_locked_report_access": False,
        },
    )
    return [
        {**row, "plan_seal_identity": seal["identity"]} for row in rows
    ], seal, fast_status


def prepare_conditional_full_search(
    *, result_root: str | Path, cache_parent: str | Path
) -> dict[str, Any]:
    """Publish the deterministic R1 authorization bridge after budget closure."""

    result = Path(result_root).resolve(strict=True)
    if result.is_symlink():
        raise ValueError("M7F-H1 result root cannot be symlinked")
    identity = read_json(result / "M7F_H1_GIT_AND_CONFIG_IDENTITY.json")
    identity_basis = identity.get("basis", {})
    if identity.get("identity") != canonical_json_hash(identity_basis):
        raise ValueError("M7F-H1 git/config identity changed")
    authorization = read_json(result / "DBLP_AMINER_SEARCH_AUTHORIZATION.json")
    validate_search_authorization(
        authorization,
        measured_budget_rows=read_csv(result / "DBLP_AMINER_MEASURED_BUDGET.csv"),
    )
    caches = {
        name: load_cache_resolution(Path(cache_parent) / name)
        for name in FAST_DATASET_NAMES
    }
    rows, seal, fast_status = build_authorized_r1_plan(
        authorization=authorization,
        caches=caches,
        source_commit=str(identity_basis["source_commit"]),
        source_tree=str(identity_basis["source_tree"]),
    )
    write_json_once(result / FAST_STATUS_NAME, fast_status)
    if seal is not None:
        write_csv_once(result / R1_PLAN, rows, fieldnames=list(rows[0]))
        write_json_once(result / R1_SEAL, seal)
    validated, _ = validate_search_authorization(authorization)
    bridge = envelope(
        "metachorus.v53.m7f-h1.full-search-authorization-bridge.v1",
        "READY_FOR_FULL_SEARCH_R1" if rows else "NO_DATASET_AUTHORIZED",
        {
            "search_authorization_identity": validated["identity"],
            "active_datasets": list(validated["basis"]["active_datasets"]),
            "blocked_datasets": list(validated["basis"]["blocked_datasets"]),
            "out_of_scope_datasets": ["imdb"],
            "fast_dataset_execution_status_identity": fast_status["identity"],
            "r1_plan_seal_identity": None if seal is None else seal["identity"],
            "source_commit": identity_basis["source_commit"],
            "source_tree": identity_basis["source_tree"],
            "candidate_library_reduced": False,
            "algorithm_changed": False,
            "budget_canaries_used_for_scientific_selection": False,
            "labels_mounted_during_training": False,
            "prelim_locked_report_used_for_selection": False,
        },
    )
    write_json_once(result / BRIDGE_NAME, bridge)
    return bridge


def _governed_file(result: Path, relative: str) -> Path:
    safe = safe_relative(relative)
    cursor = result
    for part in safe.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("full-search evidence contains a symlink")
    source = cursor.resolve(strict=True)
    try:
        source.relative_to(result)
    except ValueError as exc:
        raise ValueError("full-search evidence escapes the H1 result root") from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _validated_json(result: Path, name: str, *, schema: str | None = None, status: str | None = None) -> dict[str, Any]:
    value = read_json(_governed_file(result, name))
    basis = value.get("basis", {})
    if (
        value.get("identity") != canonical_json_hash(basis)
        or (schema is not None and value.get("schema_version") != schema)
        or (status is not None and value.get("status") != status)
    ):
        raise ValueError(f"full-search sealed JSON is invalid: {name}")
    return value


def _validated_search_metric_seals(
    result: Path,
    *,
    fast_status: Mapping[str, Any],
    active_datasets: Sequence[str],
) -> dict[int, dict[str, Any]]:
    expected_active = list(active_datasets)
    expected_blocked = [
        dataset for dataset in FAST_DATASET_NAMES if dataset not in expected_active
    ]
    plan_seal_identities = {
        endpoint: _validated_json(
            result,
            seal_name,
            status=seal_status,
        )["identity"]
        for _, endpoint, _, seal_name, seal_status in SEARCH_STAGE_SPECS
    }
    output: dict[int, dict[str, Any]] = {}
    for endpoint in (300, 1000, 2000, 3000):
        metric_rows = _normalized_search_metric_rows(
            read_csv(
                _governed_file(
                    result, f"FAST_SEARCH_METRICS_{endpoint:06d}.csv"
                )
            )
        )
        cohorts: dict[tuple[str, str], set[int]] = {}
        for row in metric_rows:
            key = (str(row["dataset"]), str(row["candidate_set_hash"]))
            cohorts.setdefault(key, set()).add(int(row["model_seed"]))
        value = _validated_json(
            result,
            f"FAST_SEARCH_METRICS_{endpoint:06d}_SEAL.json",
            schema="metachorus.v53.m6f-m7f.search-metrics-seal.v1",
            status="PASS",
        )
        basis = value["basis"]
        if (
            int(basis.get("endpoint_update", -1)) != endpoint
            or basis.get("plan_seal_identity")
            != plan_seal_identities[endpoint]
            or list(basis.get("datasets", [])) != expected_active
            or list(basis.get("blocked_datasets", [])) != expected_blocked
            or basis.get("dataset_execution_status_identity")
            != fast_status.get("identity")
            or list(map(int, basis.get("model_seeds", [])))
            != list(SEARCH_MODEL_SEEDS)
            or basis.get("metric_role") != "GLOBAL_SEARCH_POOL_ONLY"
            or basis.get("prelim_locked_report_access") is not False
            or int(basis.get("row_count", -1)) != len(metric_rows)
            or basis.get("rows_identity") != canonical_json_hash(metric_rows)
            or not metric_rows
            or len({str(row["physical_run_id"]) for row in metric_rows})
            != len(metric_rows)
            or {str(row["dataset"]) for row in metric_rows}
            != set(expected_active)
            or any(int(row["endpoint_update"]) != endpoint for row in metric_rows)
            or any(row["prelim_locked_report_access"] is not False for row in metric_rows)
            or any(row["status"] != "PASS" for row in metric_rows)
            or any(seeds != set(SEARCH_MODEL_SEEDS) for seeds in cohorts.values())
        ):
            raise ValueError("full-search metric seal violated the GLOBAL-only boundary")
        output[endpoint] = value
    return output


def _plan_candidate_hashes(
    rows: Sequence[Mapping[str, Any]],
    *,
    active_datasets: Sequence[str],
    endpoint: int,
) -> dict[str, set[str]]:
    active = tuple(active_datasets)
    if {str(row.get("dataset", "")) for row in rows} != set(active):
        raise ValueError("full-search stage dataset subset changed")
    output: dict[str, set[str]] = {}
    for dataset in active:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        cohorts: dict[str, set[int]] = {}
        seen: set[tuple[str, int]] = set()
        for row in dataset_rows:
            digest = str(row.get("candidate_set_hash", ""))
            seed = int(row.get("model_seed", -1))
            key = (digest, seed)
            if (
                int(row.get("endpoint_update", -1)) != int(endpoint)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or key in seen
            ):
                raise ValueError("full-search stage candidate/seed matrix changed")
            seen.add(key)
            cohorts.setdefault(digest, set()).add(seed)
        if not cohorts or any(
            seeds != set(SEARCH_MODEL_SEEDS) for seeds in cohorts.values()
        ):
            raise ValueError("full-search stage is not the exact three-search-seed matrix")
        output[dataset] = set(cohorts)
    return output


def _record_candidate_hashes(
    records: Sequence[Mapping[str, Any]],
    *,
    active_datasets: Sequence[str],
    expected_status: str,
) -> dict[str, set[str]]:
    active = tuple(active_datasets)
    output = {dataset: set() for dataset in active}
    seen: set[tuple[str, str]] = set()
    for row in records:
        dataset = str(row.get("dataset", ""))
        digest = str(row.get("candidate_set_hash", ""))
        key = (dataset, digest)
        if (
            dataset not in output
            or row.get("status") != expected_status
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or key in seen
        ):
            raise ValueError("full-search selection registry is not canonical")
        seen.add(key)
        output[dataset].add(digest)
    if any(not values for values in output.values()):
        raise ValueError("full-search selection registry is empty for an active dataset")
    return output


def _validate_search_candidate_chain(
    *,
    stage_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    active_datasets: Sequence[str],
    expected_r1_candidate_counts: Mapping[str, int],
    high_recall_records: Sequence[Mapping[str, Any]],
    r2_no_pruning_records: Sequence[Mapping[str, Any]],
    robust_shortlist_records: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, set[str]]]:
    """Prove all-set -> high-recall -> no-pruning -> shortlist inheritance."""

    if set(stage_rows) != {300, 1000, 2000, 3000}:
        raise ValueError("full-search stage endpoint set changed")
    active = tuple(active_datasets)
    hashes = {
        endpoint: _plan_candidate_hashes(
            stage_rows[endpoint],
            active_datasets=active,
            endpoint=endpoint,
        )
        for endpoint in (300, 1000, 2000, 3000)
    }
    if set(expected_r1_candidate_counts) != set(active) or any(
        len(hashes[300][dataset]) != int(expected_r1_candidate_counts[dataset])
        for dataset in active
    ):
        raise ValueError("R1 is not the exact measured full candidate bank")
    high_recall = _record_candidate_hashes(
        high_recall_records,
        active_datasets=active,
        expected_status="SELECTED_FOR_R2_R3",
    )
    r2_no_pruning = _record_candidate_hashes(
        r2_no_pruning_records,
        active_datasets=active,
        expected_status="R2_NO_PRUNING_PRESERVED_FOR_R3",
    )
    robust_shortlist = _record_candidate_hashes(
        robust_shortlist_records,
        active_datasets=active,
        expected_status="SELECTED_FOR_R4",
    )
    for dataset in active:
        if (
            not high_recall[dataset].issubset(hashes[300][dataset])
            or hashes[1000][dataset] != high_recall[dataset]
            or hashes[2000][dataset] != hashes[1000][dataset]
            or r2_no_pruning[dataset] != hashes[2000][dataset]
            or not robust_shortlist[dataset].issubset(hashes[2000][dataset])
            or hashes[3000][dataset] != robust_shortlist[dataset]
        ):
            raise ValueError("R1-R4 candidate inheritance semantics changed")
    return hashes


def collect_full_search_run_evidence(
    result: Path, active: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_fast = [name for name in FAST_DATASET_NAMES if name in set(active)]
    measured_rows = _normalized_budget_rows(
        read_csv(_governed_file(result, "DBLP_AMINER_MEASURED_BUDGET.csv"))
    )
    authorization, _ = validate_search_authorization(
        read_json(_governed_file(result, "DBLP_AMINER_SEARCH_AUTHORIZATION.json")),
        measured_budget_rows=measured_rows,
    )
    if active_fast != [
        name
        for name in FAST_DATASET_NAMES
        if name in authorization["basis"]["active_datasets"]
    ]:
        raise ValueError("full-search evidence differs from measured authorization")
    fast_status = _validated_json(
        result,
        FAST_STATUS_NAME,
        schema="metachorus.v53.m6f-m7f.dataset-execution-status.v1",
    )
    if (
        fast_status["basis"].get("active_datasets") != active_fast
        or fast_status["basis"].get("h1_search_authorization_identity")
        != authorization["identity"]
    ):
        raise ValueError("full-search FAST compatibility identity changed")

    stage_rows: dict[int, list[dict[str, Any]]] = {}
    stage_seals: dict[int, dict[str, Any]] = {}
    for _, endpoint, plan_name, seal_name, seal_status in SEARCH_STAGE_SPECS:
        rows, seal = validate_fast_plan(
            result / plan_name,
            result / seal_name,
            expected_status=seal_status,
            expected_stage="SEARCH",
        )
        stage_rows[endpoint] = rows
        stage_seals[endpoint] = seal

    report_by_dataset = {
        str(row["dataset"]): row for row in measured_rows
    }
    high_recall = _validated_json(
        result,
        "FAST_R1_HIGH_RECALL_POOL.json",
        schema="metachorus.v53.m6f-m7f.r1-high-recall-pool.v1",
        status="PASS",
    )
    robust_shortlist = _validated_json(
        result,
        "FAST_R3_ROBUST_SHORTLIST.json",
        schema="metachorus.v53.m6f-m7f.r3-robust-shortlist.v1",
        status="PASS",
    )
    candidate_hashes = _validate_search_candidate_chain(
        stage_rows=stage_rows,
        active_datasets=active_fast,
        expected_r1_candidate_counts={
            dataset: int(report_by_dataset[dataset]["candidate_count"])
            for dataset in active_fast
        },
        high_recall_records=list(high_recall["basis"].get("records", [])),
        r2_no_pruning_records=read_csv(
            _governed_file(result, "FAST_R2_NO_PRUNING_REGISTRY.csv")
        ),
        robust_shortlist_records=list(
            robust_shortlist["basis"].get("records", [])
        ),
    )

    r1_seal = stage_seals[300]
    r1_basis = r1_seal["basis"]
    candidate_counts = {
        str(dataset): int(count)
        for dataset, count in r1_basis.get("candidate_count_by_dataset", {}).items()
    }
    candidate_hash_identities = {
        str(dataset): str(identity)
        for dataset, identity in r1_basis.get(
            "candidate_hashes_identity_by_dataset", {}
        ).items()
    }
    candidate_bank_identities = {
        str(dataset): str(identity)
        for dataset, identity in r1_basis.get(
            "candidate_bank_identity_by_dataset", {}
        ).items()
    }
    if (
        r1_seal.get("schema_version")
        != "metachorus.v53.m7f-h1.full-search-r1-plan-seal.v1"
        or r1_basis.get("search_authorization_identity")
        != authorization["identity"]
        or r1_basis.get("fast_dataset_execution_status_identity")
        != fast_status["identity"]
        or list(r1_basis.get("active_datasets", [])) != active_fast
        or list(r1_basis.get("blocked_h1_datasets", []))
        != list(authorization["basis"]["blocked_datasets"])
        or list(r1_basis.get("out_of_scope_datasets", [])) != ["imdb"]
        or int(r1_basis.get("endpoint_update", -1)) != 300
        or list(map(int, r1_basis.get("search_model_seeds", [])))
        != list(SEARCH_MODEL_SEEDS)
        or set(candidate_counts) != set(active_fast)
        or set(candidate_hash_identities) != set(active_fast)
        or set(candidate_bank_identities) != set(active_fast)
        or any(
            candidate_counts[dataset] != len(candidate_hashes[300][dataset])
            or candidate_hash_identities[dataset]
            != canonical_json_hash(sorted(candidate_hashes[300][dataset]))
            or not re.fullmatch(
                r"[0-9a-f]{64}", candidate_bank_identities[dataset]
            )
            for dataset in active_fast
        )
        or any(
            str(row.get("all_sets_identity", ""))
            != candidate_bank_identities[str(row["dataset"])]
            or str(row.get("source_commit", ""))
            != str(r1_basis.get("source_commit", ""))
            or str(row.get("source_tree", ""))
            != str(r1_basis.get("source_tree", ""))
            for row in stage_rows[300]
        )
        or not re.fullmatch(r"[0-9a-f]{40}", str(r1_basis.get("source_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(r1_basis.get("source_tree", "")))
        or r1_basis.get("candidate_library_reduced") is not False
        or r1_basis.get("algorithm_changed") is not False
        or r1_basis.get("labels_mounted") is not False
        or r1_basis.get("prelim_locked_report_access") is not False
    ):
        raise ValueError("H1 R1 plan is not bound to the exact frozen candidate bank")

    expected_blocked_fast = [
        name for name in FAST_DATASET_NAMES if name not in active_fast
    ]
    prior_identity = r1_seal["identity"]
    prior_endpoint = 300
    for endpoint in (1000, 2000, 3000):
        seal = stage_seals[endpoint]
        basis = seal["basis"]
        if (
            seal.get("schema_version")
            != "metachorus.v53.m6f-m7f.search-continuation-plan-seal.v1"
            or basis.get("prior_plan_seal_identity") != prior_identity
            or int(basis.get("continuation_from_update", -1)) != prior_endpoint
            or int(basis.get("endpoint_update", -1)) != endpoint
            or list(basis.get("active_datasets", [])) != active_fast
            or list(basis.get("blocked_datasets", [])) != expected_blocked_fast
            or basis.get("dataset_execution_status_identity")
            != fast_status["identity"]
            or basis.get("labels_mounted") is not False
            or basis.get("prelim_locked_report_access") is not False
            or any(
                str(row.get("all_sets_identity", ""))
                != candidate_bank_identities[str(row["dataset"])]
                for row in stage_rows[endpoint]
            )
        ):
            raise ValueError("R1-R4 continuation seal chain changed")
        prior_identity = seal["identity"]
        prior_endpoint = endpoint

    registry: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []

    def collect(
        *,
        logical_stage: str,
        engine_stage: str,
        endpoint: int,
        plan_name: str,
        seal_name: str,
        seal_status: str,
        base: str,
    ) -> None:
        rows, _ = validate_fast_plan(
            result / plan_name,
            result / seal_name,
            expected_status=seal_status,
            expected_stage=engine_stage,
        )
        if {str(row["dataset"]) for row in rows} != set(active):
            raise ValueError("full-search plan differs from its H1 authorized dataset subset")
        expected_seeds = (
            set(EVAL_MODEL_SEEDS)
            if engine_stage == "PRELIMINARY"
            else set(SEARCH_MODEL_SEEDS)
        )
        cohorts: dict[tuple[str, str], set[int]] = {}
        for row in rows:
            if int(row.get("endpoint_update", -1)) != endpoint:
                raise ValueError("full-search plan endpoint changed")
            key = (str(row["dataset"]), str(row["candidate_set_hash"]))
            cohorts.setdefault(key, set()).add(int(row["model_seed"]))
        if not cohorts or any(seeds != expected_seeds for seeds in cohorts.values()):
            raise ValueError("full-search plan is not the exact three-seed matrix")
        marker_name = f"M6F_M7F_{engine_stage}_{endpoint:06d}_COMPLETE.json"
        for row in rows:
            attempt = int(row.get("attempt_index", 0))
            marker_path = (
                result
                / base
                / str(row["physical_run_id"])
                / f"attempt_{attempt:02d}"
                / marker_name
            )
            marker = read_json(_governed_file(result, marker_path.relative_to(result).as_posix()))
            basis = validate_fast_run_marker(
                marker,
                row,
                stage=engine_stage,
                endpoint=endpoint,
                expected_attempt_index=attempt,
            )
            run_boundary_id = f"{row['physical_run_id']}@{endpoint:06d}"
            registry.append(
                {
                    "run_boundary_id": run_boundary_id,
                    "stage": logical_stage,
                    "physical_run_id": row["physical_run_id"],
                    "dataset": row["dataset"],
                    "candidate_set_id": row["candidate_set_id"],
                    "candidate_set_hash": row["candidate_set_hash"],
                    "model_seed": int(row["model_seed"]),
                    "continuation_from_update": int(row.get("continuation_from_update", 0)),
                    "endpoint_update": endpoint,
                    "optimizer_steps": int(basis["optimizer_steps"]),
                    "model_total_seconds": float(basis["model_total_seconds"]),
                    "marker_relative_path": marker_path.relative_to(result).as_posix(),
                    "marker_identity": marker["identity"],
                    "checkpoint_relative_path": basis["checkpoint_relative_path"],
                    "checkpoint_sha256": basis["checkpoint_sha256"],
                    "representation_relative_path": basis["representation_relative_path"],
                    "representation_sha256": basis["representation_sha256"],
                    "labels_mounted": False,
                    "status": "PASS",
                }
            )
            for artifact_role, relative_key, hash_key in (
                ("COMPLETE_CHECKPOINT", "checkpoint_relative_path", "checkpoint_sha256"),
                ("REPRESENTATION", "representation_relative_path", "representation_sha256"),
            ):
                source = _governed_file(result, str(basis[relative_key]))
                digest = sha256_file(source)
                if digest != str(basis[hash_key]):
                    raise ValueError("full-search retained artifact hash changed")
                retention.append(
                    {
                        "run_boundary_id": run_boundary_id,
                        "physical_run_id": row["physical_run_id"],
                        "dataset": row["dataset"],
                        "stage": logical_stage,
                        "artifact_role": artifact_role,
                        "update": endpoint,
                        "remote_relative_path": str(basis[relative_key]),
                        "size_bytes": source.stat().st_size,
                        "sha256": digest,
                        "retention_status": "RETAIN_REMOTE_UNTIL_MIN_PACKAGE_VERIFIED",
                        "included_in_minimal_package": False,
                        "download_authorized": False,
                    }
                )

    for logical_stage, endpoint, plan, seal, seal_status in SEARCH_STAGE_SPECS:
        collect(
            logical_stage=logical_stage,
            engine_stage="SEARCH",
            endpoint=endpoint,
            plan_name=plan,
            seal_name=seal,
            seal_status=seal_status,
            base="search_runs",
        )
    collect(
        logical_stage="FULL_SEARCH_PRELIMINARY",
        engine_stage="PRELIMINARY",
        endpoint=3000,
        plan_name="FAST_PRELIMINARY_RUN_PLAN.csv",
        seal_name="FAST_PRELIMINARY_RUN_PLAN_SEAL.json",
        seal_status="SEALED_BEFORE_PRELIMINARY_GPU_TRAINING",
        base="preliminary_runs",
    )
    if len({row["run_boundary_id"] for row in registry}) != len(registry):
        raise ValueError("full-search run boundary identities are duplicated")
    return registry, retention


def _blank_metric(dataset: str, role: str, seed: int) -> dict[str, Any]:
    return {
        **{key: "" for key in METRIC_FIELDS},
        "dataset": dataset,
        "logical_role": role,
        "model_seed": int(seed),
        "confirmatory_claim_allowed": False,
        "status": "MEASURED_BUDGET_BLOCKED",
    }


def close_conditional_full_search(*, result_root: str | Path) -> dict[str, Any]:
    """Close authorized searches and publish stable H1 scientific summaries."""

    result = Path(result_root).resolve(strict=True)
    authorization, auth_by_dataset = validate_search_authorization(
        read_json(result / "DBLP_AMINER_SEARCH_AUTHORIZATION.json"),
        measured_budget_rows=read_csv(
            _governed_file(result, "DBLP_AMINER_MEASURED_BUDGET.csv")
        ),
    )
    bridge = _validated_json(
        result,
        BRIDGE_NAME,
        schema="metachorus.v53.m7f-h1.full-search-authorization-bridge.v1",
    )
    if bridge["basis"].get("search_authorization_identity") != authorization["identity"]:
        raise ValueError("full-search bridge is not bound to the measured authorization")
    active = tuple(
        dataset for dataset in MANUAL_DATASETS if dataset in authorization["basis"]["active_datasets"]
    )
    blocked = tuple(dataset for dataset in MANUAL_DATASETS if dataset not in active)

    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    shared_identities: dict[str, Any] = {}
    decision_by_dataset: dict[str, dict[str, Any]] = {}

    if active:
        fast_status = _validated_json(
            result,
            FAST_STATUS_NAME,
            schema="metachorus.v53.m6f-m7f.dataset-execution-status.v1",
        )
        expected_fast_active = [name for name in FAST_DATASET_NAMES if name in active]
        if (
            fast_status["basis"].get("active_datasets") != expected_fast_active
            or fast_status["basis"].get("h1_search_authorization_identity") != authorization["identity"]
        ):
            raise ValueError("FAST compatibility status differs from H1 authorization")
        metric_seals = _validated_search_metric_seals(
            result,
            fast_status=fast_status,
            active_datasets=expected_fast_active,
        )
        high_recall = _validated_json(result, "FAST_R1_HIGH_RECALL_POOL.json", status="PASS")
        shortlist = _validated_json(result, "FAST_R3_ROBUST_SHORTLIST.json", status="PASS")
        registry = _validated_json(
            result,
            "FAST_FINAL_FROZEN_SET_REGISTRY.json",
            schema="metachorus.v53.m6f-m7f.final-frozen-set-registry.v1",
            status="PASS",
        )
        selected_document = _validated_json(
            result,
            "FAST_SEARCH_SELECTED_SET.json",
            schema="metachorus.v53.m6f-m7f.search-selected-set.v1",
            status="PASS",
        )
        selection_seal = _validated_json(
            result,
            "FAST_SELECTION_SEAL.json",
            schema="metachorus.v53.m6f-m7f.selection-seal.v1",
            status="SEALED_AFTER_SEARCH_BEFORE_PRELIMINARY_EVALUATION",
        )
        prelim_plan_seal = _validated_json(
            result,
            "FAST_PRELIMINARY_RUN_PLAN_SEAL.json",
            schema="metachorus.v53.m6f-m7f.preliminary-run-plan-seal.v1",
            status="SEALED_BEFORE_PRELIMINARY_GPU_TRAINING",
        )
        prelim_registry_seal = _validated_json(
            result,
            "FAST_PRELIMINARY_TRAINING_REGISTRY_SEAL.json",
            schema="metachorus.v53.m6f-m7f.preliminary-training-registry-seal.v1",
            status="SEALED_BEFORE_PRELIMINARY_EVALUATION",
        )
        access = _validated_json(
            result,
            "FAST_PRELIMINARY_ACCESS_AUDIT.json",
            schema="metachorus.v53.m6f-m7f.preliminary-access-audit.v1",
            status="PASS",
        )
        if (
            tuple(access["basis"].get("active_datasets", [])) != tuple(expected_fast_active)
            or int(access["basis"].get("prelim_locked_report_access_count", -1)) != len(active)
            or access["basis"].get("raw_labels_persisted") is not False
            or access["basis"].get("confirmatory_claim_allowed") is not False
            or selection_seal["basis"].get("evaluation_seeds_started") is not False
            or selection_seal["basis"].get("further_search_or_second_choice") is not False
        ):
            raise ValueError("full-search selection/preliminary boundary is invalid")
        registry_records = list(registry["basis"].get("records", []))
        if {str(row.get("dataset")) for row in registry_records} != set(active):
            raise ValueError("full-search final registry differs from authorized datasets")
        decisions = list(selected_document["basis"].get("decisions", []))
        for decision in decisions:
            if decision.get("identity") != canonical_json_hash(decision.get("basis", {})):
                raise ValueError("full-search decision identity changed")
            decision_by_dataset[str(decision["basis"]["dataset"])] = dict(decision)
        if set(decision_by_dataset) != set(active):
            raise ValueError("full-search decisions differ from authorized datasets")
        for dataset in active:
            selected = next(row for row in registry_records if str(row["dataset"]) == dataset)
            decision = decision_by_dataset[dataset]
            safety_rows = list(decision["basis"].get("manual_safety_rows", []))
            selected_safety = [
                row
                for row in safety_rows
                if str(row.get("candidate_set_hash")) == str(selected["candidate_set_hash"])
            ]
            if len(selected_safety) != 1 or selected_safety[0].get("status") != "PASS":
                raise ValueError("full-search selected set did not pass the frozen Manual gate")
            selection_rows.append(
                {
                    "dataset": dataset,
                    "status": selected["status"],
                    "candidate_set_id": selected["candidate_set_id"],
                    "candidate_set_hash": selected["candidate_set_hash"],
                    "candidate_set_ordinal": int(selected["candidate_set_ordinal"]),
                    "path_ids": list(selected["path_ids"]),
                    "selection_decision_identity": decision["identity"],
                    "manual_safety_status": "PASS",
                }
            )
        source_metrics = read_csv(_governed_file(result, "FAST_PRELIMINARY_METRICS_BY_SEED.csv"))
        source_summary = read_csv(_governed_file(result, "FAST_PRELIMINARY_SUMMARY.csv"))
        source_comparison = read_csv(_governed_file(result, "FAST_MANUAL_AUTOMATIC_COMPARISON.csv"))
        metric_rows = [
            {key: row.get(key, "") for key in METRIC_FIELDS}
            for row in source_metrics
            if str(row.get("dataset")) in MANUAL_DATASETS
            and str(row.get("logical_role")) in {"FINAL_FROZEN_METHOD", "MANUAL"}
        ]
        summary_rows = [
            {key: row.get(key, "") for key in SUMMARY_FIELDS}
            for row in source_summary
            if str(row.get("dataset")) in MANUAL_DATASETS
            and str(row.get("logical_role")) in {"FINAL_FROZEN_METHOD", "MANUAL"}
        ]
        comparison_rows = [
            {key: row.get(key, "") for key in COMPARISON_FIELDS}
            for row in source_comparison
            if str(row.get("dataset")) in MANUAL_DATASETS
            and str(row.get("comparison")) == "FINAL_FROZEN_METHOD_VS_MANUAL"
        ]
        for dataset in active:
            if (
                len([row for row in metric_rows if row["dataset"] == dataset and row["status"] == "PASS"]) != 6
                or len([row for row in summary_rows if row["dataset"] == dataset and row["status"] == "PASS"]) != 2
                or len([row for row in comparison_rows if row["dataset"] == dataset and row["status"] == "PASS"]) != 3
            ):
                raise ValueError("full-search preliminary comparison is not the exact fresh three-seed matrix")
        registry_rows, retention_rows = collect_full_search_run_evidence(result, active)
        shared_identities = {
            "fast_dataset_execution_status_identity": fast_status["identity"],
            "search_metric_seal_identities": {
                str(endpoint): value["identity"] for endpoint, value in metric_seals.items()
            },
            "r1_high_recall_pool_identity": high_recall["identity"],
            "r3_robust_shortlist_identity": shortlist["identity"],
            "final_registry_identity": registry["identity"],
            "selection_seal_identity": selection_seal["identity"],
            "preliminary_plan_seal_identity": prelim_plan_seal["identity"],
            "preliminary_registry_seal_identity": prelim_registry_seal["identity"],
            "preliminary_access_audit_identity": access["identity"],
        }

    for dataset in blocked:
        selection_rows.append(
            {
                "dataset": dataset,
                "status": "MEASURED_BUDGET_BLOCKED",
                "candidate_set_id": "",
                "candidate_set_hash": "",
                "candidate_set_ordinal": "",
                "path_ids": [],
                "selection_decision_identity": "",
                "manual_safety_status": "NOT_RUN_BUDGET_BLOCKED",
            }
        )
        for role in ("FINAL_FROZEN_METHOD", "MANUAL"):
            for seed in EVAL_MODEL_SEEDS:
                metric_rows.append(_blank_metric(dataset, role, seed))
            summary_rows.append(
                {
                    **{key: "" for key in SUMMARY_FIELDS},
                    "dataset": dataset,
                    "logical_role": role,
                    "model_seed_count": 0,
                    "evidence_level": "MEASURED_BUDGET_BLOCKED",
                    "status": "MEASURED_BUDGET_BLOCKED",
                }
            )
        comparison_rows.append(
            {
                **{key: "" for key in COMPARISON_FIELDS},
                "dataset": dataset,
                "comparison": "FINAL_FROZEN_METHOD_VS_MANUAL",
                "confirmatory_claim_allowed": False,
                "decision_status": "MEASURED_BUDGET_BLOCKED",
                "status": "MEASURED_BUDGET_BLOCKED",
            }
        )

    selection_rows.sort(key=lambda row: MANUAL_DATASETS.index(str(row["dataset"])))
    metric_rows.sort(
        key=lambda row: (
            MANUAL_DATASETS.index(str(row["dataset"])),
            str(row["logical_role"]),
            int(row["model_seed"]),
        )
    )
    summary_rows.sort(
        key=lambda row: (MANUAL_DATASETS.index(str(row["dataset"])), str(row["logical_role"]))
    )
    comparison_rows.sort(
        key=lambda row: (
            MANUAL_DATASETS.index(str(row["dataset"])),
            -1 if row["model_seed"] == "" else int(row["model_seed"]),
        )
    )
    write_csv_once(
        result / "DBLP_AMINER_FULL_SEARCH_SELECTION_REGISTRY.csv",
        selection_rows,
        fieldnames=SELECTION_FIELDS,
    )
    write_csv_once(
        result / "DBLP_AMINER_FULL_SEARCH_PRELIMINARY_METRICS_BY_SEED.csv",
        metric_rows,
        fieldnames=METRIC_FIELDS,
    )
    write_csv_once(
        result / "DBLP_AMINER_FULL_SEARCH_PRELIMINARY_SUMMARY.csv",
        summary_rows,
        fieldnames=SUMMARY_FIELDS,
    )
    write_csv_once(
        result / "DBLP_AMINER_FULL_SEARCH_COMPARISON.csv",
        comparison_rows,
        fieldnames=COMPARISON_FIELDS,
    )
    write_csv_once(
        result / "DBLP_AMINER_FULL_SEARCH_RUN_REGISTRY.csv",
        registry_rows,
        fieldnames=RUN_REGISTRY_FIELDS,
    )
    write_csv_once(
        result / "DBLP_AMINER_FULL_SEARCH_RETENTION_MANIFEST.csv",
        retention_rows,
        fieldnames=RETENTION_FIELDS,
    )
    
    
    serialized_selection = read_csv(
        result / "DBLP_AMINER_FULL_SEARCH_SELECTION_REGISTRY.csv"
    )
    serialized_metrics = read_csv(
        result / "DBLP_AMINER_FULL_SEARCH_PRELIMINARY_METRICS_BY_SEED.csv"
    )
    serialized_summary = read_csv(
        result / "DBLP_AMINER_FULL_SEARCH_PRELIMINARY_SUMMARY.csv"
    )
    serialized_comparison = read_csv(
        result / "DBLP_AMINER_FULL_SEARCH_COMPARISON.csv"
    )
    serialized_registry = read_csv_allow_empty(
        result / "DBLP_AMINER_FULL_SEARCH_RUN_REGISTRY.csv"
    )
    serialized_retention = read_csv_allow_empty(
        result / "DBLP_AMINER_FULL_SEARCH_RETENTION_MANIFEST.csv"
    )

    completion_identities: dict[str, str] = {}
    completion_documents: dict[str, dict[str, Any]] = {}
    for dataset in active:
        selected = next(row for row in selection_rows if row["dataset"] == dataset)
        decision = decision_by_dataset[dataset]
        dataset_registry = [
            row for row in serialized_registry if row["dataset"] == dataset
        ]
        dataset_retention = [
            row for row in serialized_retention if row["dataset"] == dataset
        ]
        dataset_metrics = [
            row for row in serialized_metrics if row["dataset"] == dataset
        ]
        basis = {
            "dataset": dataset,
            "search_authorization_identity": authorization["identity"],
            "authorization_record_identity": canonical_json_hash(auth_by_dataset[dataset]),
            "authorization_bridge_identity": bridge["identity"],
            "final_search_endpoint_update": 3000,
            "search_seed_count": len(SEARCH_MODEL_SEEDS),
            "evaluation_seed_count": len(EVAL_MODEL_SEEDS),
            "selected_candidate_set_id": selected["candidate_set_id"],
            "selected_candidate_set_hash": selected["candidate_set_hash"],
            "selection_decision_identity": decision["identity"],
            "manual_safety_status": "PASS",
            "run_registry_rows_identity": canonical_json_hash(dataset_registry),
            "retention_rows_identity": canonical_json_hash(dataset_retention),
            "preliminary_metric_rows_identity": canonical_json_hash(dataset_metrics),
            **shared_identities,
            "labels_mounted_during_training": False,
            "prelim_locked_report_used_for_selection": False,
            "candidate_library_reduced": False,
            "algorithm_changed": False,
            "confirmatory_claim_allowed": False,
        }
        completion = envelope(
            "metachorus.v53.m7f-h1.full-search-complete.v1", "PASS", basis
        )
        write_json_once(
            result / f"M7F_H1_{dataset.upper()}_FULL_SEARCH_COMPLETE.json",
            completion,
        )
        completion_identities[dataset] = completion["identity"]
        completion_documents[dataset] = completion

    status_records = [
        {
            "dataset": dataset,
            "status": (
                "FULL_SEARCH_AUTHORIZED_AND_COMPLETE"
                if dataset in active
                else "MEASURED_BUDGET_BLOCKED"
            ),
            "authorized": dataset in active,
            "completion_identity": completion_identities.get(dataset, ""),
            "completion_basis": completion_documents.get(dataset, {}).get(
                "basis", {}
            ),
            "authorization_record_identity": canonical_json_hash(auth_by_dataset[dataset]),
        }
        for dataset in MANUAL_DATASETS
    ]
    status = envelope(
        "metachorus.v53.m7f-h1.full-search-status.v1",
        "FULL_SEARCH_COMPLETE" if active else "NO_DATASET_AUTHORIZED",
        {
            "search_authorization_identity": authorization["identity"],
            "authorization_bridge_identity": bridge["identity"],
            "active_datasets": list(active),
            "blocked_datasets": list(blocked),
            "records": status_records,
            "selection_rows_identity": canonical_json_hash(serialized_selection),
            "preliminary_metric_rows_identity": canonical_json_hash(serialized_metrics),
            "preliminary_summary_rows_identity": canonical_json_hash(serialized_summary),
            "comparison_rows_identity": canonical_json_hash(serialized_comparison),
            "run_registry_rows_identity": canonical_json_hash(serialized_registry),
            "retention_rows_identity": canonical_json_hash(serialized_retention),
            "authorized_dataset_count": len(active),
            "all_authorized_datasets_complete": bool(active),
            "candidate_library_reduced": False,
            "algorithm_changed": False,
            "labels_mounted_during_training": False,
            "prelim_locked_report_used_for_selection": False,
            "confirmatory_claim_allowed": False,
        },
    )
    write_json_once(result / STATUS_NAME, status)
    return status


__all__ = [
    "BRIDGE_NAME",
    "STATUS_NAME",
    "SEARCH_STAGE_SPECS",
    "collect_full_search_run_evidence",
    "normalize_full_search_registry_rows",
    "normalize_full_search_retention_rows",
    "validate_search_authorization",
    "build_authorized_r1_plan",
    "prepare_conditional_full_search",
    "close_conditional_full_search",
]
