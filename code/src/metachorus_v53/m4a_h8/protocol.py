"""Frozen paired statistics, confirmation gates, and final H8 method decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    validate_envelope,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    BOOTSTRAP_RESAMPLES,
    FROZEN_PATH_SETS,
    H8_TERMINAL_STATES,
    IMPROVEMENT_OTHER_TASK_MARGIN,
    NONINFERIOR_MARGINS,
    PARENT_SELECTION_IDENTITY,
    PER_SEED_TASK_DROP_LIMIT,
    S_EVAL_MODEL_SEEDS,
    TASKBOOK_SHA256,
)


RAW_METRICS = ("Micro_F1", "Macro_F1", "NMI", "ARI")
COMPOSITE_METRICS = ("F_cls", "F_clu", "F_joint")
METRICS = RAW_METRICS + COMPOSITE_METRICS

PAIRWISE_FIELDS = (
    "comparison",
    "dataset",
    "candidate_role",
    "baseline_role",
    "model_seed",
    "metric",
    "candidate_value",
    "baseline_value",
    "delta",
    "paired_by_same_model_seed",
    "status",
)

INTERVAL_FIELDS = (
    "comparison",
    "dataset",
    "metric",
    "pair_count",
    "mean_delta",
    "sample_sd",
    "standard_error",
    "paired_t_lower_95",
    "paired_t_upper_95",
    "bootstrap_lower_95",
    "bootstrap_upper_95",
    "bootstrap_resamples",
    "non_worse_count",
    "minimum_delta",
    "status",
)


def _finite_float(value: Any, *, field: str) -> float:
    observed = float(value)
    if not np.isfinite(observed):
        raise ValueError(f"H8 metric is not finite: {field}")
    return observed


def _typed_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    if str(value.get("status")) != "PASS":
        raise ValueError("H8 locked metric row is not PASS")
    value["model_seed"] = int(value["model_seed"])
    for field in METRICS:
        value[field] = _finite_float(value[field], field=field)
    expected_cls = (value["Micro_F1"] + value["Macro_F1"]) / 2.0
    expected_clu = (value["NMI"] + value["ARI"]) / 2.0
    expected_joint = (expected_cls + expected_clu) / 2.0
    if not (
        np.isclose(value["F_cls"], expected_cls, rtol=0.0, atol=1e-12)
        and np.isclose(value["F_clu"], expected_clu, rtol=0.0, atol=1e-12)
        and np.isclose(value["F_joint"], expected_joint, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("H8 composite metric formula mismatch")
    return value


def _seed_map(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        value = _typed_metric_row(row)
        seed = int(value["model_seed"])
        if seed in output:
            raise ValueError("H8 comparison contains a duplicate model seed")
        output[seed] = value
    if set(output) != set(S_EVAL_MODEL_SEEDS) or len(output) != 10:
        raise ValueError("H8 comparison must consume the exact ten frozen model seeds")
    return output


def paired_delta_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    comparison: str,
) -> list[dict[str, Any]]:
    """Compute A-B deltas by the same frozen model-seed identity."""

    candidate = _seed_map(candidate_rows)
    baseline = _seed_map(baseline_rows)
    datasets = {str(row["dataset"]) for row in candidate.values()} | {
        str(row["dataset"]) for row in baseline.values()
    }
    candidate_roles = {str(row["role"]) for row in candidate.values()}
    baseline_roles = {str(row["role"]) for row in baseline.values()}
    if len(datasets) != 1 or len(candidate_roles) != 1 or len(baseline_roles) != 1:
        raise ValueError("H8 paired comparison mixes datasets or roles")
    dataset = next(iter(datasets))
    candidate_role = next(iter(candidate_roles))
    baseline_role = next(iter(baseline_roles))
    for values, role in ((candidate, candidate_role), (baseline, baseline_role)):
        frozen = _frozen(dataset, role)
        if any(
            str(row.get("candidate_set_id")) != str(frozen["candidate_set_id"])
            or str(row.get("candidate_set_hash")) != str(frozen["candidate_set_hash"])
            for row in values.values()
        ):
            raise ValueError("H8 paired comparison role is not bound to its frozen set")
    rows: list[dict[str, Any]] = []
    for seed in S_EVAL_MODEL_SEEDS:
        for metric in METRICS:
            candidate_value = float(candidate[seed][metric])
            baseline_value = float(baseline[seed][metric])
            rows.append(
                {
                    "comparison": comparison,
                    "dataset": dataset,
                    "candidate_role": candidate_role,
                    "baseline_role": baseline_role,
                    "model_seed": int(seed),
                    "metric": metric,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "delta": candidate_value - baseline_value,
                    "paired_by_same_model_seed": True,
                    "status": "PASS",
                }
            )
    return rows


def bootstrap_seed(comparison: str, metric: str) -> int:
    payload = {
        "namespace": "TC_LPMSO_H8_PAIRED_BOOTSTRAP",
        "taskbook_sha256": TASKBOOK_SHA256,
        "comparison": comparison,
        "metric": metric,
        "resamples": BOOTSTRAP_RESAMPLES,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def paired_interval(
    rows: Sequence[Mapping[str, Any]], *, comparison: str, metric: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row["comparison"]) == comparison and str(row["metric"]) == metric
    ]
    if len(selected) != 10 or {int(row["model_seed"]) for row in selected} != set(
        S_EVAL_MODEL_SEEDS
    ):
        raise ValueError("H8 paired interval requires ten exact seed pairs")
    datasets = {str(row.get("dataset")) for row in selected}
    if (
        len(datasets) != 1
        or any(str(row.get("status")) != "PASS" for row in selected)
        or any(row.get("paired_by_same_model_seed") is not True for row in selected)
        or any(
            not np.isclose(
                _finite_float(row.get("candidate_value"), field="candidate_value")
                - _finite_float(row.get("baseline_value"), field="baseline_value"),
                _finite_float(row.get("delta"), field="delta"),
                rtol=0.0,
                atol=1e-12,
            )
            for row in selected
        )
    ):
        raise ValueError("H8 paired interval row lineage is invalid")
    by_seed = {int(row["model_seed"]): _finite_float(row["delta"], field="delta") for row in selected}
    deltas = np.asarray([by_seed[seed] for seed in S_EVAL_MODEL_SEEDS], dtype=np.float64)
    mean = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    se = sd / float(np.sqrt(deltas.size))
    critical = float(student_t.ppf(0.975, df=deltas.size - 1))
    t_lower = mean - critical * se
    t_upper = mean + critical * se
    seed = bootstrap_seed(comparison, metric)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(BOOTSTRAP_RESAMPLES, deltas.size))
    bootstrap_means = deltas[indices].mean(axis=1)
    bootstrap_lower, bootstrap_upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    dataset = str(selected[0]["dataset"])
    interval = {
        "comparison": comparison,
        "dataset": dataset,
        "metric": metric,
        "pair_count": int(deltas.size),
        "mean_delta": mean,
        "sample_sd": sd,
        "standard_error": se,
        "paired_t_lower_95": float(t_lower),
        "paired_t_upper_95": float(t_upper),
        "bootstrap_lower_95": float(bootstrap_lower),
        "bootstrap_upper_95": float(bootstrap_upper),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "non_worse_count": int(np.count_nonzero(deltas >= 0.0)),
        "minimum_delta": float(np.min(deltas)),
        "status": "PASS",
    }
    manifest = {
        "comparison": comparison,
        "metric": metric,
        "bootstrap_seed": seed,
        "resamples": BOOTSTRAP_RESAMPLES,
        "sampling_unit": "PAIRED_MODEL_SEED_INDEX",
        "pair_count": 10,
        "model_seeds": list(S_EVAL_MODEL_SEEDS),
        "arms_resampled_independently": False,
        "status": "SEALED",
    }
    return interval, manifest


def _metric_intervals(
    interval_rows: Sequence[Mapping[str, Any]], comparison: str
) -> dict[str, Mapping[str, Any]]:
    output = {
        str(row["metric"]): row
        for row in interval_rows
        if str(row["comparison"]) == comparison
    }
    if set(output) != set(METRICS):
        raise ValueError("H8 comparison interval set is incomplete")
    return output


def _task_seed_deltas(
    delta_rows: Sequence[Mapping[str, Any]], comparison: str
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for metric in ("F_cls", "F_clu"):
        by_seed = {
            int(row["model_seed"]): _finite_float(row["delta"], field="delta")
            for row in delta_rows
            if str(row["comparison"]) == comparison and str(row["metric"]) == metric
        }
        if set(by_seed) != set(S_EVAL_MODEL_SEEDS):
            raise ValueError("H8 task-family per-seed delta set is incomplete")
        output[metric] = [by_seed[seed] for seed in S_EVAL_MODEL_SEEDS]
    return output


def improvement_gate(
    interval_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    *,
    comparison: str,
) -> dict[str, Any]:
    intervals = _metric_intervals(interval_rows, comparison)
    per_seed = _task_seed_deltas(delta_rows, comparison)
    cls_lower = float(intervals["F_cls"]["bootstrap_lower_95"])
    clu_lower = float(intervals["F_clu"]["bootstrap_lower_95"])
    joint_lower = float(intervals["F_joint"]["bootstrap_lower_95"])
    task_family_rule = (cls_lower > 0.0 and clu_lower >= IMPROVEMENT_OTHER_TASK_MARGIN) or (
        clu_lower > 0.0 and cls_lower >= IMPROVEMENT_OTHER_TASK_MARGIN
    )
    per_seed_rule = all(
        value > PER_SEED_TASK_DROP_LIMIT
        for metric in ("F_cls", "F_clu")
        for value in per_seed[metric]
    )
    checks = {
        "bootstrap_joint_lower_gt_zero": joint_lower > 0.0,
        "one_task_positive_other_not_below_minus_0_01": task_family_rule,
        "joint_non_worse_count_at_least_7": int(intervals["F_joint"]["non_worse_count"]) >= 7,
        "every_seed_task_delta_gt_minus_0_02": per_seed_rule,
    }
    return {
        "comparison": comparison,
        "gate": "IMPROVEMENT",
        "checks": checks,
        "passed": all(checks.values()),
    }


def noninferior_gate(
    interval_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    *,
    comparison: str,
) -> dict[str, Any]:
    intervals = _metric_intervals(interval_rows, comparison)
    per_seed = _task_seed_deltas(delta_rows, comparison)
    checks = {
        "bootstrap_cls_lower_at_least_minus_0_01": float(
            intervals["F_cls"]["bootstrap_lower_95"]
        )
        >= NONINFERIOR_MARGINS["F_cls"],
        "bootstrap_clu_lower_at_least_minus_0_01": float(
            intervals["F_clu"]["bootstrap_lower_95"]
        )
        >= NONINFERIOR_MARGINS["F_clu"],
        "bootstrap_joint_lower_at_least_minus_0_005": float(
            intervals["F_joint"]["bootstrap_lower_95"]
        )
        >= NONINFERIOR_MARGINS["F_joint"],
        "joint_non_worse_count_at_least_6": int(intervals["F_joint"]["non_worse_count"]) >= 6,
        "every_seed_task_delta_gt_minus_0_02": all(
            value > PER_SEED_TASK_DROP_LIMIT
            for metric in ("F_cls", "F_clu")
            for value in per_seed[metric]
        ),
    }
    return {
        "comparison": comparison,
        "gate": "NONINFERIOR",
        "checks": checks,
        "passed": all(checks.values()),
    }


def _select_rows(
    rows: Sequence[Mapping[str, Any]], *, dataset: str, role: str
) -> list[dict[str, Any]]:
    selected = [
        _typed_metric_row(row)
        for row in rows
        if str(row["dataset"]) == dataset and str(row["role"]) == role
    ]
    if len(selected) != 10:
        raise ValueError(f"H8 locked metric role count mismatch: {dataset}/{role}")
    frozen = _frozen(dataset, role)
    if any(
        str(row.get("candidate_set_id")) != str(frozen["candidate_set_id"])
        or str(row.get("candidate_set_hash")) != str(frozen["candidate_set_hash"])
        for row in selected
    ):
        raise ValueError("H8 locked metric role is not bound to its frozen set")
    return selected


def _frozen(dataset: str, role: str) -> Mapping[str, Any]:
    matches = [
        row for row in FROZEN_PATH_SETS if row["dataset"] == dataset and row["role"] == role
    ]
    if len(matches) != 1:
        raise RuntimeError("H8 frozen set identity lookup failed")
    return matches[0]


def decide_confirmation(metric_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(metric_rows) != 50:
        raise ValueError("H8 decision requires exactly fifty locked metric rows")
    identity_set = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"]))
        for row in metric_rows
    }
    expected_set = {
        (str(frozen["dataset"]), str(frozen["candidate_set_hash"]), int(seed))
        for frozen in FROZEN_PATH_SETS
        for seed in S_EVAL_MODEL_SEEDS
    }
    if identity_set != expected_set or len(identity_set) != 50:
        raise ValueError("H8 decision metric identities differ from the frozen matrix")

    comparisons = {
        "FREEBASE_H7_SELECTED_VS_MANUAL": (
            _select_rows(metric_rows, dataset="freebase", role="H7_SELECTED"),
            _select_rows(metric_rows, dataset="freebase", role="MANUAL"),
        ),
        "FREEBASE_H7_SELECTED_VS_CURRENT_AUTOMATIC": (
            _select_rows(metric_rows, dataset="freebase", role="H7_SELECTED"),
            _select_rows(metric_rows, dataset="freebase", role="CURRENT_AUTOMATIC"),
        ),
        "FREEBASE_CURRENT_AUTOMATIC_VS_MANUAL": (
            _select_rows(metric_rows, dataset="freebase", role="CURRENT_AUTOMATIC"),
            _select_rows(metric_rows, dataset="freebase", role="MANUAL"),
        ),
        "ACM_MANUAL_VS_CURRENT_AUTOMATIC": (
            _select_rows(metric_rows, dataset="acm", role="H7_SELECTED"),
            _select_rows(metric_rows, dataset="acm", role="CURRENT_AUTOMATIC"),
        ),
    }
    delta_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    seed_manifest_rows: list[dict[str, Any]] = []
    for comparison, (candidate, baseline) in comparisons.items():
        comparison_deltas = paired_delta_rows(
            candidate, baseline, comparison=comparison
        )
        delta_rows.extend(comparison_deltas)
        for metric in METRICS:
            interval, seed_manifest = paired_interval(
                comparison_deltas, comparison=comparison, metric=metric
            )
            interval_rows.append(interval)
            seed_manifest_rows.append(seed_manifest)

    primary = "FREEBASE_H7_SELECTED_VS_MANUAL"
    fallback = "FREEBASE_CURRENT_AUTOMATIC_VS_MANUAL"
    primary_improvement = improvement_gate(interval_rows, delta_rows, comparison=primary)
    primary_noninferior = noninferior_gate(interval_rows, delta_rows, comparison=primary)
    fallback_noninferior = noninferior_gate(interval_rows, delta_rows, comparison=fallback)
    if primary_improvement["passed"]:
        terminal = "TC_LPMSO_H8_CONFIRMED_IMPROVEMENT"
        primary_status = "IMPROVEMENT"
        freebase_role = "H7_SELECTED"
    elif primary_noninferior["passed"]:
        terminal = "TC_LPMSO_H8_CONFIRMED_NONINFERIOR"
        primary_status = "NONINFERIOR"
        freebase_role = "H7_SELECTED"
    elif fallback_noninferior["passed"]:
        terminal = "TC_LPMSO_H8_FALLBACK_CURRENT_AUTOMATIC"
        primary_status = "FAIL"
        freebase_role = "CURRENT_AUTOMATIC"
    else:
        terminal = "TC_LPMSO_H8_FALLBACK_MANUAL"
        primary_status = "FAIL"
        freebase_role = "MANUAL"
    if terminal not in H8_TERMINAL_STATES:
        raise RuntimeError("H8 terminal status is not registered")
    selected_sets = [_frozen("acm", "H7_SELECTED"), _frozen("freebase", freebase_role)]
    decision_basis = {
        "terminal_status": terminal,
        "primary_confirmation_status": primary_status,
        "primary_comparison": primary,
        "primary_improvement_gate": primary_improvement,
        "primary_noninferior_gate": primary_noninferior,
        "fallback_comparison": fallback,
        "fallback_noninferior_gate": fallback_noninferior,
        "acm_final_role": "MANUAL",
        "freebase_final_role": freebase_role,
        "selected_sets": [
            {
                "dataset": row["dataset"],
                "role": row["role"],
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": row["candidate_set_ordinal"],
            }
            for row in selected_sets
        ],
        "selected_count": 2,
        "second_choice_used": False,
        "fallback_rule_changed": False,
        "all_model_seeds_consumed": True,
        "model_seed_count": 10,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paired_by_same_model_seed": True,
        "further_retuning": False,
        "parent_selection_identity": PARENT_SELECTION_IDENTITY,
    }
    return {
        "decision": envelope(
            "metachorus.v53.m4a-h8.confirmation-decision.v1",
            terminal,
            decision_basis,
        ),
        "delta_rows": delta_rows,
        "interval_rows": interval_rows,
        "bootstrap_manifest": envelope(
            "metachorus.v53.m4a-h8.bootstrap-seed-manifest.v1",
            "SEALED",
            {
                "resamples_per_comparison_metric": BOOTSTRAP_RESAMPLES,
                "comparison_metric_count": len(seed_manifest_rows),
                "sampling_unit": "PAIRED_MODEL_SEED_INDEX",
                "model_seeds": list(S_EVAL_MODEL_SEEDS),
                "records": seed_manifest_rows,
                "records_identity": canonical_json_hash(seed_manifest_rows),
            },
        ),
    }


def finalize_confirmation(result_root: Path) -> dict[str, Any]:
    result_root = result_root.resolve(strict=True)
    access = validate_envelope(
        read_json(result_root / "M4A_H8_LOCKED_ACCESS_AUDIT.json"),
        schema_version="metachorus.v53.m4a-h8.locked-access-audit.v1",
        status="PASS",
    )
    if int(access["basis"].get("h5_locked_report_access_count", -1)) != 2:
        raise ValueError("H8 final decision requires one locked access per dataset")
    metrics_path = result_root / "TC_LPMSO_H8_LOCKED_METRICS_BY_MODEL_SEED.csv"
    metric_rows = read_csv(metrics_path)
    resolved = decide_confirmation(metric_rows)
    decision = resolved["decision"]
    write_csv_once(
        result_root / "TC_LPMSO_H8_LOCKED_PAIRWISE_DELTAS.csv",
        resolved["delta_rows"],
        fieldnames=PAIRWISE_FIELDS,
    )
    write_csv_once(
        result_root / "TC_LPMSO_H8_LOCKED_INTERVALS.csv",
        resolved["interval_rows"],
        fieldnames=INTERVAL_FIELDS,
    )
    write_json_once(
        result_root / "TC_LPMSO_H8_BOOTSTRAP_SEED_MANIFEST.json",
        resolved["bootstrap_manifest"],
    )
    write_json_once(result_root / "TC_LPMSO_H8_CONFIRMATION_DECISION.json", decision)
    write_text_once(
        result_root / "TC_LPMSO_H8_CONFIRMATION_DECISION.md",
        "# M4A-H8 locked confirmation decision\n\n"
        f"- Terminal status: `{decision['status']}`\n"
        f"- Primary outcome: `{decision['basis']['primary_confirmation_status']}`\n"
        "- ACM final method: frozen Manual / H7-selected complete set.\n"
        f"- Freebase final role: `{decision['basis']['freebase_final_role']}`\n"
        "- All ten frozen model seeds were consumed; no second choice was used.\n"
        "- Further TC-LPMSO retuning is frozen.\n",
    )
    final_records = [dict(row) for row in decision["basis"]["selected_sets"]]
    registry_basis = {
        "terminal_status": decision["status"],
        "decision_identity": decision["identity"],
        "records": final_records,
        "record_count": len(final_records),
        "single_complete_set_per_dataset": True,
        "taskwise_set_splicing": False,
        "second_choice_used": False,
        "further_retuning": False,
    }
    registry = envelope(
        "metachorus.v53.m4a-h8.final-path-set-registry.v1",
        decision["status"],
        registry_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H8_FINAL_PATH_SET_REGISTRY.json", registry)
    freeze_basis = {
        "terminal_status": decision["status"],
        "decision_identity": decision["identity"],
        "final_path_set_registry_identity": registry["identity"],
        "locked_metric_sha256": sha256_file(metrics_path),
        "all_fifty_runs_consumed": True,
        "all_ten_model_seeds_consumed": True,
        "algorithm_changed": False,
        "candidate_sets_changed": False,
        "gate_changed": False,
        "fallback_changed": False,
        "further_retuning": False,
        "method_frozen": True,
    }
    freeze = envelope(
        "metachorus.v53.m4a-h8.final-method-freeze.v1",
        "FROZEN_AFTER_H8_CONFIRMATION",
        freeze_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H8_FINAL_METHOD_FREEZE.json", freeze)

    final_by_dataset = {row["dataset"]: row for row in final_records}
    comparison_rows: list[dict[str, Any]] = []
    for frozen in FROZEN_PATH_SETS:
        matching = [
            _typed_metric_row(row)
            for row in metric_rows
            if str(row["dataset"]) == frozen["dataset"]
            and str(row["candidate_set_hash"]) == frozen["candidate_set_hash"]
        ]
        comparison_rows.append(
            {
                "dataset": frozen["dataset"],
                "role": frozen["role"],
                "candidate_set_id": frozen["candidate_set_id"],
                "candidate_set_hash": frozen["candidate_set_hash"],
                "mean_Micro_F1": float(np.mean([row["Micro_F1"] for row in matching])),
                "mean_Macro_F1": float(np.mean([row["Macro_F1"] for row in matching])),
                "mean_NMI": float(np.mean([row["NMI"] for row in matching])),
                "mean_ARI": float(np.mean([row["ARI"] for row in matching])),
                "mean_F_cls": float(np.mean([row["F_cls"] for row in matching])),
                "mean_F_clu": float(np.mean([row["F_clu"] for row in matching])),
                "mean_F_joint": float(np.mean([row["F_joint"] for row in matching])),
                "is_final_selection": final_by_dataset[frozen["dataset"]]["candidate_set_hash"]
                == frozen["candidate_set_hash"],
                "status": "PASS",
            }
        )
    write_csv_once(
        result_root / "TC_LPMSO_H8_MANUAL_AUTOMATIC_COMPARISON.csv",
        comparison_rows,
        fieldnames=list(comparison_rows[0]),
    )
    append_jsonl_once(
        result_root / "M4A_H8_EXECUTION_LOG.jsonl",
        {
            "event_id": "H8_LOCKED_CONFIRMATION_AND_METHOD_FREEZE",
            "status": decision["status"],
            "decision_identity": decision["identity"],
            "final_registry_identity": registry["identity"],
            "further_retuning": False,
        },
    )
    return decision


__all__ = [
    "COMPOSITE_METRICS",
    "INTERVAL_FIELDS",
    "METRICS",
    "PAIRWISE_FIELDS",
    "RAW_METRICS",
    "bootstrap_seed",
    "decide_confirmation",
    "finalize_confirmation",
    "improvement_gate",
    "noninferior_gate",
    "paired_delta_rows",
    "paired_interval",
]
