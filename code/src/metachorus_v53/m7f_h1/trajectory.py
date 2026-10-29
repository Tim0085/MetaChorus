"""IMDb trajectory metrics, checkpoint rules, and post-seal summaries."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .common import canonical_json_hash, envelope, write_csv_once, write_json_once, write_text_once
from .constants import CHECKPOINT_SELECTORS, EVALUATION_MODEL_SEEDS


METRIC_NAMES = ("Micro_F1", "Macro_F1", "NMI", "ARI", "F_cls", "F_clu", "F_joint")
SET_ROLES = ("FINAL_FROZEN_METHOD", "MANUAL")
REQUIRED_EXISTING_UPDATES = (300, 1000, 2000, 3000)


def _metric_value(row: Mapping[str, Any], key: str) -> float:
    metrics = row.get("metrics", {})
    value = metrics.get(key) if isinstance(metrics, Mapping) and key in metrics else row.get(key)
    if value is None:
        raise ValueError(f"trajectory row lacks {key}")
    observed = float(value)
    if not np.isfinite(observed):
        raise ValueError(f"trajectory row has non-finite {key}")
    return observed


def normalize_trajectory_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        value = dict(raw)
        if (
            value.get("dataset") != "imdb"
            or value.get("path_set_role") not in SET_ROLES
            or int(value.get("model_seed", -1)) not in EVALUATION_MODEL_SEEDS
            or int(value.get("update", value.get("endpoint_update", -1))) <= 0
            or value.get("prelim_locked_report_access", False) is not False
            or "PRELIM_LOCKED_REPORT" in str(value.get("metric_role", ""))
        ):
            raise ValueError("checkpoint selection may use only IMDb GLOBAL_SEARCH_POOL rows")
        micro = _metric_value(value, "Micro_F1")
        macro = _metric_value(value, "Macro_F1")
        nmi = _metric_value(value, "NMI")
        ari = _metric_value(value, "ARI")
        f_cls = _metric_value(value, "F_cls") if (
            "F_cls" in value or "F_cls" in value.get("metrics", {})
        ) else (micro + macro) / 2.0
        f_clu = _metric_value(value, "F_clu") if (
            "F_clu" in value or "F_clu" in value.get("metrics", {})
        ) else (nmi + ari) / 2.0
        f_joint = _metric_value(value, "F_joint") if (
            "F_joint" in value or "F_joint" in value.get("metrics", {})
        ) else (f_cls + f_clu) / 2.0
        normalized.append(
            {
                "dataset": "imdb",
                "physical_run_id": str(value["physical_run_id"]),
                "path_set_role": str(value["path_set_role"]),
                "candidate_set_id": str(value["candidate_set_id"]),
                "candidate_set_hash": str(value["candidate_set_hash"]),
                "model_seed": int(value["model_seed"]),
                "update": int(value.get("update", value.get("endpoint_update"))),
                "Micro_F1": micro,
                "Macro_F1": macro,
                "NMI": nmi,
                "ARI": ari,
                "F_cls": f_cls,
                "F_clu": f_clu,
                "F_joint": f_joint,
                "metric_source_relative_path": str(
                    value.get("metric_source_relative_path", "")
                ),
                "metric_source_sha256": str(value.get("metric_source_sha256", "")),
                "representation_relative_path": str(
                    value.get("representation_relative_path", "")
                ),
                "representation_sha256": str(
                    value.get("representation_sha256", "")
                ),
                "evidence_type": str(
                    value.get("evidence_type", "EVALUATED_REPRESENTATION")
                ),
                "selection_role": "GLOBAL_SEARCH_POOL_ONLY",
                "prelim_locked_report_access": False,
                "status": "PASS",
            }
        )
    keys = [
        (row["path_set_role"], row["model_seed"], row["update"])
        for row in normalized
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("trajectory metric identity is duplicated")
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in normalized:
        grouped[(row["path_set_role"], row["model_seed"])].add(row["update"])
    expected_groups = {
        (role, seed) for role in SET_ROLES for seed in EVALUATION_MODEL_SEEDS
    }
    if set(grouped) != expected_groups:
        raise ValueError("IMDb trajectory metrics do not cover exact set×seed rows")
    update_sets = list(grouped.values())
    if (
        any(values != update_sets[0] for values in update_sets[1:])
        or not set(REQUIRED_EXISTING_UPDATES).issubset(update_sets[0])
    ):
        raise ValueError("IMDb trajectories lack a common required update grid")
    return sorted(
        normalized,
        key=lambda row: (
            SET_ROLES.index(row["path_set_role"]),
            EVALUATION_MODEL_SEEDS.index(row["model_seed"]),
            row["update"],
        ),
    )


def global_best_by_metric(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = normalize_trajectory_metrics(rows)
    output: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        best = min(
            metrics,
            key=lambda row: (
                -float(row[metric]),
                int(row["update"]),
                str(row["candidate_set_hash"]),
                int(row["model_seed"]),
            ),
        )
        output.append(
            {
                "metric": metric,
                "value": best[metric],
                "candidate_set_id": best["candidate_set_id"],
                "candidate_set_hash": best["candidate_set_hash"],
                "path_set_role": best["path_set_role"],
                "model_seed": best["model_seed"],
                "update": best["update"],
                "metric_source_relative_path": best["metric_source_relative_path"],
                "metric_source_sha256": best["metric_source_sha256"],
                "evidence_type": best["evidence_type"],
                "reporting_scope": "PER_METRIC_UPPER_BOUND_ONLY",
                "row_splicing_allowed": False,
            }
        )
    return output


def best_by_metric(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = normalize_trajectory_metrics(rows)
    output: list[dict[str, Any]] = []
    for role in SET_ROLES:
        for seed in EVALUATION_MODEL_SEEDS:
            candidates = [
                row for row in metrics
                if row["path_set_role"] == role and row["model_seed"] == seed
            ]
            for metric in METRIC_NAMES:
                best = min(
                    candidates,
                    key=lambda row: (-float(row[metric]), int(row["update"])),
                )
                output.append(
                    {
                        "path_set_role": role,
                        "candidate_set_id": best["candidate_set_id"],
                        "candidate_set_hash": best["candidate_set_hash"],
                        "model_seed": seed,
                        "metric": metric,
                        "selected_update": best["update"],
                        "selected_value": best[metric],
                        "representation_relative_path": best[
                            "representation_relative_path"
                        ],
                        "representation_sha256": best["representation_sha256"],
                        "selection_role": "GLOBAL_SEARCH_POOL_ONLY",
                        "locked_report_used_for_selection": False,
                    }
                )
    return output


def checkpoint_selection(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = normalize_trajectory_metrics(rows)
    by_key = {
        (row["path_set_role"], row["model_seed"], row["update"]): row
        for row in metrics
    }
    output: list[dict[str, Any]] = []
    selector_metric = {
        "MICRO_BEST": "Micro_F1",
        "NMI_BEST": "NMI",
        "JOINT_BEST": "F_joint",
    }
    for role in SET_ROLES:
        role_rows = [row for row in metrics if row["path_set_role"] == role]
        updates = sorted({int(row["update"]) for row in role_rows})
        common_update = min(
            updates,
            key=lambda update: (
                -median(
                    by_key[(role, seed, update)]["F_joint"]
                    for seed in EVALUATION_MODEL_SEEDS
                ),
                update,
            ),
        )
        for seed in EVALUATION_MODEL_SEEDS:
            seed_rows = [row for row in role_rows if row["model_seed"] == seed]
            selected_by_selector: dict[str, dict[str, Any]] = {}
            for selector, metric in selector_metric.items():
                selected_by_selector[selector] = min(
                    seed_rows,
                    key=lambda row: (-float(row[metric]), int(row["update"])),
                )
            selected_by_selector["COMMON_JOINT_UPDATE"] = by_key[
                (role, seed, common_update)
            ]
            selected_by_selector["FIXED_END_3000"] = by_key[(role, seed, 3000)]
            for selector in CHECKPOINT_SELECTORS:
                selected = selected_by_selector[selector]
                output.append(
                    {
                        "selector": selector,
                        "path_set_role": role,
                        "candidate_set_id": selected["candidate_set_id"],
                        "candidate_set_hash": selected["candidate_set_hash"],
                        "model_seed": seed,
                        "selected_update": int(selected["update"]),
                        "selection_metric": selector_metric.get(
                            selector,
                            "median_F_joint_across_3_seeds"
                            if selector == "COMMON_JOINT_UPDATE"
                            else "Fixed-End",
                        ),
                        "selection_value": (
                            median(
                                by_key[(role, value, common_update)]["F_joint"]
                                for value in EVALUATION_MODEL_SEEDS
                            )
                            if selector == "COMMON_JOINT_UPDATE"
                            else selected[
                                selector_metric.get(selector, "F_joint")
                            ]
                        ),
                        "representation_relative_path": selected[
                            "representation_relative_path"
                        ],
                        "representation_sha256": selected[
                            "representation_sha256"
                        ],
                        "selection_role": "GLOBAL_SEARCH_POOL_ONLY",
                        "locked_report_used_for_selection": False,
                        "per_metric_row_splicing_allowed": False,
                    }
                )
    if len(output) != len(CHECKPOINT_SELECTORS) * len(SET_ROLES) * len(EVALUATION_MODEL_SEEDS):
        raise RuntimeError("checkpoint selection table is not exact")
    basis = {
        "selection_rows_identity": canonical_json_hash(output),
        "row_count": len(output),
        "selectors": list(CHECKPOINT_SELECTORS),
        "path_set_roles": list(SET_ROLES),
        "evaluation_model_seeds": list(EVALUATION_MODEL_SEEDS),
        "selection_role": "GLOBAL_SEARCH_POOL_ONLY",
        "prelim_locked_report_access": False,
        "common_joint_update_one_per_set": True,
        "per_metric_row_splicing_allowed": False,
        "locked_evaluation_started": False,
    }
    seal = envelope(
        "metachorus.v53.m7f-h1.checkpoint-selection-seal.v1",
        "SEALED_BEFORE_PRELIM_LOCKED_REPORT_EVALUATION",
        basis,
    )
    return output, seal


def validate_selection_seal(
    selection_rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]
) -> None:
    rows: list[dict[str, Any]] = []
    for raw in selection_rows:
        row = dict(raw)
        for key in ("model_seed", "selected_update"):
            row[key] = int(row[key])
        row["selection_value"] = float(row["selection_value"])
        for key in (
            "locked_report_used_for_selection",
            "per_metric_row_splicing_allowed",
        ):
            value = row[key]
            row[key] = value.casefold() == "true" if isinstance(value, str) else bool(value)
        rows.append(row)
    basis = seal.get("basis", {})
    if (
        seal.get("status")
        != "SEALED_BEFORE_PRELIM_LOCKED_REPORT_EVALUATION"
        or seal.get("identity") != canonical_json_hash(basis)
        or basis.get("selection_rows_identity") != canonical_json_hash(rows)
        or int(basis.get("row_count", -1)) != len(rows)
        or basis.get("selection_role") != "GLOBAL_SEARCH_POOL_ONLY"
        or basis.get("prelim_locked_report_access") is not False
        or basis.get("locked_evaluation_started") is not False
        or any(row.get("locked_report_used_for_selection") is not False for row in rows)
    ):
        raise ValueError("checkpoint selection seal is invalid")


def summarize_locked_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_rows: Sequence[Mapping[str, Any]],
    selection_seal: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = [dict(row) for row in selection_rows]
    validate_selection_seal(selection, selection_seal)
    selection_keys = {
        (
            row["selector"],
            row["path_set_role"],
            int(row["model_seed"]),
            int(row["selected_update"]),
        )
        for row in selection
    }
    values = [dict(row) for row in rows]
    observed_keys = {
        (
            row["selector"],
            row["path_set_role"],
            int(row["model_seed"]),
            int(row["selected_update"]),
        )
        for row in values
    }
    if (
        observed_keys != selection_keys
        or len(values) != len(selection_keys)
        or any(row.get("metric_role") != "PRELIM_LOCKED_REPORT" for row in values)
    ):
        raise ValueError("locked metrics differ from the sealed selection table")
    for row in values:
        micro, macro, nmi, ari = (
            float(row[name]) for name in ("Micro_F1", "Macro_F1", "NMI", "ARI")
        )
        row["F_cls"] = (micro + macro) / 2.0
        row["F_clu"] = (nmi + ari) / 2.0
        row["F_joint"] = (row["F_cls"] + row["F_clu"]) / 2.0
        row["selection_seal_identity"] = selection_seal["identity"]
        row["per_metric_row_splicing_allowed"] = False
    summaries: list[dict[str, Any]] = []
    for selector in CHECKPOINT_SELECTORS:
        for role in SET_ROLES:
            group = [
                row for row in values
                if row["selector"] == selector and row["path_set_role"] == role
            ]
            if len(group) != 3:
                raise ValueError("locked selector summary requires exactly three seeds")
            summary = {
                "selector": selector,
                "path_set_role": role,
                "model_seed_count": 3,
                "evidence_level": "3_SEED_PRELIMINARY_NONCONFIRMATORY",
                "per_metric_row_splicing_allowed": False,
            }
            for metric in METRIC_NAMES:
                metric_values = [float(row[metric]) for row in group]
                mean = sum(metric_values) / 3.0
                sample_sd = sqrt(
                    sum((value - mean) ** 2 for value in metric_values) / 2.0
                )
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_sample_sd"] = sample_sd
            summaries.append(summary)
    deltas: list[dict[str, Any]] = []
    indexed = {
        (row["selector"], row["path_set_role"], int(row["model_seed"])): row
        for row in values
    }
    for selector in CHECKPOINT_SELECTORS:
        for seed in EVALUATION_MODEL_SEEDS:
            final = indexed[(selector, "FINAL_FROZEN_METHOD", seed)]
            manual = indexed[(selector, "MANUAL", seed)]
            row = {"selector": selector, "model_seed": seed}
            for metric in METRIC_NAMES:
                row[f"delta_{metric}"] = float(final[metric]) - float(manual[metric])
            row["paired_same_seed"] = True
            row["per_metric_row_splicing_allowed"] = False
            deltas.append(row)
    return summaries, deltas


def write_selection_outputs(
    result_root: str | Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(result_root)
    metrics = normalize_trajectory_metrics(rows)
    global_best = global_best_by_metric(metrics)
    per_metric = best_by_metric(metrics)
    selections, seal = checkpoint_selection(metrics)
    write_csv_once(
        root / "IMDB_TRAJECTORY_METRICS_LONG.csv",
        metrics,
        fieldnames=list(metrics[0]),
    )
    write_csv_once(
        root / "IMDB_GLOBAL_BEST_BY_METRIC_AND_UPDATE.csv",
        global_best,
        fieldnames=list(global_best[0]),
    )
    write_csv_once(
        root / "IMDB_BEST_BY_METRIC.csv",
        per_metric,
        fieldnames=list(per_metric[0]),
    )
    write_csv_once(
        root / "IMDB_CHECKPOINT_SELECTION_TABLE.csv",
        selections,
        fieldnames=list(selections[0]),
    )
    write_json_once(root / "IMDB_CHECKPOINT_SELECTION_SEAL.json", seal)
    return selections, seal


def trajectory_decision(
    deltas: Sequence[Mapping[str, Any]],
    *,
    evidence_source_status: str,
) -> dict[str, Any]:
    """Apply the predeclared locked-report early-checkpoint gate."""

    rows = [dict(row) for row in deltas]
    gates: list[dict[str, Any]] = []
    for selector in ("JOINT_BEST", "COMMON_JOINT_UPDATE"):
        group = [row for row in rows if row.get("selector") == selector]
        if len(group) != 3:
            raise ValueError("trajectory decision requires three paired deltas per rule")
        cls = [float(row["delta_F_cls"]) for row in group]
        clu = [float(row["delta_F_clu"]) for row in group]
        joint = [float(row["delta_F_joint"]) for row in group]
        mean_cls = sum(cls) / 3.0
        mean_clu = sum(clu) / 3.0
        mean_joint = sum(joint) / 3.0
        task_family_gate = (
            (mean_cls > 0.0 and mean_clu >= -0.01)
            or (mean_clu > 0.0 and mean_cls >= -0.01)
        )
        passed = (
            mean_joint > 0.0
            and task_family_gate
            and sum(value >= 0.0 for value in joint) >= 2
            and all(value > -0.02 for value in cls)
            and all(value > -0.02 for value in clu)
        )
        gates.append(
            {
                "selector": selector,
                "mean_delta_F_cls": mean_cls,
                "mean_delta_F_clu": mean_clu,
                "mean_delta_F_joint": mean_joint,
                "joint_non_worse_seed_count": sum(
                    value >= 0.0 for value in joint
                ),
                "all_seed_delta_F_cls_above_minus_2pp": all(
                    value > -0.02 for value in cls
                ),
                "all_seed_delta_F_clu_above_minus_2pp": all(
                    value > -0.02 for value in clu
                ),
                "task_family_gate": task_family_gate,
                "passed": passed,
            }
        )
    promising = any(row["passed"] for row in gates)
    status = (
        "IMDB_EARLY_CHECKPOINT_PROMISING"
        if promising
        else "IMDB_MANUAL_FALLBACK_CONFIRMED"
    )
    basis = {
        "evidence_source_status": evidence_source_status,
        "rule_gates": gates,
        "early_checkpoint_ablation_promising": promising,
        "frozen_fixed_end_method_replaced": False,
        "algorithm_retuning": False,
        "ten_seed_ablation_started": False,
        "next_action": (
            "AWAIT_USER_DECISION_FOR_OPTIONAL_10_SEED_EARLY_STOPPING_ABLATION"
            if promising
            else "KEEP_IMDB_MANUAL_FALLBACK"
        ),
        "confirmatory_claim_allowed": False,
    }
    return envelope(
        "metachorus.v53.m7f-h1.imdb-trajectory-decision.v1", status, basis
    )


def write_trajectory_decision(
    result_root: str | Path,
    *,
    deltas: Sequence[Mapping[str, Any]],
    evidence_source_status: str,
) -> dict[str, Any]:
    root = Path(result_root)
    decision = trajectory_decision(
        deltas, evidence_source_status=evidence_source_status
    )
    write_json_once(root / "IMDB_TRAJECTORY_DECISION.json", decision)
    gates = decision["basis"]["rule_gates"]
    lines = [
        "# IMDb trajectory decision",
        "",
        f"- Status: {decision['status']}",
        f"- Evidence source: {evidence_source_status}",
        "- Checkpoint selection used GLOBAL_SEARCH_POOL only: yes",
        "- PRELIM_LOCKED_REPORT was opened only after the selection seal: yes",
        "- Per-metric row splicing: forbidden",
        "- Frozen Fixed-End method replaced: no",
        "",
        "## Unified-rule gates",
        "",
    ]
    for gate in gates:
        lines.append(
            f"- {gate['selector']}: {'PASS' if gate['passed'] else 'FAIL'}; "
            f"mean delta F_joint={gate['mean_delta_F_joint']:.6f}"
        )
    write_text_once(root / "IMDB_TRAJECTORY_DECISION.md", "\n".join(lines) + "\n")
    return decision


__all__ = [
    "METRIC_NAMES",
    "SET_ROLES",
    "normalize_trajectory_metrics",
    "global_best_by_metric",
    "best_by_metric",
    "checkpoint_selection",
    "validate_selection_seal",
    "summarize_locked_metrics",
    "write_selection_outputs",
    "trajectory_decision",
    "write_trajectory_decision",
]
