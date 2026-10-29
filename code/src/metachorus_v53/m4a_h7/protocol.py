"""Pure H7 shortlist, safety, and robust-selection protocol functions."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

from metachorus_v53.audit.common import canonical_json_hash

from .constants import (
    BALANCED_TOP_COUNTS,
    CONTROLS,
    DEVELOPMENT_MODEL_SEEDS,
    EXPECTED_SHORTLIST_COUNTS,
    EXPECTED_SHORTLIST_IDENTITY,
    EXPECTED_SHORTLIST_TOTAL,
    PARETO_SUPPORT_MINIMUM,
    R4_PER_SEED_DROP_LIMIT,
)


def _finite(value: Any) -> float:
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError("H7 protocol received a non-finite value")
    return observed


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["candidate_set_hash"])


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]) -> bool:
    left_values = [_finite(left[field]) for field in fields]
    right_values = [_finite(right[field]) for field in fields]
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _first_front(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if not any(
            _identity(other) != _identity(row) and _dominates(other, row, fields)
            for other in rows
        )
    ]


def _validated_sources(
    objective_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    objectives = [dict(row) for row in objective_rows]
    objective_keys = [_identity(row) for row in objectives]
    if len(objective_keys) != len(set(objective_keys)):
        raise ValueError("duplicate H6 objective identity")
    pool: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pool_rows:
        value = dict(row)
        key = _identity(value)
        if key in pool:
            raise ValueError(f"duplicate H6 pool identity: {key}")
        if any(
            _bool(value.get(field, False))
            for field in ("truth_fields_read", "h5_locked_report_access", "s_eval_access")
        ):
            raise ValueError("H7 shortlist source contains forbidden evidence access")
        pool[key] = value
    if not objectives or set(objective_keys) != set(pool):
        raise ValueError("H7 objective/pool identity mismatch")
    datasets = {dataset for dataset, _ in objective_keys}
    if datasets != {"acm", "freebase"}:
        raise ValueError("H7 shortlist source dataset set mismatch")
    for row in objectives:
        if int(row["seed_count"]) != len(DEVELOPMENT_MODEL_SEEDS):
            raise ValueError("H7 requires exactly three development seeds")
        source = pool[_identity(row)]
        for field in ("candidate_set_id", "candidate_set_ordinal"):
            if str(row[field]) != str(source[field]):
                raise ValueError(f"H7 parent identity field mismatch: {field}")
    return objectives, pool


def per_seed_pareto(
    objective_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return first-front rows and per-candidate support counts."""

    expanded: list[dict[str, Any]] = []
    for row in objective_rows:
        payload = json.loads(str(row["per_seed_json"]))
        if not isinstance(payload, list) or len(payload) != len(DEVELOPMENT_MODEL_SEEDS):
            raise ValueError("invalid H6 per_seed_json")
        observed_seeds = {int(item["model_seed"]) for item in payload}
        if observed_seeds != set(DEVELOPMENT_MODEL_SEEDS):
            raise ValueError("H6 per_seed_json seed set mismatch")
        for item in payload:
            expanded.append(
                {
                    "dataset": str(row["dataset"]),
                    "candidate_set_id": str(row["candidate_set_id"]),
                    "candidate_set_hash": str(row["candidate_set_hash"]),
                    "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                    "model_seed": int(item["model_seed"]),
                    "F_cls": _finite(item["F_cls"]),
                    "F_clu": _finite(item["F_clu"]),
                }
            )
    fronts: list[dict[str, Any]] = []
    support: defaultdict[tuple[str, str], int] = defaultdict(int)
    for dataset in ("acm", "freebase"):
        for seed in DEVELOPMENT_MODEL_SEEDS:
            cohort = [
                row
                for row in expanded
                if row["dataset"] == dataset and int(row["model_seed"]) == seed
            ]
            if not cohort:
                raise ValueError(f"missing H7 per-seed cohort: {dataset}/{seed}")
            for row in _first_front(cohort, ("F_cls", "F_clu")):
                value = {**row, "front_rank": 1, "status": "FIRST_PARETO_FRONT"}
                fronts.append(value)
                support[_identity(row)] += 1
    supports: list[dict[str, Any]] = []
    for row in objective_rows:
        count = support[_identity(row)]
        supports.append(
            {
                "dataset": str(row["dataset"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "pareto_support_count": count,
                "minimum_support": PARETO_SUPPORT_MINIMUM,
                "selected": count >= PARETO_SUPPORT_MINIMUM,
            }
        )
    return (
        sorted(fronts, key=lambda row: (row["dataset"], row["model_seed"], row["candidate_set_hash"])),
        sorted(supports, key=lambda row: (row["dataset"], row["candidate_set_hash"])),
    )


def robust_four_objective_pareto(
    objective_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        cohort = [dict(row) for row in objective_rows if str(row["dataset"]) == dataset]
        for row in _first_front(cohort, ("Q_cls", "Q_clu", "W_cls", "W_clu")):
            output.append(
                {
                    "dataset": dataset,
                    "candidate_set_id": str(row["candidate_set_id"]),
                    "candidate_set_hash": str(row["candidate_set_hash"]),
                    "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                    "Q_cls": _finite(row["Q_cls"]),
                    "Q_clu": _finite(row["Q_clu"]),
                    "W_cls": _finite(row["W_cls"]),
                    "W_clu": _finite(row["W_clu"]),
                    "front_rank": 1,
                    "status": "ROBUST_PARETO_4D",
                }
            )
    return sorted(output, key=lambda row: (row["dataset"], row["candidate_set_hash"]))


def aggregate_balanced_rank(
    objective_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        rows = [dict(row) for row in objective_rows if str(row["dataset"]) == dataset]
        ranked = sorted(
            rows,
            key=lambda row: (
                max(
                    _finite(row["classification_normalized_midrank"]),
                    _finite(row["clustering_normalized_midrank"]),
                ),
                mean(
                    (
                        _finite(row["classification_normalized_midrank"]),
                        _finite(row["clustering_normalized_midrank"]),
                    )
                ),
                str(row["candidate_set_hash"]),
            ),
        )
        for position, row in enumerate(ranked, 1):
            r_cls = _finite(row["classification_normalized_midrank"])
            r_clu = _finite(row["clustering_normalized_midrank"])
            output.append(
                {
                    "dataset": dataset,
                    "candidate_set_id": str(row["candidate_set_id"]),
                    "candidate_set_hash": str(row["candidate_set_hash"]),
                    "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                    "classification_normalized_midrank": r_cls,
                    "clustering_normalized_midrank": r_clu,
                    "R_balanced": max(r_cls, r_clu),
                    "mean_task_rank": mean((r_cls, r_clu)),
                    "balanced_position": position,
                    "selected": position <= BALANCED_TOP_COUNTS[dataset],
                }
            )
    return output


def derive_robust_shortlist(
    objective_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    objectives, pool = _validated_sources(objective_rows, pool_rows)
    fronts, support = per_seed_pareto(objectives)
    robust = robust_four_objective_pareto(objectives)
    balanced = aggregate_balanced_rank(objectives)

    memberships: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in support:
        if row["selected"]:
            memberships[_identity(row)].add("PARETO_SUPPORT_AT_LEAST_TWO")
    for row in robust:
        memberships[_identity(row)].add("ROBUST_PARETO_4D")
    for row in balanced:
        if row["selected"]:
            memberships[_identity(row)].add("AGGREGATE_BALANCED_TOP")
    for dataset, controls in CONTROLS.items():
        for role, (candidate_id, candidate_hash) in controls.items():
            source = pool.get((dataset, candidate_hash))
            if source is None or str(source["candidate_set_id"]) != candidate_id:
                raise ValueError(f"missing exact H7 control identity: {dataset}/{role}")
            memberships[(dataset, candidate_hash)].add(role.upper())

    component_rows: list[dict[str, Any]] = []
    shortlist: list[dict[str, Any]] = []
    for key in sorted(memberships):
        source = pool[key]
        components = sorted(memberships[key])
        for component in components:
            component_rows.append(
                {
                    "dataset": key[0],
                    "path_set_id": str(source["candidate_set_id"]),
                    "path_set_hash": key[1],
                    "candidate_set_ordinal": int(source["candidate_set_ordinal"]),
                    "component": component,
                }
            )
        shortlist.append(
            {
                "dataset": key[0],
                "path_set_id": str(source["candidate_set_id"]),
                "path_set_hash": key[1],
                "candidate_set_id": str(source["candidate_set_id"]),
                "candidate_set_hash": key[1],
                "candidate_set_ordinal": int(source["candidate_set_ordinal"]),
                "components_json": json.dumps(components, ensure_ascii=False),
                "is_manual": _bool(source["is_manual"]),
                "is_current_automatic": _bool(source["is_current_automatic"]),
                "truth_fields_read": False,
                "h5_locked_report_access": False,
                "s_eval_access": False,
            }
        )
    shortlist.sort(key=lambda row: (row["dataset"], row["path_set_hash"]))
    counts = {
        dataset: sum(row["dataset"] == dataset for row in shortlist)
        for dataset in ("acm", "freebase")
    }
    identity_basis = [
        {
            "dataset": row["dataset"],
            "path_set_id": row["path_set_id"],
            "path_set_hash": row["path_set_hash"],
        }
        for row in shortlist
    ]
    identity = canonical_json_hash(identity_basis)
    if counts != EXPECTED_SHORTLIST_COUNTS or len(shortlist) != EXPECTED_SHORTLIST_TOTAL:
        raise RuntimeError("BLOCKED_ROBUST_SHORTLIST_IDENTITY_MISMATCH: counts")
    if identity != EXPECTED_SHORTLIST_IDENTITY:
        raise RuntimeError(
            f"BLOCKED_ROBUST_SHORTLIST_IDENTITY_MISMATCH: {identity}"
        )
    return {
        "per_seed_fronts": fronts,
        "support": support,
        "robust_four_objective": robust,
        "balanced_rank": balanced,
        "components": component_rows,
        "shortlist": shortlist,
        "identity_basis": identity_basis,
        "identity": identity,
        "counts": {**counts, "total": len(shortlist)},
    }


def _normalized_midranks(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    ordered = sorted(
        ((str(row["candidate_set_hash"]), _finite(row[field])) for row in rows),
        key=lambda item: (-item[1], item[0]),
    )
    if len(ordered) == 1:
        return {ordered[0][0]: 0.0}
    output: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        midrank = ((index + 1) + end) / 2.0
        normalized = (midrank - 1.0) / (len(ordered) - 1.0)
        for position in range(index, end):
            output[ordered[position][0]] = normalized
        index = end
    return output


def aggregate_r4_objectives(
    search_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(search_rows) != EXPECTED_SHORTLIST_TOTAL * len(DEVELOPMENT_MODEL_SEEDS):
        raise ValueError("H7 R4 search metric row count mismatch")
    if {str(row["dataset"]) for row in search_rows} != {"acm", "freebase"}:
        raise ValueError("H7 R4 search metric dataset set mismatch")
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in search_rows:
        grouped[_identity(row)].append(dict(row))
    output: list[dict[str, Any]] = []
    for (dataset, candidate_hash), rows in sorted(grouped.items()):
        seeds = {int(row["model_seed"]) for row in rows}
        if seeds != set(DEVELOPMENT_MODEL_SEEDS) or len(rows) != len(DEVELOPMENT_MODEL_SEEDS):
            raise ValueError(f"incomplete H7 R4 seed cohort: {dataset}/{candidate_hash}")
        cls_values = [_finite(row["mean_F_cls"]) for row in rows]
        clu_values = [_finite(row["mean_F_clu"]) for row in rows]
        source = rows[0]
        output.append(
            {
                "dataset": dataset,
                "candidate_set_id": str(source["candidate_set_id"]),
                "candidate_set_hash": candidate_hash,
                "candidate_set_ordinal": int(source["candidate_set_ordinal"]),
                "seed_count": len(rows),
                "Q_cls_R4": median(cls_values),
                "Q_clu_R4": median(clu_values),
                "W_cls_R4": min(cls_values),
                "W_clu_R4": min(clu_values),
                "mean_F_cls_R4": mean(cls_values),
                "mean_F_clu_R4": mean(clu_values),
                "sample_sd_F_cls_R4": stdev(cls_values),
                "sample_sd_F_clu_R4": stdev(clu_values),
                "assignment_stability_mean": mean(
                    _finite(row["assignment_stability_mean"]) for row in rows
                ),
                "per_seed_json": json.dumps(
                    [
                        {
                            "model_seed": int(row["model_seed"]),
                            "F_cls": _finite(row["mean_F_cls"]),
                            "F_clu": _finite(row["mean_F_clu"]),
                        }
                        for row in sorted(rows, key=lambda item: int(item["model_seed"]))
                    ],
                    sort_keys=True,
                ),
            }
        )
    return output


def manual_safety_audit(
    objective_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_identity = {_identity(row): dict(row) for row in objective_rows}
    output: list[dict[str, Any]] = []
    for row in objective_rows:
        dataset = str(row["dataset"])
        manual_hash = CONTROLS[dataset]["manual"][1]
        manual = by_identity.get((dataset, manual_hash))
        if manual is None:
            raise ValueError(f"missing H7 Manual R4 row: {dataset}")
        candidate_seed = {
            int(item["model_seed"]): item for item in json.loads(str(row["per_seed_json"]))
        }
        manual_seed = {
            int(item["model_seed"]): item for item in json.loads(str(manual["per_seed_json"]))
        }
        candidate_payload = json.loads(str(row["per_seed_json"]))
        manual_payload = json.loads(str(manual["per_seed_json"]))
        if (
            len(candidate_payload) != len(DEVELOPMENT_MODEL_SEEDS)
            or len(manual_payload) != len(DEVELOPMENT_MODEL_SEEDS)
            or set(candidate_seed) != set(DEVELOPMENT_MODEL_SEEDS)
            or set(manual_seed) != set(DEVELOPMENT_MODEL_SEEDS)
        ):
            raise ValueError("H7 Manual safety requires exact paired seeds")
        delta_cls = [
            _finite(candidate_seed[seed]["F_cls"]) - _finite(manual_seed[seed]["F_cls"])
            for seed in DEVELOPMENT_MODEL_SEEDS
        ]
        delta_clu = [
            _finite(candidate_seed[seed]["F_clu"]) - _finite(manual_seed[seed]["F_clu"])
            for seed in DEVELOPMENT_MODEL_SEEDS
        ]
        is_manual = str(row["candidate_set_hash"]) == manual_hash
        se_cls = stdev(delta_cls) / math.sqrt(len(delta_cls))
        se_clu = stdev(delta_clu) / math.sqrt(len(delta_clu))
        passed = is_manual or (
            mean(delta_cls) >= -se_cls
            and mean(delta_clu) >= -se_clu
            and min(delta_cls) > -R4_PER_SEED_DROP_LIMIT
            and min(delta_clu) > -R4_PER_SEED_DROP_LIMIT
        )
        output.append(
            {
                "dataset": dataset,
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "manual_candidate_set_hash": manual_hash,
                "mean_delta_cls": mean(delta_cls),
                "mean_delta_clu": mean(delta_clu),
                "paired_SE_cls": se_cls,
                "paired_SE_clu": se_clu,
                "minimum_delta_cls": min(delta_cls),
                "minimum_delta_clu": min(delta_clu),
                "single_seed_disaster": (
                    min(delta_cls) <= -R4_PER_SEED_DROP_LIMIT
                    or min(delta_clu) <= -R4_PER_SEED_DROP_LIMIT
                ),
                "is_manual": is_manual,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return sorted(output, key=lambda row: (row["dataset"], row["candidate_set_hash"]))


def robust_r4_selection(
    objective_rows: Sequence[Mapping[str, Any]],
    safety_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    objective_keys = [_identity(row) for row in objective_rows]
    safety_keys = [_identity(row) for row in safety_rows]
    if (
        len(objective_keys) != len(set(objective_keys))
        or len(safety_keys) != len(set(safety_keys))
        or set(objective_keys) != set(safety_keys)
    ):
        raise ValueError("H7 R4 objective/safety identity mismatch")
    safety = {_identity(row): str(row["status"]) for row in safety_rows}
    fronts: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    unsafe_datasets: list[str] = []
    for dataset in ("acm", "freebase"):
        safe = [
            dict(row)
            for row in objective_rows
            if str(row["dataset"]) == dataset and safety.get(_identity(row)) == "PASS"
        ]
        if not safe:
            unsafe_datasets.append(dataset)
            continue
        ranks = {
            field: _normalized_midranks(safe, field)
            for field in ("Q_cls_R4", "Q_clu_R4", "W_cls_R4", "W_clu_R4")
        }
        enriched: list[dict[str, Any]] = []
        for row in safe:
            candidate_hash = str(row["candidate_set_hash"])
            r_median = max(ranks["Q_cls_R4"][candidate_hash], ranks["Q_clu_R4"][candidate_hash])
            r_worst = max(ranks["W_cls_R4"][candidate_hash], ranks["W_clu_R4"][candidate_hash])
            enriched.append(
                {
                    **row,
                    "R_Q_cls": ranks["Q_cls_R4"][candidate_hash],
                    "R_Q_clu": ranks["Q_clu_R4"][candidate_hash],
                    "R_W_cls": ranks["W_cls_R4"][candidate_hash],
                    "R_W_clu": ranks["W_clu_R4"][candidate_hash],
                    "R_median": r_median,
                    "R_worst": r_worst,
                    "R_robust": max(r_median, r_worst),
                }
            )
        first = _first_front(enriched, ("Q_cls_R4", "Q_clu_R4", "W_cls_R4", "W_clu_R4"))
        first_hashes = {str(row["candidate_set_hash"]) for row in first}
        ordered = sorted(
            enriched,
            key=lambda row: (
                row["R_robust"],
                mean((row["R_median"], row["R_worst"])),
                -min(row["W_cls_R4"], row["W_clu_R4"]),
                -row["assignment_stability_mean"],
                row["candidate_set_hash"],
            ),
        )
        ordered_front = [row for row in ordered if str(row["candidate_set_hash"]) in first_hashes]
        choice = dict(ordered_front[0])
        selected.append(choice)
        for row in enriched:
            rankings.append({**row, "first_robust_pareto_front": row["candidate_set_hash"] in first_hashes})
        fronts.extend({**row, "front_rank": 1} for row in first)

    if unsafe_datasets:
        return {
            "fronts": sorted(fronts, key=lambda row: (row["dataset"], row["candidate_set_hash"])),
            "rankings": sorted(rankings, key=lambda row: (row["dataset"], row["candidate_set_hash"])),
            "selected": [],
            "unsafe_datasets": unsafe_datasets,
            "status": "TC_LPMSO_H7_NO_SAFE_SET",
            "s_eval_started": False,
        }
    if len(selected) != 2 or {row["dataset"] for row in selected} != {"acm", "freebase"}:
        raise RuntimeError("H7 must select one complete path set per dataset")
    kinds: list[str] = []
    for row in selected:
        dataset = str(row["dataset"])
        candidate_hash = str(row["candidate_set_hash"])
        if candidate_hash == CONTROLS[dataset]["manual"][1]:
            kinds.append("manual")
        elif candidate_hash == CONTROLS[dataset]["current_automatic"][1]:
            kinds.append("current_automatic")
        else:
            kinds.append("new")
    if "new" in kinds:
        status = "TC_LPMSO_H7_NEW_SET_SELECTED"
    elif "current_automatic" in kinds:
        status = "TC_LPMSO_H7_SELECTED_CURRENT_AUTOMATIC"
    else:
        status = "TC_LPMSO_H7_SELECTED_MANUAL"
    return {
        "fronts": sorted(fronts, key=lambda row: (row["dataset"], row["candidate_set_hash"])),
        "rankings": sorted(rankings, key=lambda row: (row["dataset"], row["candidate_set_hash"])),
        "selected": sorted(selected, key=lambda row: row["dataset"]),
        "unsafe_datasets": [],
        "status": status,
        "s_eval_started": False,
    }
