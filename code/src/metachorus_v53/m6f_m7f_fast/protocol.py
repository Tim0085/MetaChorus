"""Frozen search selection and three-seed preliminary decision protocol."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import canonical_json_hash

from .constants import EVAL_MODEL_SEEDS, PRELIM_ENDPOINT


METRICS = ("Micro_F1", "Macro_F1", "NMI", "ARI", "F_cls", "F_clu", "F_joint")
OBJECTIVES = ("Q_cls", "Q_clu", "W_cls", "W_clu")


def metric_values(micro_f1: float, macro_f1: float, nmi: float, ari: float) -> dict[str, float]:
    values = [float(micro_f1), float(macro_f1), float(nmi), float(ari)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("metric values must be finite")
    f_cls = (values[0] + values[1]) / 2.0
    f_clu = (values[2] + values[3]) / 2.0
    return {
        "Micro_F1": values[0],
        "Macro_F1": values[1],
        "NMI": values[2],
        "ARI": values[3],
        "F_cls": f_cls,
        "F_clu": f_clu,
        "F_joint": (f_cls + f_clu) / 2.0,
    }


def manual_safety_gate(candidate_rows: Sequence[Mapping[str, Any]], manual_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = {int(row["model_seed"]): row for row in candidate_rows}
    manual = {int(row["model_seed"]): row for row in manual_rows}
    if len(candidate) != 3 or set(candidate) != set(manual):
        raise ValueError("Manual safety gate requires exactly three paired search seeds")
    delta_cls = [float(candidate[seed]["F_cls"]) - float(manual[seed]["F_cls"]) for seed in sorted(candidate)]
    delta_clu = [float(candidate[seed]["F_clu"]) - float(manual[seed]["F_clu"]) for seed in sorted(candidate)]
    se_cls = statistics.stdev(delta_cls) / math.sqrt(3)
    se_clu = statistics.stdev(delta_clu) / math.sqrt(3)
    passed = (
        statistics.mean(delta_cls) >= -se_cls
        and statistics.mean(delta_clu) >= -se_clu
        and min(delta_cls) > -0.02
        and min(delta_clu) > -0.02
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "paired_seed_count": 3,
        "mean_delta_F_cls": statistics.mean(delta_cls),
        "mean_delta_F_clu": statistics.mean(delta_clu),
        "paired_SE_F_cls": se_cls,
        "paired_SE_F_clu": se_clu,
        "minimum_delta_F_cls": min(delta_cls),
        "minimum_delta_F_clu": min(delta_clu),
        "per_seed_drop_limit": 0.02,
    }


def close_search_selection(
    *,
    dataset: str,
    metric_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    manual_set: Mapping[str, Any],
    current_automatic: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the final Manual gate and H7-frozen robust selector once."""

    candidates = {str(row["candidate_set_hash"]): dict(row) for row in candidate_rows}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        if str(row.get("dataset")) != dataset:
            raise ValueError("search selection metric dataset mismatch")
        grouped[str(row["candidate_set_hash"])].append(dict(row))
    if set(grouped) != set(candidates):
        raise ValueError("final search metrics/candidate shortlist mismatch")
    manual_hash = str(manual_set["candidate_set_hash"])
    if manual_hash not in grouped:
        raise ValueError("final search shortlist does not contain Manual")
    manual_rows = grouped[manual_hash]
    objective_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    for digest, rows in sorted(grouped.items()):
        objective = aggregate_objectives(rows)
        safety = (
            {"status": "PASS", "passed": True, "manual_identity": True}
            if digest == manual_hash
            else manual_safety_gate(rows, manual_rows)
        )
        role = "MANUAL" if digest == manual_hash else "CANDIDATE"
        if (
            current_automatic.get("status") == "EXACT_IDENTITY_RESOLVED"
            and digest == str(current_automatic.get("candidate_set_hash"))
        ):
            role = "CURRENT_AUTOMATIC"
        stability = statistics.mean(float(row.get("assignment_stability_mean", 0.0)) for row in rows)
        objective_rows.append(
            {
                **candidates[digest],
                **objective,
                "dataset": dataset,
                "role": role,
                "assignment_stability_mean": stability,
                "manual_safety_status": safety["status"],
            }
        )
        safety_rows.append(
            {
                "dataset": dataset,
                "candidate_set_hash": digest,
                "candidate_set_id": candidates[digest]["candidate_set_id"],
                **safety,
            }
        )
    selected = select_unique_robust_set(objective_rows)
    if selected["selected_set"] is None:
        raise RuntimeError("SEARCH_FAILED: Manual itself was not a safe registered candidate")
    basis = {
        "dataset": dataset,
        "status": selected["status"],
        "selected_set": selected["selected_set"],
        "manual_set": dict(manual_set),
        "current_automatic": dict(current_automatic),
        "objective_rows": objective_rows,
        "manual_safety_rows": safety_rows,
        "safe_set_count": selected["safe_set_count"],
        "selection_identity": selected["selection_identity"],
        "no_second_choice": True,
        "algorithm_retuning": False,
    }
    return {"status": selected["status"], "basis": basis, "identity": canonical_json_hash(basis)}


def aggregate_objectives(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3 or len({int(row["model_seed"]) for row in rows}) != 3:
        raise ValueError("four-objective aggregation requires exactly three unique seeds")
    cls = [float(row["F_cls"]) for row in rows]
    clu = [float(row["F_clu"]) for row in rows]
    if not all(math.isfinite(value) for value in cls + clu):
        raise ValueError("non-finite search metric")
    return {
        "Q_cls": statistics.median(cls),
        "Q_clu": statistics.median(clu),
        "W_cls": min(cls),
        "W_clu": min(clu),
    }


def derive_r1_high_recall_pool(
    metric_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    manual_candidate_set_hash: str,
    current_automatic_candidate_set_hash: str | None = None,
) -> dict[str, Any]:
    """Freeze the >40-set R1 exploitation/exploration union.

    Classification ranking is the three-seed mean F_cls, descending with the
    candidate hash as the final tie-break.  The non-exploitation remainder is
    split into four equal ordinal bands; each band uses a frozen hash order.
    """

    candidates = {str(row["candidate_set_hash"]): dict(row) for row in candidate_rows}
    if len(candidates) != len(candidate_rows) or len(candidates) <= 40:
        raise ValueError("R1 high-recall pool requires more than 40 unique candidate sets")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        digest = str(row["candidate_set_hash"])
        if digest not in candidates:
            raise ValueError("R1 metric row references an unknown candidate set")
        grouped[digest].append(dict(row))
    if set(grouped) != set(candidates):
        raise ValueError("R1 metric matrix does not cover the full candidate bank")
    aggregates: list[dict[str, Any]] = []
    for digest, rows in grouped.items():
        seeds = {int(row["model_seed"]) for row in rows}
        if len(rows) != 3 or len(seeds) != 3:
            raise ValueError("R1 requires exactly three model seeds per candidate")
        values = [float(row["F_cls"]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("R1 classification ranking contains non-finite values")
        aggregates.append(
            {
                **candidates[digest],
                "mean_F_cls_R1": statistics.mean(values),
                "minimum_F_cls_R1": min(values),
            }
        )
    ranked = sorted(aggregates, key=lambda row: (-float(row["mean_F_cls_R1"]), str(row["candidate_set_hash"])))
    exploit_count = min(80, math.ceil(0.40 * len(ranked)))
    explore_count = min(40, math.ceil(0.20 * len(ranked)))
    exploitation = ranked[:exploit_count]
    remainder = ranked[exploit_count:]
    band_sizes = [len(remainder) // 4 + int(index < len(remainder) % 4) for index in range(4)]
    bands: list[list[dict[str, Any]]] = []
    cursor = 0
    for size in band_sizes:
        bands.append(remainder[cursor : cursor + size])
        cursor += size
    exploration: list[dict[str, Any]] = []
    for band_index, band in enumerate(bands):
        quota = explore_count // 4 + int(band_index < explore_count % 4)
        ordered = sorted(
            band,
            key=lambda row: canonical_json_hash(
                {
                    "namespace": "M6F_R1_CLASSIFICATION_BAND_EXPLORATION",
                    "band_index": band_index,
                    "candidate_set_hash": row["candidate_set_hash"],
                }
            ),
        )
        exploration.extend(ordered[:quota])
    components: defaultdict[str, set[str]] = defaultdict(set)
    for row in exploitation:
        components[str(row["candidate_set_hash"])].add("R1_CLASSIFICATION_EXPLOITATION")
    for row in exploration:
        components[str(row["candidate_set_hash"])].add("R1_STRATIFIED_EXPLORATION")
    controls = [(manual_candidate_set_hash, "MANUAL")]
    if current_automatic_candidate_set_hash:
        controls.append((current_automatic_candidate_set_hash, "CURRENT_AUTOMATIC"))
    for digest, role in controls:
        if digest not in candidates:
            raise ValueError(f"R1 control is absent from the candidate bank: {role}")
        components[digest].add(role)
    selected = []
    for digest in sorted(components):
        selected.append(
            {
                **candidates[digest],
                "components": sorted(components[digest]),
                "selected_for_r2_r3": True,
            }
        )
    basis = {
        "candidate_set_count": len(candidates),
        "exploitation_count": exploit_count,
        "exploration_count": len(exploration),
        "high_recall_pool_count": len(selected),
        "ranking_rule": "MEAN_F_CLS_DESC_THEN_CANDIDATE_HASH",
        "remainder_band_count": 4,
        "selection_identity": canonical_json_hash(
            [{"candidate_set_hash": row["candidate_set_hash"], "components": row["components"]} for row in selected]
        ),
    }
    return {"status": "PASS", "basis": basis, "ranking": ranked, "selected": selected}


def derive_robust_shortlist(
    metric_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    manual_candidate_set_hash: str,
    current_automatic_candidate_set_hash: str | None = None,
) -> dict[str, Any]:
    """Derive the frozen R3 shortlist union at update 2000."""

    candidates = {str(row["candidate_set_hash"]): dict(row) for row in candidate_rows}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        digest = str(row["candidate_set_hash"])
        if digest not in candidates:
            raise ValueError("R3 metric row references an unknown candidate")
        grouped[digest].append(dict(row))
    if set(grouped) != set(candidates):
        raise ValueError("R3 metrics do not exactly cover the high-recall pool")
    objectives: list[dict[str, Any]] = []
    support_count: defaultdict[str, int] = defaultdict(int)
    for digest, rows in grouped.items():
        if len(rows) != 3 or len({int(row["model_seed"]) for row in rows}) != 3:
            raise ValueError("R3 shortlist requires three seeds per candidate")
        objective = aggregate_objectives(rows)
        objectives.append({**candidates[digest], **objective})
    seeds = sorted({int(row["model_seed"]) for row in metric_rows})
    if len(seeds) != 3:
        raise ValueError("R3 shortlist seed matrix is not exact")
    for seed in seeds:
        cohort = [dict(row) for row in metric_rows if int(row["model_seed"]) == seed]
        for row in first_pareto_front(cohort, ("F_cls", "F_clu")):
            support_count[str(row["candidate_set_hash"])] += 1
    robust_hashes = {str(row["candidate_set_hash"]) for row in first_pareto_front(objectives)}
    cls_ranks = normalized_midranks(objectives, "Q_cls")
    clu_ranks = normalized_midranks(objectives, "Q_clu")
    balanced = sorted(
        objectives,
        key=lambda row: (
            max(cls_ranks[str(row["candidate_set_hash"])], clu_ranks[str(row["candidate_set_hash"])]),
            (cls_ranks[str(row["candidate_set_hash"])] + clu_ranks[str(row["candidate_set_hash"])]) / 2.0,
            str(row["candidate_set_hash"]),
        ),
    )
    balanced_count = max(3, math.ceil(0.20 * len(objectives)))
    components: defaultdict[str, set[str]] = defaultdict(set)
    for digest, count in support_count.items():
        if count >= 2:
            components[digest].add("PER_SEED_PARETO_SUPPORT_AT_LEAST_TWO")
    for digest in robust_hashes:
        components[digest].add("MEDIAN_WORST_FOUR_OBJECTIVE_PARETO")
    for row in balanced[:balanced_count]:
        components[str(row["candidate_set_hash"])].add("AGGREGATE_BALANCED_TOP")
    controls = [(manual_candidate_set_hash, "MANUAL")]
    if current_automatic_candidate_set_hash:
        controls.append((current_automatic_candidate_set_hash, "CURRENT_AUTOMATIC"))
    for digest, role in controls:
        if digest not in candidates:
            raise ValueError(f"R3 control is absent from the high-recall pool: {role}")
        components[digest].add(role)
    shortlist = [
        {**candidates[digest], "components": sorted(components[digest])}
        for digest in sorted(components)
    ]
    return {
        "status": "PASS",
        "shortlist": shortlist,
        "objectives": objectives,
        "pareto_support_count": dict(sorted(support_count.items())),
        "balanced_top_count": balanced_count,
        "shortlist_identity": canonical_json_hash(
            [{"candidate_set_hash": row["candidate_set_hash"], "components": row["components"]} for row in shortlist]
        ),
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]) -> bool:
    return all(float(left[key]) >= float(right[key]) for key in fields) and any(
        float(left[key]) > float(right[key]) for key in fields
    )


def first_pareto_front(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str] = OBJECTIVES
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    if not fields or any(not math.isfinite(float(row[field])) for row in values for field in fields):
        raise ValueError("Pareto front fields must be finite and non-empty")
    return [
        row
        for index, row in enumerate(values)
        if not any(
            _dominates(other, row, fields)
            for other_index, other in enumerate(values)
            if other_index != index
        )
    ]


def normalized_midranks(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = sorted(
        [(str(row["candidate_set_hash"]), float(row[field])) for row in rows],
        key=lambda item: (-item[1], item[0]),
    )
    if len(values) == 1:
        return {values[0][0]: 0.0}
    result: dict[str, float] = {}
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[end][1] == values[cursor][1]:
            end += 1
        
        midrank = ((cursor + 1) + end) / 2.0
        normalized = (midrank - 1.0) / (len(values) - 1.0)
        for index in range(cursor, end):
            result[values[index][0]] = normalized
        cursor = end
    return result


def select_unique_robust_set(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe = [dict(row) for row in rows if row.get("manual_safety_status") == "PASS"]
    if not safe:
        return {"status": "SEARCH_FAILED", "selected_set": None, "safe_set_count": 0}
    front = first_pareto_front(safe)
    rank_by_field = {field: normalized_midranks(front, field) for field in OBJECTIVES}
    ranked: list[dict[str, Any]] = []
    for row in front:
        digest = str(row["candidate_set_hash"])
        r_median = max(rank_by_field["Q_cls"][digest], rank_by_field["Q_clu"][digest])
        r_worst = max(rank_by_field["W_cls"][digest], rank_by_field["W_clu"][digest])
        ranked.append(
            {
                **row,
                "R_median": r_median,
                "R_worst": r_worst,
                "R_robust": max(r_median, r_worst),
            }
        )
    ranked.sort(
        key=lambda row: (
            float(row["R_robust"]),
            (float(row["R_median"]) + float(row["R_worst"])) / 2.0,
            -min(float(row["W_cls"]), float(row["W_clu"])),
            -float(row.get("assignment_stability_mean", 0.0)),
            str(row["candidate_set_hash"]),
        )
    )
    selected = ranked[0]
    role = str(selected.get("role", "CANDIDATE"))
    status = {
        "MANUAL": "MANUAL_SELECTED",
        "CURRENT_AUTOMATIC": "CURRENT_AUTOMATIC_SELECTED",
        "TRIVIAL": "TRIVIAL_SINGLE_SET",
    }.get(role, "NEW_SET_SELECTED")
    return {
        "status": status,
        "selected_set": selected,
        "safe_set_count": len(safe),
        "first_robust_pareto_front_count": len(front),
        "ranking": ranked,
        "selection_identity": canonical_json_hash(selected),
    }


def preliminary_decision(candidate_rows: Sequence[Mapping[str, Any]], manual_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_rows = [dict(row) for row in candidate_rows] + [dict(row) for row in manual_rows]
    datasets = {str(row.get("dataset", "")) for row in all_rows}
    candidate_hashes = {str(row.get("candidate_set_hash", "")) for row in candidate_rows}
    manual_hashes = {str(row.get("candidate_set_hash", "")) for row in manual_rows}
    if (
        len(datasets) != 1
        or "" in datasets
        or len(candidate_hashes) != 1
        or "" in candidate_hashes
        or len(manual_hashes) != 1
        or "" in manual_hashes
        or any(row.get("status") != "PASS" for row in all_rows)
        or any(int(row.get("endpoint_update", -1)) != PRELIM_ENDPOINT for row in all_rows)
        or {str(row.get("logical_role")) for row in candidate_rows} != {"FINAL_FROZEN_METHOD"}
        or {str(row.get("logical_role")) for row in manual_rows} != {"MANUAL"}
    ):
        raise ValueError("preliminary decision arm identities or roles are not exact")
    candidate = {int(row["model_seed"]): row for row in candidate_rows}
    manual = {int(row["model_seed"]): row for row in manual_rows}
    if len(candidate) != 3 or set(candidate) != set(manual) or set(candidate) != set(EVAL_MODEL_SEEDS):
        raise ValueError("preliminary decision requires exactly three paired evaluation seeds")
    for row in all_rows:
        f_cls = float(row["F_cls"])
        f_clu = float(row["F_clu"])
        f_joint = float(row["F_joint"])
        if not all(math.isfinite(value) for value in (f_cls, f_clu, f_joint)) or not math.isclose(
            f_joint, (f_cls + f_clu) / 2.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("preliminary decision metric formula is invalid")
    deltas = {
        metric: [float(candidate[seed][metric]) - float(manual[seed][metric]) for seed in sorted(candidate)]
        for metric in ("F_cls", "F_clu", "F_joint")
    }
    means = {metric: statistics.mean(values) for metric, values in deltas.items()}
    minimum = {metric: min(values) for metric, values in deltas.items()}
    non_worse = sum(value >= 0.0 for value in deltas["F_joint"])
    disaster = minimum["F_cls"] <= -0.02 or minimum["F_clu"] <= -0.02
    unsafe = means["F_cls"] < -0.01 or means["F_clu"] < -0.01 or disaster
    promising = (
        means["F_joint"] > 0.0
        and (means["F_cls"] > 0.0 or means["F_clu"] > 0.0)
        and means["F_cls"] >= -0.01
        and means["F_clu"] >= -0.01
        and non_worse >= 2
        and not disaster
    )
    noninferior = (
        means["F_cls"] >= -0.01
        and means["F_clu"] >= -0.01
        and means["F_joint"] >= -0.005
        and not disaster
    )
    
    
    if unsafe:
        status = "UNSAFE"
    elif promising:
        status = "PROMISING"
    elif noninferior:
        status = "NONINFERIOR"
    else:
        status = "INDETERMINATE"
    return {
        "status": status,
        "mean_delta_F_cls": means["F_cls"],
        "mean_delta_F_clu": means["F_clu"],
        "mean_delta_F_joint": means["F_joint"],
        "minimum_delta_F_cls": minimum["F_cls"],
        "minimum_delta_F_clu": minimum["F_clu"],
        "joint_non_worse_count": non_worse,
        "paired_seed_count": 3,
        "confirmatory_claim_allowed": False,
    }


__all__ = [
    "METRICS",
    "OBJECTIVES",
    "metric_values",
    "manual_safety_gate",
    "close_search_selection",
    "aggregate_objectives",
    "derive_r1_high_recall_pool",
    "derive_robust_shortlist",
    "first_pareto_front",
    "normalized_midranks",
    "select_unique_robust_set",
    "preliminary_decision",
]
