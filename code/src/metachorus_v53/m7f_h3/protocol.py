"""H3 budgeted-pool composition around the immutable FAST protocol functions."""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.m6f_m7f_fast import protocol as frozen_protocol
from metachorus_v53.m6f_m7f_fast.protocol import (
    OBJECTIVES,
    aggregate_objectives,
    close_search_selection,
    first_pareto_front,
    manual_safety_gate,
    normalized_midranks,
    preliminary_decision,
    select_unique_robust_set,
)

from .common import canonical_json_hash, envelope, sha256_file, write_json_once, write_text_once
from .constants import (
    ALL_SETS_IDENTITY,
    DATASET,
    FROZEN_PROTOCOL_RELATIVE_PATH,
    FROZEN_PROTOCOL_SHA256,
    MANUAL_SET_HASH,
    R1_MODEL_SEEDS,
    R1_POOL_SIZE,
    R3_SHORTLIST_SIZE,
    SEARCH_MODEL_SEEDS,
    TASKBOOK_SHA256,
)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("H3 ranking metric must be finite")
    return result


def _balanced_order(rows: Sequence[Mapping[str, Any]], cls_field: str, clu_field: str) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    """Order rows by the frozen normalized-midranks implementation."""

    values = [dict(row) for row in rows]
    cls_ranks = normalized_midranks(values, cls_field)
    clu_ranks = normalized_midranks(values, clu_field)
    ordered = sorted(
        values,
        key=lambda row: (
            max(cls_ranks[str(row["candidate_set_hash"])], clu_ranks[str(row["candidate_set_hash"])]),
            (cls_ranks[str(row["candidate_set_hash"])] + clu_ranks[str(row["candidate_set_hash"])] ) / 2.0,
            str(row["candidate_set_hash"]),
        ),
    )
    return ordered, cls_ranks, clu_ranks


def derive_r1_pool(
    metric_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact sealed 24-set pool from the one-seed 220-set R1."""

    candidates = {str(row["candidate_set_hash"]): dict(row) for row in candidate_rows}
    if len(candidates) != 220 or MANUAL_SET_HASH not in candidates:
        raise ValueError("H3 R1 pool requires the exact 220-set bank including Manual")
    rows = [dict(row) for row in metric_rows]
    if len(rows) != 220 or len({str(row.get("candidate_set_hash")) for row in rows}) != 220:
        raise ValueError("H3 R1 metrics must cover 220 unique sets once")
    for row in rows:
        digest = str(row.get("candidate_set_hash"))
        if (
            digest not in candidates
            or str(row.get("dataset")) != DATASET
            or int(row.get("model_seed", -1)) != R1_MODEL_SEEDS[0]
            or int(row.get("endpoint_update", -1)) != 300
            or str(row.get("metric_role")) != "GLOBAL_SEARCH_POOL_ONLY"
            or bool(row.get("prelim_locked_report_access", False))
        ):
            raise ValueError("H3 R1 metric identity or label role is invalid")
        _finite(row["F_cls"])
        _finite(row["F_clu"])
    ranked_cls = sorted(rows, key=lambda row: (-_finite(row["F_cls"]), str(row["candidate_set_hash"])))
    ranked_clu = sorted(rows, key=lambda row: (-_finite(row["F_clu"]), str(row["candidate_set_hash"])))
    balanced, cls_ranks, clu_ranks = _balanced_order(rows, "F_cls", "F_clu")
    metrics_identity = canonical_json_hash(sorted(rows, key=lambda row: str(row["candidate_set_hash"])))
    components: defaultdict[str, set[str]] = defaultdict(set)
    components[MANUAL_SET_HASH].add("MANUAL")
    for row in ranked_cls[:7]:
        components[str(row["candidate_set_hash"])].add("TOP7_F_CLS")
    for row in ranked_clu[:7]:
        components[str(row["candidate_set_hash"])].add("TOP7_F_CLU")
    for row in balanced[:5]:
        components[str(row["candidate_set_hash"])].add("TOP5_BALANCED_MINIMAX_RANK")
    remaining = [row for row in balanced if str(row["candidate_set_hash"]) not in components]
    band_sizes = [len(remaining) // 4 + int(index < len(remaining) % 4) for index in range(4)]
    cursor = 0
    exploration_basis = {
        "taskbook_sha256": TASKBOOK_SHA256,
        "all_sets_identity": ALL_SETS_IDENTITY,
        "r1_metrics_identity": metrics_identity,
        "namespace": "DBLP_H3_R1_EXPLORATION",
    }
    exploration_order_rows: list[dict[str, Any]] = []
    for band_index, size in enumerate(band_sizes):
        band = remaining[cursor : cursor + size]
        cursor += size
        if not band:
            raise RuntimeError("H3 R1 exploration interval is unexpectedly empty")
        frozen_order = sorted(
            band,
            key=lambda row: (
                canonical_json_hash(
                    {
                        **exploration_basis,
                        "band_index": band_index,
                        "candidate_set_hash": row["candidate_set_hash"],
                    }
                ),
                str(row["candidate_set_hash"]),
            ),
        )
        exploration_order_rows.extend(
            {
                "band_index": band_index,
                "frozen_order_position": position,
                "candidate_set_hash": str(row["candidate_set_hash"]),
            }
            for position, row in enumerate(frozen_order)
        )
        chosen = frozen_order[0]
        components[str(chosen["candidate_set_hash"])].add(f"FROZEN_EXPLORATION_BAND_{band_index}")
    for row in balanced:
        if len(components) >= R1_POOL_SIZE:
            break
        components[str(row["candidate_set_hash"])].add("BALANCED_RANK_TOP_UP")
    if len(components) != R1_POOL_SIZE:
        raise RuntimeError("H3 R1 pool is not exactly 24 unique sets")
    balanced_rank = {
        str(row["candidate_set_hash"]): index + 1 for index, row in enumerate(balanced)
    }
    component_rows = []
    selected = []
    for digest in sorted(components):
        rank = balanced_rank[digest]
        component_rows.append(
            {
                "dataset": DATASET,
                "candidate_set_id": candidates[digest]["candidate_set_id"],
                "candidate_set_hash": digest,
                "components": sorted(components[digest]),
                "component_count": len(components[digest]),
                "balanced_rank": rank,
                "normalized_rank_F_cls": cls_ranks[digest],
                "normalized_rank_F_clu": clu_ranks[digest],
            }
        )
        selected.append({**candidates[digest], "components": sorted(components[digest]), "balanced_rank": rank})
    selected.sort(key=lambda row: (int(row["balanced_rank"]), str(row["candidate_set_hash"])))
    seal_basis = {
        "dataset": DATASET,
        "status": "SEALED_AFTER_R1_BEFORE_R2",
        "selected_candidate_set_count": len(selected),
        "selected_candidate_hashes": sorted(components),
        "selected_rows_identity": canonical_json_hash(selected),
        "component_rows_identity": canonical_json_hash(component_rows),
        "r1_metrics_identity": metrics_identity,
        "all_sets_identity": ALL_SETS_IDENTITY,
        "exploration_basis": exploration_basis,
        "exploration_order_identity": canonical_json_hash(exploration_order_rows),
        "exploration_order_row_count": len(exploration_order_rows),
        "metric_role": "GLOBAL_SEARCH_POOL_ONLY",
        "prelim_locked_report_access": False,
        "manual_included": MANUAL_SET_HASH in components,
    }
    return {
        "status": "PASS",
        "metrics_identity": metrics_identity,
        "ranking": balanced,
        "components": component_rows,
        "exploration_order": exploration_order_rows,
        "selected": selected,
        "seal": envelope("metachorus.v53.m7f-h3.r1-pool-seal.v1", "SEALED_AFTER_R1_BEFORE_R2", seal_basis),
    }


def derive_r3_shortlist(
    metric_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact sealed 8-set R3 shortlist using frozen Pareto/rank helpers."""

    candidates = {str(row["candidate_set_hash"]): dict(row) for row in candidate_rows}
    rows = [dict(row) for row in metric_rows]
    if len(candidates) != R1_POOL_SIZE or MANUAL_SET_HASH not in candidates or len(rows) != R1_POOL_SIZE * 3:
        raise ValueError("H3 R3 shortlist requires the sealed 24-by-3 matrix")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("candidate_set_hash"))
        if (
            digest not in candidates
            or str(row.get("dataset")) != DATASET
            or int(row.get("endpoint_update", -1)) != 2000
            or str(row.get("metric_role")) != "GLOBAL_SEARCH_POOL_ONLY"
            or bool(row.get("prelim_locked_report_access", False))
        ):
            raise ValueError("H3 R3 metric identity or label role is invalid")
        grouped[digest].append(row)
    if set(grouped) != set(candidates):
        raise ValueError("H3 R3 metrics/candidate pool mismatch")
    objectives = []
    for digest, cohort in grouped.items():
        if len(cohort) != 3 or {int(row["model_seed"]) for row in cohort} != set(SEARCH_MODEL_SEEDS):
            raise ValueError("H3 R3 candidate lacks the exact three search seeds")
        objectives.append({**candidates[digest], **aggregate_objectives(cohort)})
    support_count: defaultdict[str, int] = defaultdict(int)
    per_seed_balanced: defaultdict[str, list[float]] = defaultdict(list)
    for seed in SEARCH_MODEL_SEEDS:
        cohort = [row for row in rows if int(row["model_seed"]) == seed]
        for row in first_pareto_front(cohort, ("F_cls", "F_clu")):
            support_count[str(row["candidate_set_hash"])] += 1
        _order, cls_ranks, clu_ranks = _balanced_order(cohort, "F_cls", "F_clu")
        for row in cohort:
            digest = str(row["candidate_set_hash"])
            per_seed_balanced[digest].append(max(cls_ranks[digest], clu_ranks[digest]))
    robust_front = {str(row["candidate_set_hash"]) for row in first_pareto_front(objectives, OBJECTIVES)}
    balanced, q_cls_ranks, q_clu_ranks = _balanced_order(objectives, "Q_cls", "Q_clu")
    balanced_rank = {str(row["candidate_set_hash"]): index + 1 for index, row in enumerate(balanced)}
    components: defaultdict[str, set[str]] = defaultdict(set)
    for digest, count in support_count.items():
        if count >= 2:
            components[digest].add("A_PER_SEED_PARETO_SUPPORT_AT_LEAST_TWO")
    for digest in robust_front:
        components[digest].add("B_FOUR_OBJECTIVE_FIRST_PARETO")
    for row in balanced[:6]:
        components[str(row["candidate_set_hash"])].add("C_AGGREGATED_BALANCED_TOP6")
    components[MANUAL_SET_HASH].add("D_MANUAL")
    def priority(digest: str) -> tuple[Any, ...]:
        return (
            0 if digest == MANUAL_SET_HASH else 1,
            -len(components[digest]),
            -support_count[digest],
            balanced_rank[digest],
            max(per_seed_balanced[digest]),
            digest,
        )
    chosen = sorted(components, key=priority)[:R3_SHORTLIST_SIZE]
    if MANUAL_SET_HASH not in chosen:
        raise RuntimeError("H3 R3 shortlist truncation removed Manual")
    for row in balanced:
        if len(chosen) >= R3_SHORTLIST_SIZE:
            break
        digest = str(row["candidate_set_hash"])
        if digest not in chosen:
            components[digest].add("BALANCED_RANK_TOP_UP")
            chosen.append(digest)
    if len(chosen) != R3_SHORTLIST_SIZE or len(set(chosen)) != R3_SHORTLIST_SIZE:
        raise RuntimeError("H3 R3 shortlist is not exactly eight unique sets")
    objective_rows = []
    for row in objectives:
        digest = str(row["candidate_set_hash"])
        objective_rows.append(
            {
                **row,
                "pareto_support_count": support_count[digest],
                "aggregate_balanced_rank": balanced_rank[digest],
                "worst_seed_balanced_rank": max(per_seed_balanced[digest]),
                "normalized_rank_Q_cls": q_cls_ranks[digest],
                "normalized_rank_Q_clu": q_clu_ranks[digest],
            }
        )
    shortlist = [
        {
            **candidates[digest],
            "components": sorted(components[digest]),
            "component_count": len(components[digest]),
            "pareto_support_count": support_count[digest],
            "aggregate_balanced_rank": balanced_rank[digest],
            "worst_seed_balanced_rank": max(per_seed_balanced[digest]),
        }
        for digest in chosen
    ]
    shortlist.sort(key=lambda row: priority(str(row["candidate_set_hash"])))
    seal_basis = {
        "dataset": DATASET,
        "status": "SEALED_AFTER_R3_BEFORE_R4",
        "selected_candidate_set_count": len(shortlist),
        "selected_candidate_hashes": sorted(chosen),
        "shortlist_identity": canonical_json_hash(shortlist),
        "objectives_identity": canonical_json_hash(objective_rows),
        "metric_rows_identity": canonical_json_hash(sorted(rows, key=lambda row: (str(row["candidate_set_hash"]), int(row["model_seed"])))),
        "metric_role": "GLOBAL_SEARCH_POOL_ONLY",
        "prelim_locked_report_access": False,
        "manual_included": MANUAL_SET_HASH in chosen,
    }
    return {
        "status": "PASS",
        "objectives": objective_rows,
        "components": [
            {
                "candidate_set_hash": digest,
                "candidate_set_id": candidates[digest]["candidate_set_id"],
                "components": sorted(components[digest]),
                "selected": digest in chosen,
            }
            for digest in sorted(components)
        ],
        "shortlist": shortlist,
        "seal": envelope("metachorus.v53.m7f-h3.r3-shortlist-seal.v1", "SEALED_AFTER_R3_BEFORE_R4", seal_basis),
    }


def build_protocol_scope_audit(repo_root: str | Path, result_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    result = Path(result_root).resolve(strict=True)
    source = repo / FROZEN_PROTOCOL_RELATIVE_PATH
    if sha256_file(source) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("BLOCKED_PROTOCOL_SOURCE_IDENTITY_MISMATCH")
    module_path = Path(inspect.getsourcefile(frozen_protocol) or "").resolve(strict=True)
    if module_path != source:
        raise RuntimeError("frozen protocol import does not resolve to the governed source")
    functions = {}
    for name in (
        "manual_safety_gate", "first_pareto_front", "normalized_midranks",
        "select_unique_robust_set", "close_search_selection", "preliminary_decision",
    ):
        function = getattr(frozen_protocol, name)
        functions[name] = {
            "module": function.__module__,
            "source_sha256": canonical_json_hash(inspect.getsource(function)),
        }
    basis = {
        "protocol_relative_path": FROZEN_PROTOCOL_RELATIVE_PATH,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "functions": functions,
        "manual_safety_gate": "EXACT_THREE_PAIRED_SEEDS_MEAN_NOT_BELOW_NEGATIVE_PAIRED_SE_AND_EACH_DROP_GT_NEGATIVE_0.02",
        "four_objective_pareto": "FIRST_PARETO_FRONT_OVER_Q_CLS_Q_CLU_W_CLS_W_CLU",
        "robust_rank_normalization_scope": "SAFE_ROWS_FIRST_FOUR_OBJECTIVE_PARETO_FRONT_ONLY",
        "normalized_midrank_formula": "(MIDRANK_MINUS_1)/(COHORT_SIZE_MINUS_1); SINGLETON_ZERO",
        "final_tie_break": "R_ROBUST,MEAN_R_MEDIAN_R_WORST,NEG_MIN_W,NEG_STABILITY,CANDIDATE_HASH",
        "h3_r1_rank_scope": "ALL_220_ONE_SEED_R1_ROWS",
        "h3_r3_rank_scope": "SEALED_24_SET_THREE_SEED_REFINEMENT_ROWS",
        "direct_frozen_function_calls_required": True,
        "algorithm_retuning": False,
    }
    audit = envelope("metachorus.v53.m7f-h3.protocol-scope-audit.v1", "PASS", basis)
    write_json_once(result / "M7F_H3_PROTOCOL_SCOPE_AUDIT.json", audit)
    write_text_once(
        result / "M7F_H3_PROTOCOL_SCOPE_AUDIT.md",
        "# M7F-H3 Frozen Protocol Scope Audit\n\n"
        f"- Protocol SHA-256: `{FROZEN_PROTOCOL_SHA256}`\n"
        "- Manual safety, Pareto, normalized midrank, robust selector and preliminary decision are imported directly.\n"
        "- Robust selector rank normalization is over the safe first four-objective Pareto front.\n"
        "- H3 R1 balanced ranks use all 220 frozen one-seed rows; H3 R3 balanced ranks use the sealed 24-set refinement cohort.\n"
        "- No algorithm retuning or copied selector implementation is permitted.\n",
    )
    return audit


__all__ = [
    "derive_r1_pool", "derive_r3_shortlist", "build_protocol_scope_audit",
    "manual_safety_gate", "first_pareto_front", "normalized_midranks",
    "select_unique_robust_set", "close_search_selection", "preliminary_decision",
]
