"""Pure deterministic H6 pool, reliability, fidelity, and Pareto primitives."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import re
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence

from scipy.stats import kendalltau, rankdata, spearmanr

from .constants import (
    ALL_DEVELOPMENT_MODEL_SEEDS,
    ASSIGNMENT_STABILITY_MEDIAN_MIN,
    CLASSIFICATION_LOO_RHO_MEDIAN_MIN,
    CLASSIFICATION_TOP_QUARTILE_OVERLAP_MEDIAN_MIN,
    CLASSIFICATION_TRUTH_RHO_MIN,
    CLUSTERING_TRUTH_RHO_MIN,
    COMPONENT_TOP_K,
    CONTROLS,
    EXPECTED_PARENT_R3_ROWS,
    EXPECTED_POOL_COUNTS,
    EXPECTED_POOL_TOTAL,
    FREEBASE_CANARY_HASH,
    JOINT_TRUTH_RHO_MIN,
    PAIRWISE_MINIMUM_RHO,
    PAIRWISE_PASS_COUNT_MIN,
    PAIRWISE_RHO_THRESHOLD,
    R4_PER_SEED_DROP_LIMIT,
    R4_PROMOTION_COUNTS,
    SEARCH_MODEL_SEEDS,
    THIRD_DEVELOPMENT_MODEL_SEED,
    TRUTH_KENDALL_MIN_EXCLUSIVE,
)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value!r}")
    return result


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation requires matched vectors with at least two values")
    value = float(spearmanr(left, right).statistic)
    if not math.isfinite(value):
        return -1.0
    return math.copysign(1.0, value) if abs(abs(value) - 1.0) < 1e-12 else value


def _safe_kendall(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation requires matched vectors with at least two values")
    value = float(kendalltau(left, right).statistic)
    if not math.isfinite(value):
        return -1.0
    return math.copysign(1.0, value) if abs(abs(value) - 1.0) < 1e-12 else value


def _source_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(row["dataset"]),
        "candidate_set_id": str(row["candidate_set_id"]),
        "candidate_set_hash": str(row["candidate_set_hash"]),
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
    }


def _top_component(
    rows: Sequence[Mapping[str, Any]],
    *,
    component: str,
    score_field: str,
    model_seed: int | None,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (-_finite(row[score_field]), str(row["candidate_set_hash"])),
    )[:COMPONENT_TOP_K]
    return [
        {
            **_source_identity(row),
            "component": component,
            "component_rank": index,
            "component_score": _finite(row[score_field]),
            "model_seed": "" if model_seed is None else int(model_seed),
            "source_field": score_field,
            "truth_fields_read": False,
            "h5_locked_report_access": False,
            "s_eval_access": False,
        }
        for index, row in enumerate(ordered, start=1)
    ]


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        _finite(left["Q_cls"]) >= _finite(right["Q_cls"])
        and _finite(left["Q_clu"]) >= _finite(right["Q_clu"])
        and (
            _finite(left["Q_cls"]) > _finite(right["Q_cls"])
            or _finite(left["Q_clu"]) > _finite(right["Q_clu"])
        )
    )


def derive_uncertainty_pool(
    search_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ""

    if len(search_rows) != EXPECTED_PARENT_R3_ROWS:
        raise ValueError(f"H5 R3 search metrics must contain {EXPECTED_PARENT_R3_ROWS} rows")
    for row in search_rows:
        forbidden = [
            key
            for key in row
            if any(token in key.casefold() for token in ("truth", "locked_report", "s_eval"))
        ]
        if forbidden:
            raise ValueError(f"pool source contains forbidden fields: {sorted(forbidden)}")
        if int(row["model_seed"]) not in SEARCH_MODEL_SEEDS:
            raise ValueError("pool source includes a non-search model seed")

    by_dataset_seed: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    by_dataset_hash: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen_rows: set[tuple[str, str, int]] = set()
    candidate_identities: dict[tuple[str, str], tuple[str, int]] = {}
    ordinals: dict[tuple[str, int], str] = {}
    for row in search_rows:
        dataset = str(row["dataset"])
        if dataset not in EXPECTED_POOL_COUNTS:
            raise ValueError(f"unexpected H5 R3 dataset: {dataset}")
        seed = int(row["model_seed"])
        candidate_hash = str(row["candidate_set_hash"])
        row_key = (dataset, candidate_hash, seed)
        if row_key in seen_rows:
            raise ValueError(f"duplicate H5 R3 search identity: {row_key}")
        seen_rows.add(row_key)
        identity_key = (dataset, candidate_hash)
        identity_value = (str(row["candidate_set_id"]), int(row["candidate_set_ordinal"]))
        if identity_key in candidate_identities and candidate_identities[identity_key] != identity_value:
            raise ValueError(f"inconsistent H5 R3 candidate identity: {identity_key}")
        candidate_identities[identity_key] = identity_value
        ordinal_key = (dataset, int(row["candidate_set_ordinal"]))
        if ordinal_key in ordinals and ordinals[ordinal_key] != candidate_hash:
            raise ValueError(f"duplicate H5 R3 candidate ordinal: {ordinal_key}")
        ordinals[ordinal_key] = candidate_hash
        by_dataset_seed[(dataset, seed)].append(row)
        by_dataset_hash[(dataset, candidate_hash)].append(row)
    if len(seen_rows) != EXPECTED_PARENT_R3_ROWS:
        raise ValueError("H5 R3 search identities are not unique")

    components: list[dict[str, Any]] = []
    acm_by_hash = {
        str(row["candidate_set_hash"]): row
        for row in search_rows
        if str(row["dataset"]) == "acm"
    }
    if len(acm_by_hash) != EXPECTED_POOL_COUNTS["acm"]:
        raise ValueError("H6 must retain all ten ACM identities")
    for index, row in enumerate(
        sorted(acm_by_hash.values(), key=lambda value: int(value["candidate_set_ordinal"])),
        start=1,
    ):
        components.append(
            {
                **_source_identity(row),
                "component": "ACM_ALL10",
                "component_rank": index,
                "component_score": "",
                "model_seed": "",
                "source_field": "identity_only",
                "truth_fields_read": False,
                "h5_locked_report_access": False,
                "s_eval_access": False,
            }
        )

    for seed in SEARCH_MODEL_SEEDS:
        cohort = by_dataset_seed[("freebase", seed)]
        if len(cohort) != 122:
            raise ValueError(f"Freebase H5 R3 seed {seed} must contain 122 candidates")
        components.extend(
            _top_component(
                cohort,
                component=f"SEED_{seed}_CLS_TOP20",
                score_field="mean_F_cls",
                model_seed=seed,
            )
        )
        components.extend(
            _top_component(
                cohort,
                component=f"SEED_{seed}_CLU_TOP20",
                score_field="mean_F_clu",
                model_seed=seed,
            )
        )

    aggregate: list[dict[str, Any]] = []
    for (dataset, candidate_hash), rows in sorted(by_dataset_hash.items()):
        if dataset != "freebase":
            continue
        if sorted(int(row["model_seed"]) for row in rows) != sorted(SEARCH_MODEL_SEEDS):
            raise ValueError(f"Freebase candidate lacks two H5 seeds: {candidate_hash}")
        first = rows[0]
        cls = [_finite(row["mean_F_cls"]) for row in rows]
        clu = [_finite(row["mean_F_clu"]) for row in rows]
        aggregate.append(
            {
                **_source_identity(first),
                "Q_cls": median(cls),
                "Q_clu": median(clu),
                "W_cls": min(cls),
                "W_clu": min(clu),
            }
        )
    if len(aggregate) != 122:
        raise ValueError("Freebase aggregate must contain 122 candidates")
    for component, field in (
        ("Q_CLS_TOP20", "Q_cls"),
        ("Q_CLU_TOP20", "Q_clu"),
        ("W_CLS_TOP20", "W_cls"),
        ("W_CLU_TOP20", "W_clu"),
    ):
        components.extend(
            _top_component(
                aggregate,
                component=component,
                score_field=field,
                model_seed=None,
            )
        )
    first_front = [
        row
        for row in aggregate
        if not any(_dominates(other, row) for other in aggregate if other is not row)
    ]
    first_front.sort(key=lambda row: str(row["candidate_set_hash"]))
    for index, row in enumerate(first_front, start=1):
        components.append(
            {
                **_source_identity(row),
                "component": "FIRST_PARETO_FRONT_Q_CLS_Q_CLU",
                "component_rank": index,
                "component_score": "",
                "model_seed": "",
                "source_field": "Q_cls,Q_clu",
                "truth_fields_read": False,
                "h5_locked_report_access": False,
                "s_eval_access": False,
            }
        )

    by_hash = {
        str(row["candidate_set_hash"]): row
        for row in search_rows
        if str(row["dataset"]) == "freebase"
    }
    for control_name, (_, candidate_hash) in CONTROLS["freebase"].items():
        if candidate_hash not in by_hash:
            raise ValueError(f"missing Freebase control: {control_name}")
        components.append(
            {
                **_source_identity(by_hash[candidate_hash]),
                "component": f"CONTROL_{control_name.upper()}",
                "component_rank": 1,
                "component_score": "",
                "model_seed": "",
                "source_field": "frozen_control_identity",
                "truth_fields_read": False,
                "h5_locked_report_access": False,
                "s_eval_access": False,
            }
        )

    roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    for row in components:
        key = (str(row["dataset"]), str(row["candidate_set_hash"]))
        roles[key].add(str(row["component"]))
        identities[key] = _source_identity(row)
    pool: list[dict[str, Any]] = []
    for key in sorted(roles, key=lambda value: (value[0], int(identities[value]["candidate_set_ordinal"]))):
        dataset, candidate_hash = key
        control_names = [
            name
            for name, (_, control_hash) in CONTROLS[dataset].items()
            if control_hash == candidate_hash
        ]
        pool.append(
            {
                **identities[key],
                "roles_json": json.dumps(sorted(roles[key]), ensure_ascii=False),
                "is_manual": "manual" in control_names,
                "is_current_automatic": "current_automatic" in control_names,
                "truth_fields_read": False,
                "h5_locked_report_access": False,
                "s_eval_access": False,
            }
        )
    observed_counts = {
        dataset: sum(str(row["dataset"]) == dataset for row in pool)
        for dataset in ("acm", "freebase")
    }
    if observed_counts != EXPECTED_POOL_COUNTS or len(pool) != EXPECTED_POOL_TOTAL:
        raise ValueError(
            f"H6 uncertainty pool is not exactly 10/75/85: {observed_counts}/{len(pool)}"
        )
    return components, pool


def build_canary_identities(
    pool_rows: Sequence[Mapping[str, Any]],
    pre_hotfix_failure_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_hash = {str(row["candidate_set_hash"]): row for row in pool_rows}
    if len(by_hash) != len(pool_rows) or len(pool_rows) != EXPECTED_POOL_TOTAL:
        raise ValueError("H6 canary source pool contains duplicate or missing candidate identities")
    by_freebase_ordinal = {
        int(row["candidate_set_ordinal"]): str(row["candidate_set_hash"])
        for row in pool_rows
        if str(row["dataset"]) == "freebase"
    }
    failed_freebase: set[str] = set()
    seen_failure_ids: set[str] = set()
    for row in pre_hotfix_failure_records:
        run_id = str(row.get("physical_run_id", ""))
        if not run_id or run_id in seen_failure_ids:
            raise ValueError("H5 pre-hotfix failure manifest contains a missing or duplicate run identity")
        seen_failure_ids.add(run_id)
        match = re.fullmatch(r"M4A-H5-FREEBASE-C(\d{3})-S(?:00|01)", run_id)
        if match is None:
            raise ValueError(f"invalid H5 pre-hotfix failure identity: {run_id}")
        if row.get("candidate_set_hash"):
            if str(row.get("dataset", "")).casefold() != "freebase":
                raise ValueError(f"invalid H5 pre-hotfix failure dataset: {run_id}")
            candidate_hash = str(row["candidate_set_hash"])
            mapped = by_freebase_ordinal.get(int(match.group(1)))
            if mapped is not None and candidate_hash != mapped:
                raise ValueError(f"H5 pre-hotfix failure hash/ordinal mismatch: {run_id}")
            failed_freebase.add(candidate_hash)
            continue
        if int(match.group(1)) in by_freebase_ordinal:
            failed_freebase.add(by_freebase_ordinal[int(match.group(1))])
    intersection = sorted(
        candidate_hash
        for candidate_hash in failed_freebase
        if candidate_hash in by_hash and str(by_hash[candidate_hash]["dataset"]) == "freebase"
    )
    if not intersection:
        raise RuntimeError("BLOCKED_CANARY_IDENTITY_NOT_FOUND")
    freebase_hash = intersection[0]
    if freebase_hash != FREEBASE_CANARY_HASH:
        raise RuntimeError("frozen parent/canary identity drift")
    acm_hash = CONTROLS["acm"]["manual"][1]
    selected = [by_hash[acm_hash], by_hash[freebase_hash]]
    output: list[dict[str, Any]] = []
    for gpu, row in enumerate(selected):
        dataset = str(row["dataset"])
        ordinal = int(row["candidate_set_ordinal"])
        output.append(
            {
                **_source_identity(row),
                "stage": "CANARY",
                "physical_run_id": (
                    f"M4A-H6-CANARY-{dataset.upper()}-C{ordinal:03d}-S02"
                ),
                "model_seed": THIRD_DEVELOPMENT_MODEL_SEED,
                "continuation_from_update": 0,
                "endpoint_update": 20,
                "continuation_mode": "FROM_ZERO_ENGINEERING_CANARY",
                "execution_action": "RUN_ENGINEERING_CANARY",
                "run_role": "M4A_H6_ENGINEERING_CANARY",
                "eligible_for_scientific_result": False,
                "eligible_for_selector_development": False,
                "labels_mounted": False,
                "downstream_evaluations": 0,
                "reusable_for_science": False,
                "gpu_preference": gpu,
                "attempt_index": 0,
                "status": "PENDING_GPU",
            }
        )
    return output


def three_seed_pairwise_reliability(
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    task_field: str,
) -> dict[str, Any]:
    cohort = [row for row in per_seed_rows if str(row["dataset"]) == dataset]
    by_key = {
        (str(row["candidate_set_hash"]), int(row["model_seed"])): _finite(row[task_field])
        for row in cohort
    }
    if len(by_key) != len(cohort):
        raise ValueError(f"duplicate three-seed rows for {dataset}/{task_field}")
    hashes = sorted({candidate for candidate, _ in by_key})
    if len(hashes) != EXPECTED_POOL_COUNTS[dataset]:
        raise ValueError(f"unexpected {dataset} H6 audit cohort size")
    if any((candidate, seed) not in by_key for candidate in hashes for seed in ALL_DEVELOPMENT_MODEL_SEEDS):
        raise ValueError(f"incomplete three-seed cohort for {dataset}/{task_field}")
    pair_rows: list[dict[str, Any]] = []
    for index, left in enumerate(ALL_DEVELOPMENT_MODEL_SEEDS):
        for right in ALL_DEVELOPMENT_MODEL_SEEDS[index + 1 :]:
            rho = _safe_spearman(
                [by_key[(candidate, left)] for candidate in hashes],
                [by_key[(candidate, right)] for candidate in hashes],
            )
            pair_rows.append(
                {
                    "left_model_seed": left,
                    "right_model_seed": right,
                    "spearman_rho": rho,
                    "passes_primary_threshold": rho >= PAIRWISE_RHO_THRESHOLD,
                    "passes_minimum_threshold": rho >= PAIRWISE_MINIMUM_RHO,
                }
            )
    values = [float(row["spearman_rho"]) for row in pair_rows]
    pass_count = sum(bool(row["passes_primary_threshold"]) for row in pair_rows)
    reliable = (
        median(values) >= PAIRWISE_RHO_THRESHOLD
        and pass_count >= PAIRWISE_PASS_COUNT_MIN
        and min(values) >= PAIRWISE_MINIMUM_RHO
    )
    return {
        "dataset": dataset,
        "task_field": task_field,
        "candidate_count": len(hashes),
        "model_seed_count": 3,
        "pairwise_json": json.dumps(pair_rows, ensure_ascii=False, sort_keys=True),
        "median_pairwise_rho": median(values),
        "pairwise_primary_pass_count": pass_count,
        "minimum_pairwise_rho": min(values),
        "median_threshold": PAIRWISE_RHO_THRESHOLD,
        "required_primary_pass_count": PAIRWISE_PASS_COUNT_MIN,
        "minimum_threshold": PAIRWISE_MINIMUM_RHO,
        "status": "RELIABLE" if reliable else "UNRELIABLE",
    }


def classification_seed_stability(
    repeat_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in repeat_rows:
        if str(row["dataset"]) != dataset:
            continue
        key = (str(row["candidate_set_hash"]), int(row["model_seed"]))
        split = int(row["classification_split_index"])
        if split in grouped[key]:
            raise ValueError(f"duplicate classification repeat: {key}/{split}")
        grouped[key][split] = _finite(row["F_cls"])
    hashes = sorted({candidate for candidate, _ in grouped})
    if len(hashes) != EXPECTED_POOL_COUNTS[dataset]:
        raise ValueError(f"unexpected {dataset} classification cohort")
    output: list[dict[str, Any]] = []
    top_k = int(math.ceil(0.25 * len(hashes)))
    for seed in ALL_DEVELOPMENT_MODEL_SEEDS:
        for candidate in hashes:
            if sorted(grouped.get((candidate, seed), {})) != list(range(5)):
                raise ValueError(f"incomplete classification repeats: {candidate}/{seed}")
        full = {
            candidate: mean(grouped[(candidate, seed)].values()) for candidate in hashes
        }
        full_order = sorted(hashes, key=lambda value: (-full[value], value))
        full_top = set(full_order[:top_k])
        rhos: list[float] = []
        overlaps: list[float] = []
        for omitted in range(5):
            loo = {
                candidate: mean(
                    value
                    for split, value in grouped[(candidate, seed)].items()
                    if split != omitted
                )
                for candidate in hashes
            }
            rhos.append(_safe_spearman([full[value] for value in hashes], [loo[value] for value in hashes]))
            loo_order = sorted(hashes, key=lambda value: (-loo[value], value))
            overlaps.append(len(full_top & set(loo_order[:top_k])) / top_k)
        rho_value = median(rhos)
        overlap_value = median(overlaps)
        reliable = (
            rho_value >= CLASSIFICATION_LOO_RHO_MEDIAN_MIN
            and overlap_value >= CLASSIFICATION_TOP_QUARTILE_OVERLAP_MEDIAN_MIN
        )
        output.append(
            {
                "dataset": dataset,
                "model_seed": seed,
                "candidate_count": len(hashes),
                "classification_split_count": 5,
                "LOO_rho_median": rho_value,
                "LOO_top_quartile_overlap_median": overlap_value,
                "top_quartile_k": top_k,
                "loo_rho_threshold": CLASSIFICATION_LOO_RHO_MEDIAN_MIN,
                "top_overlap_threshold": CLASSIFICATION_TOP_QUARTILE_OVERLAP_MEDIAN_MIN,
                "status": "RELIABLE" if reliable else "UNRELIABLE",
            }
        )
    return output


def clustering_seed_stability(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    hashes_by_seed: dict[int, set[str]] = defaultdict(set)
    seen: set[tuple[str, int]] = set()
    for row in summary_rows:
        if str(row["dataset"]) != dataset:
            continue
        seed = int(row["model_seed"])
        key = (str(row["candidate_set_hash"]), seed)
        if key in seen:
            raise ValueError(f"duplicate clustering summary: {dataset}/{key}")
        seen.add(key)
        by_seed[seed].append(_finite(row["assignment_stability_mean"]))
        hashes_by_seed[seed].add(str(row["candidate_set_hash"]))
        if int(row.get("repeat_count", 20)) != 20:
            raise ValueError("H6 clustering must keep exactly twenty KMeans seeds")
    output: list[dict[str, Any]] = []
    for seed in ALL_DEVELOPMENT_MODEL_SEEDS:
        if len(hashes_by_seed[seed]) != EXPECTED_POOL_COUNTS[dataset]:
            raise ValueError(f"incomplete clustering stability cohort: {dataset}/{seed}")
        value = median(by_seed[seed])
        output.append(
            {
                "dataset": dataset,
                "model_seed": seed,
                "candidate_count": len(hashes_by_seed[seed]),
                "kmeans_seed_count": 20,
                "assignment_stability_mean_median": value,
                "assignment_stability_threshold": ASSIGNMENT_STABILITY_MEDIAN_MIN,
                "status": "RELIABLE" if value >= ASSIGNMENT_STABILITY_MEDIAN_MIN else "UNRELIABLE",
            }
        )
    return output


def aggregate_three_seed_objectives(
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    cohort = [row for row in per_seed_rows if str(row["dataset"]) == dataset]
    by_key = {
        (str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in cohort
    }
    if len(by_key) != len(cohort):
        raise ValueError(f"duplicate three-seed objective identity for {dataset}")
    hashes = sorted({candidate for candidate, _ in by_key})
    if len(hashes) != EXPECTED_POOL_COUNTS[dataset]:
        raise ValueError(f"unexpected {dataset} objective cohort")
    output: list[dict[str, Any]] = []
    for candidate in hashes:
        rows = [by_key[(candidate, seed)] for seed in ALL_DEVELOPMENT_MODEL_SEEDS]
        cls = [_finite(row["mean_F_cls"]) for row in rows]
        clu = [_finite(row["mean_F_clu"]) for row in rows]
        first = rows[0]
        output.append(
            {
                **_source_identity(first),
                "seed_count": 3,
                "Q_cls": median(cls),
                "Q_clu": median(clu),
                "W_cls": min(cls),
                "W_clu": min(clu),
                "assignment_stability_mean": median(
                    _finite(row["assignment_stability_mean"]) for row in rows
                ),
                "per_seed_json": json.dumps(
                    [
                        {
                            "model_seed": seed,
                            "F_cls": cls[index],
                            "F_clu": clu[index],
                        }
                        for index, seed in enumerate(ALL_DEVELOPMENT_MODEL_SEEDS)
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return output


def joint_normalized_midrank(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ""

    output = [dict(row) for row in rows]
    if len(output) < 2:
        raise ValueError("joint ranks require at least two candidates")
    cls_ranks = rankdata([-_finite(row["Q_cls"]) for row in output], method="average")
    clu_ranks = rankdata([-_finite(row["Q_clu"]) for row in output], method="average")
    denominator = max(len(output) - 1, 1)
    for row, cls_rank, clu_rank in zip(output, cls_ranks, clu_ranks):
        row["classification_normalized_midrank"] = (float(cls_rank) - 1.0) / denominator
        row["clustering_normalized_midrank"] = (float(clu_rank) - 1.0) / denominator
        row["joint_normalized_midrank"] = mean(
            [row["classification_normalized_midrank"], row["clustering_normalized_midrank"]]
        )
        row["joint_order_score"] = -row["joint_normalized_midrank"]
    return output


def truth_fidelity(
    stage_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    stage_field: str,
    truth_field: str,
    task: str,
) -> dict[str, Any]:
    stage_cohort = [row for row in stage_rows if str(row["dataset"]) == dataset]
    truth_cohort = [row for row in truth_rows if str(row["dataset"]) == dataset]
    stage = {
        str(row["candidate_set_hash"]): _finite(row[stage_field])
        for row in stage_cohort
    }
    truth = {
        str(row["candidate_set_hash"]): _finite(row[truth_field])
        for row in truth_cohort
    }
    if len(stage) != len(stage_cohort) or len(truth) != len(truth_cohort):
        raise ValueError(f"duplicate {dataset}/{task} truth fidelity identity")
    hashes = sorted(set(stage) & set(truth))
    if len(hashes) != EXPECTED_POOL_COUNTS[dataset] or set(stage) != set(hashes):
        raise ValueError(f"incomplete {dataset}/{task} truth fidelity cohort")
    left = [stage[value] for value in hashes]
    right = [truth[value] for value in hashes]
    return {
        "dataset": dataset,
        "task": task,
        "candidate_count": len(hashes),
        "spearman_to_3000_truth": _safe_spearman(left, right),
        "kendall_to_3000_truth": _safe_kendall(left, right),
        "stage_field": stage_field,
        "truth_field": truth_field,
        "truth_accessed_after_pool_seal": True,
    }


def three_seed_gate_decision(
    *,
    pairwise_rows: Sequence[Mapping[str, Any]],
    classification_stability_rows: Sequence[Mapping[str, Any]],
    clustering_stability_rows: Sequence[Mapping[str, Any]],
    fidelity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    expected_pairwise = {
        (dataset, task_field)
        for dataset in ("acm", "freebase")
        for task_field in ("mean_F_cls", "mean_F_clu")
    }
    expected_stability = {
        (dataset, seed)
        for dataset in ("acm", "freebase")
        for seed in ALL_DEVELOPMENT_MODEL_SEEDS
    }
    expected_fidelity = {
        (dataset, task)
        for dataset in ("acm", "freebase")
        for task in ("classification", "clustering", "joint")
    }
    pairwise_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    classification_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    clustering_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    fidelity_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        pairwise_groups[(str(row.get("dataset")), str(row.get("task_field")))].append(row)
    for row in classification_stability_rows:
        classification_groups[(str(row.get("dataset")), int(row.get("model_seed", -1)))].append(row)
    for row in clustering_stability_rows:
        clustering_groups[(str(row.get("dataset")), int(row.get("model_seed", -1)))].append(row)
    for row in fidelity_rows:
        fidelity_groups[(str(row.get("dataset")), str(row.get("task")))].append(row)
    for label, observed, expected in (
        ("pairwise", pairwise_groups, expected_pairwise),
        ("classification_stability", classification_groups, expected_stability),
        ("clustering_stability", clustering_groups, expected_stability),
        ("fidelity", fidelity_groups, expected_fidelity),
    ):
        for key in sorted(set(observed) | expected):
            count = len(observed.get(key, []))
            if key not in expected or count != 1:
                reasons.append(f"{label}:{key}:expected_once_observed_{count}")
    for dataset in ("acm", "freebase"):
        for task_field in ("mean_F_cls", "mean_F_clu"):
            matches = pairwise_groups.get((dataset, task_field), [])
            if len(matches) != 1 or matches[0].get("status") != "RELIABLE":
                reasons.append(f"{dataset}:{task_field}:pairwise_reliability")
        for seed in ALL_DEVELOPMENT_MODEL_SEEDS:
            cls_matches = classification_groups.get((dataset, seed), [])
            clu_matches = clustering_groups.get((dataset, seed), [])
            if len(cls_matches) != 1 or cls_matches[0].get("status") != "RELIABLE":
                reasons.append(f"{dataset}:seed_{seed}:classification_stability")
            if len(clu_matches) != 1 or clu_matches[0].get("status") != "RELIABLE":
                reasons.append(f"{dataset}:seed_{seed}:clustering_stability")
        for task, rho_threshold, require_kendall in (
            ("classification", CLASSIFICATION_TRUTH_RHO_MIN, True),
            ("clustering", CLUSTERING_TRUTH_RHO_MIN, True),
            ("joint", JOINT_TRUTH_RHO_MIN, True),
        ):
            matches = fidelity_groups.get((dataset, task), [])
            if len(matches) != 1:
                reasons.append(f"{dataset}:{task}:missing_fidelity")
                continue
            row = matches[0]
            if _finite(row["spearman_to_3000_truth"]) < rho_threshold:
                reasons.append(f"{dataset}:{task}:truth_spearman")
            if require_kendall and _finite(row["kendall_to_3000_truth"]) <= TRUTH_KENDALL_MIN_EXCLUSIVE:
                reasons.append(f"{dataset}:{task}:truth_kendall")
    return {
        "status": (
            "TC_LPMSO_H6_THREE_SEED_R3_PASS"
            if not reasons
            else "TC_LPMSO_H6_THREE_SEED_R3_FAIL"
        ),
        "r4_started": False,
        "reasons": reasons,
    }


def pareto_order(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pending = [dict(row) for row in rows]
    output: list[dict[str, Any]] = []
    front = 1
    while pending:
        members = [
            row
            for row in pending
            if not any(_dominates(other, row) for other in pending if other is not row)
        ]
        member_hashes = {str(row["candidate_set_hash"]) for row in members}
        for row in members:
            row["pareto_front"] = front
            output.append(row)
        pending = [row for row in pending if str(row["candidate_set_hash"]) not in member_hashes]
        front += 1
    denominator = max(len(output) - 1, 1)
    cls_ranks = rankdata([-_finite(row["Q_cls"]) for row in output], method="average")
    clu_ranks = rankdata([-_finite(row["Q_clu"]) for row in output], method="average")
    per_seed = {
        str(row["candidate_set_hash"]): json.loads(str(row["per_seed_json"]))
        for row in output
    }
    worst: dict[tuple[str, str], float] = {}
    for task in ("cls", "clu"):
        field = f"F_{task}"
        for seed in ALL_DEVELOPMENT_MODEL_SEEDS:
            cohort = [
                (candidate, item)
                for candidate, values in per_seed.items()
                for item in values
                if int(item["model_seed"]) == seed
            ]
            ranks = rankdata([-_finite(item[field]) for _, item in cohort], method="average")
            for (candidate, _), rank in zip(cohort, ranks):
                worst[(candidate, task)] = max(float(rank), worst.get((candidate, task), 0.0))
    for row, cls_rank, clu_rank in zip(output, cls_ranks, clu_ranks):
        candidate = str(row["candidate_set_hash"])
        row["normalized_rank_cls"] = (float(cls_rank) - 1.0) / denominator
        row["normalized_rank_clu"] = (float(clu_rank) - 1.0) / denominator
        row["minimax_normalized_rank"] = max(row["normalized_rank_cls"], row["normalized_rank_clu"])
        row["mean_normalized_rank"] = mean([row["normalized_rank_cls"], row["normalized_rank_clu"]])
        row["worst_seed_rank_cls"] = worst[(candidate, "cls")]
        row["worst_seed_rank_clu"] = worst[(candidate, "clu")]
        row["minimax_worst_seed_rank"] = max(row["worst_seed_rank_cls"], row["worst_seed_rank_clu"])
    output.sort(
        key=lambda row: (
            int(row["pareto_front"]),
            _finite(row["minimax_normalized_rank"]),
            _finite(row["minimax_worst_seed_rank"]),
            _finite(row["mean_normalized_rank"]),
            -_finite(row["assignment_stability_mean"]),
            str(row["candidate_set_hash"]),
        )
    )
    for index, row in enumerate(output, start=1):
        row["pareto_order"] = index
    return output


def r4_promotion(ordered: Sequence[Mapping[str, Any]], *, dataset: str) -> list[dict[str, Any]]:
    cohort = [dict(row) for row in ordered if str(row["dataset"]) == dataset]
    controls = {value[1] for value in CONTROLS[dataset].values()}
    selected = [row for row in cohort if str(row["candidate_set_hash"]) not in controls][
        : R4_PROMOTION_COUNTS[dataset]
    ]
    selected_hashes = {str(row["candidate_set_hash"]) for row in selected}
    output: list[dict[str, Any]] = []
    for row in cohort:
        candidate_hash = str(row["candidate_set_hash"])
        roles: list[str] = []
        if candidate_hash in selected_hashes:
            roles.append("R4_PROMOTED")
        for name, (_, control_hash) in CONTROLS[dataset].items():
            if candidate_hash == control_hash:
                roles.append(name.upper())
        if roles:
            output.append(
                {
                    **row,
                    "roles_json": json.dumps(sorted(roles), ensure_ascii=False),
                    "consumes_promotion_slot": candidate_hash in selected_hashes,
                    "controls_outside_slots": True,
                }
            )
    if sum(bool(row["consumes_promotion_slot"]) for row in output) != R4_PROMOTION_COUNTS[dataset]:
        raise RuntimeError(f"{dataset} R4 promotion count drift")
    return output


def manual_safety(
    candidate_per_seed: Mapping[int, Mapping[str, float]],
    manual_per_seed: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    if set(candidate_per_seed) != set(ALL_DEVELOPMENT_MODEL_SEEDS):
        raise ValueError("candidate safety metrics lack three development seeds")
    if set(manual_per_seed) != set(ALL_DEVELOPMENT_MODEL_SEEDS):
        raise ValueError("manual safety metrics lack three development seeds")
    output: dict[str, Any] = {"seed_count": 3}
    safe_all = True
    joint: list[float] = []
    for task in ("cls", "clu"):
        field = f"F_{task}"
        deltas = [
            _finite(candidate_per_seed[seed][field]) - _finite(manual_per_seed[seed][field])
            for seed in ALL_DEVELOPMENT_MODEL_SEEDS
        ]
        paired_se = stdev(deltas) / math.sqrt(3)
        average = mean(deltas)
        safe = average >= -paired_se and all(value > -R4_PER_SEED_DROP_LIMIT for value in deltas)
        output[f"delta_{task}_per_seed_json"] = json.dumps(deltas)
        output[f"mean_delta_{task}"] = average
        output[f"paired_se_delta_{task}"] = paired_se
        output[f"every_seed_above_minus_2pp_{task}"] = all(
            value > -R4_PER_SEED_DROP_LIMIT for value in deltas
        )
        output[f"safe_{task}"] = safe
        safe_all = safe_all and safe
        joint.append(average)
    output["mean_joint_delta"] = mean(joint)
    output["status"] = "SAFE" if safe_all else "UNSAFE"
    return output


def choose_single_r4_set(
    ordered: Sequence[Mapping[str, Any]],
    safety_by_hash: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
) -> dict[str, Any]:
    """Choose one complete safe path set with the frozen R4 tie-break chain."""

    cohort = [dict(row) for row in ordered if str(row["dataset"]) == dataset]
    safe = [
        row
        for row in cohort
        if safety_by_hash.get(str(row["candidate_set_hash"]), {}).get("status") == "SAFE"
    ]
    if not safe:
        return {
            "dataset": dataset,
            "status": "TC_LPMSO_H6_DEVELOPMENT_FAIL",
            "reason": "NO_SAFE_COMPLETE_SET",
            "complete_set_only": True,
            "taskwise_set_splicing": False,
        }
    first_front = min(int(row["pareto_front"]) for row in safe)
    safe = [row for row in safe if int(row["pareto_front"]) == first_front]
    safe.sort(
        key=lambda row: (
            _finite(row["minimax_normalized_rank"]),
            _finite(row["mean_normalized_rank"]),
            -min(_finite(row["W_cls"]), _finite(row["W_clu"])),
            -_finite(row["assignment_stability_mean"]),
            str(row["candidate_set_hash"]),
        )
    )
    selected = safe[0]
    selected_hash = str(selected["candidate_set_hash"])
    manual_hash = CONTROLS[dataset]["manual"][1]
    return {
        "dataset": dataset,
        "status": "SELECTED",
        "candidate_set_id": selected["candidate_set_id"],
        "candidate_set_hash": selected_hash,
        "candidate_set_ordinal": int(selected["candidate_set_ordinal"]),
        "selected_manual": selected_hash == manual_hash,
        "complete_set_only": True,
        "taskwise_set_splicing": False,
        "pareto_front": int(selected["pareto_front"]),
        "pareto_order": int(selected["pareto_order"]),
        "manual_safety": dict(safety_by_hash[selected_hash]),
    }
