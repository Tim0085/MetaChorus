"""Pure deterministic H5 stability, exploration, fidelity, and Pareto primitives."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from .constants import (
    ALL_DEVELOPMENT_MODEL_SEEDS,
    CLASSIFICATION_LOO_RHO_MEDIAN_MIN,
    CLASSIFICATION_MODEL_RHO_MIN,
    CLASSIFICATION_SPLIT_COUNT,
    CLASSIFICATION_TOP_QUARTILE_OVERLAP_MEDIAN_MIN,
    CONTROLS,
    FREEBASE_EXPLOITATION_COUNT,
    FREEBASE_EXPLORATION_BANDS,
    FREEBASE_SAMPLES_PER_BAND,
    R1_RECALL_THRESHOLDS,
    R2_CLASSIFICATION_TRUTH_KENDALL_MIN_EXCLUSIVE,
    R2_CLASSIFICATION_TRUTH_RHO_MIN,
    R3_ASSIGNMENT_STABILITY_MEDIAN_MIN,
    R3_AUDIT_COUNTS,
    R3_CLUSTER_MODEL_RHO_MIN,
    R3_CLUSTER_TRUTH_RHO_MIN,
    R3_JOINT_TRUTH_RHO_MIN,
    R3_TRUTH_KENDALL_MIN_EXCLUSIVE,
    R4_PER_SEED_DROP_LIMIT,
    R4_PROMOTION_COUNTS,
    SEARCH_MODEL_SEEDS,
)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value!r}")
    return result


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation requires matched vectors with at least two values")
    result = float(spearmanr(left, right).statistic)
    if not math.isfinite(result):
        return -1.0
    return math.copysign(1.0, result) if abs(abs(result) - 1.0) < 1e-12 else result


def _safe_kendall(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation requires matched vectors with at least two values")
    result = float(kendalltau(left, right).statistic)
    if not math.isfinite(result):
        return -1.0
    return math.copysign(1.0, result) if abs(abs(result) - 1.0) < 1e-12 else result


def classification_stability(
    repeat_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    fidelity: int,
    model_seeds: Sequence[int] = SEARCH_MODEL_SEEDS,
) -> dict[str, Any]:
    """Apply the H5 model-seed, LOO, and top-quartile classification gate."""

    grouped: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in repeat_rows:
        if str(row["dataset"]) != dataset:
            continue
        key = (str(row["candidate_set_hash"]), int(row["model_seed"]))
        split_index = int(row["classification_split_index"])
        if split_index in grouped[key]:
            raise ValueError(f"duplicate classification split for {key}/{split_index}")
        grouped[key][split_index] = _finite(row["F_cls"])
    candidate_hashes = sorted({candidate for candidate, _ in grouped})
    approved_seeds = tuple(int(seed) for seed in model_seeds)
    if len(candidate_hashes) < 2:
        raise ValueError(f"{dataset} classification reliability needs at least two candidates")
    for candidate in candidate_hashes:
        for seed in approved_seeds:
            values = grouped.get((candidate, seed))
            if values is None or sorted(values) != list(range(CLASSIFICATION_SPLIT_COUNT)):
                raise ValueError(f"incomplete H5 classification repeats for {candidate}/{seed}")
    means = {
        (candidate, seed): mean(grouped[(candidate, seed)].values())
        for candidate in candidate_hashes
        for seed in approved_seeds
    }
    if len(approved_seeds) != 2:
        raise ValueError("the stage classification gate requires exactly two search model seeds")
    rho_model = _safe_spearman(
        [means[(candidate, approved_seeds[0])] for candidate in candidate_hashes],
        [means[(candidate, approved_seeds[1])] for candidate in candidate_hashes],
    )
    loo_rhos: list[float] = []
    top_overlaps: list[float] = []
    top_k = int(math.ceil(0.25 * len(candidate_hashes)))

    def order(scores: Mapping[str, float]) -> list[str]:
        return sorted(candidate_hashes, key=lambda candidate: (-scores[candidate], candidate))

    for seed in approved_seeds:
        full_scores = {candidate: means[(candidate, seed)] for candidate in candidate_hashes}
        full_order = order(full_scores)
        full_vector = [full_scores[candidate] for candidate in candidate_hashes]
        full_top = set(full_order[:top_k])
        for omitted in range(CLASSIFICATION_SPLIT_COUNT):
            loo_scores = {
                candidate: mean(
                    value
                    for split, value in grouped[(candidate, seed)].items()
                    if split != omitted
                )
                for candidate in candidate_hashes
            }
            loo_rhos.append(
                _safe_spearman(full_vector, [loo_scores[candidate] for candidate in candidate_hashes])
            )
            loo_top = set(order(loo_scores)[:top_k])
            top_overlaps.append(len(full_top & loo_top) / top_k)
    raw_sds = [
        stdev(grouped[(candidate, seed)].values())
        for candidate in candidate_hashes
        for seed in approved_seeds
    ]
    rho_loo_median = median(loo_rhos)
    overlap_median = median(top_overlaps)
    reliable = (
        rho_model >= CLASSIFICATION_MODEL_RHO_MIN
        and rho_loo_median >= CLASSIFICATION_LOO_RHO_MEDIAN_MIN
        and overlap_median >= CLASSIFICATION_TOP_QUARTILE_OVERLAP_MEDIAN_MIN
    )
    return {
        "dataset": dataset,
        "fidelity": int(fidelity),
        "candidate_count": len(candidate_hashes),
        "model_seed_count": len(approved_seeds),
        "repeat_count_per_run": CLASSIFICATION_SPLIT_COUNT,
        "rho_model_cls": rho_model,
        "LOO_rho_median": rho_loo_median,
        "LOO_top_quartile_overlap_median": overlap_median,
        "top_quartile_k": top_k,
        "raw_split_sd_median_diagnostic": median(raw_sds),
        "naive_se_median_diagnostic": median(raw_sds) / math.sqrt(CLASSIFICATION_SPLIT_COUNT),
        "raw_sd_or_naive_se_used_as_primary_gate": False,
        "rho_model_threshold": CLASSIFICATION_MODEL_RHO_MIN,
        "loo_rho_threshold": CLASSIFICATION_LOO_RHO_MEDIAN_MIN,
        "top_overlap_threshold": CLASSIFICATION_TOP_QUARTILE_OVERLAP_MEDIAN_MIN,
        "status": "RELIABLE" if reliable else "UNRELIABLE",
    }


def aggregate_classification_by_seed(
    repeat_rows: Sequence[Mapping[str, Any]], dataset: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in repeat_rows:
        if str(row["dataset"]) == dataset:
            grouped[(str(row["candidate_set_hash"]), int(row["model_seed"]))].append(row)
    output: list[dict[str, Any]] = []
    for (candidate_hash, model_seed), rows in grouped.items():
        if sorted(int(row["classification_split_index"]) for row in rows) != list(
            range(CLASSIFICATION_SPLIT_COUNT)
        ):
            raise ValueError(f"incomplete classification repeats for {candidate_hash}/{model_seed}")
        first = rows[0]
        values = [_finite(row["F_cls"]) for row in rows]
        output.append(
            {
                "dataset": dataset,
                "candidate_set_id": str(first["candidate_set_id"]),
                "candidate_set_hash": candidate_hash,
                "candidate_set_ordinal": int(first["candidate_set_ordinal"]),
                "model_seed": model_seed,
                "mean_F_cls": mean(values),
                "sample_sd_F_cls": stdev(values),
                "repeat_count": len(values),
            }
        )
    return sorted(output, key=lambda row: (str(row["candidate_set_hash"]), int(row["model_seed"])))


def freebase_classification_ranking(
    by_seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in by_seed_rows:
        if str(row["dataset"]) == "freebase":
            grouped[str(row["candidate_set_hash"])].append(row)
    output: list[dict[str, Any]] = []
    for candidate_hash, rows in grouped.items():
        if sorted(int(row["model_seed"]) for row in rows) != sorted(SEARCH_MODEL_SEEDS):
            raise ValueError(f"Freebase candidate lacks both search seeds: {candidate_hash}")
        values = [_finite(row["mean_F_cls"]) for row in rows]
        first = rows[0]
        output.append(
            {
                "dataset": "freebase",
                "candidate_set_id": str(first["candidate_set_id"]),
                "candidate_set_hash": candidate_hash,
                "candidate_set_ordinal": int(first["candidate_set_ordinal"]),
                "median_F_cls": median(values),
                "minimum_F_cls": min(values),
                "sample_sd_model_seed_means": stdev(values),
            }
        )
    output.sort(
        key=lambda row: (
            -_finite(row["median_F_cls"]),
            -_finite(row["minimum_F_cls"]),
            _finite(row["sample_sd_model_seed_means"]),
            str(row["candidate_set_hash"]),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["classification_rank"] = rank
    return output


def select_freebase_exploration(
    classification_ranking: Sequence[Mapping[str, Any]],
    exploration_seed: int,
    *,
    excluded_hashes: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select exactly ten per frozen band using classification rank/hash only."""

    allowed = {"candidate_set_hash", "classification_rank"}
    minimal: list[dict[str, Any]] = []
    for row in classification_ranking:
        forbidden = {
            key
            for key in row
            if any(token in key.lower() for token in ("truth", "clu", "manual", "automatic", "control"))
        }
        if forbidden:
            raise ValueError(f"exploration input contains forbidden fields: {sorted(forbidden)}")
        minimal.append({key: row[key] for key in allowed})
    by_rank = {int(row["classification_rank"]): str(row["candidate_set_hash"]) for row in minimal}
    if sorted(by_rank) != list(range(1, len(minimal) + 1)):
        raise ValueError("classification ranking is not a complete canonical rank sequence")
    control_hashes = {value[1] for value in CONTROLS["freebase"].values()}
    exploitation_hashes = {
        str(row["candidate_set_hash"])
        for row in classification_ranking
        if str(row["candidate_set_hash"]) not in control_hashes
    }
    exploitation_hashes = set(
        sorted(
            exploitation_hashes,
            key=lambda candidate_hash: next(
                int(row["classification_rank"])
                for row in classification_ranking
                if str(row["candidate_set_hash"]) == candidate_hash
            ),
        )[:FREEBASE_EXPLOITATION_COUNT]
    )
    excluded = (
        {str(value) for value in excluded_hashes}
        | control_hashes
        | exploitation_hashes
    )
    rng = np.random.default_rng(int(exploration_seed))
    output: list[dict[str, Any]] = []
    for band_index, (start, end) in enumerate(FREEBASE_EXPLORATION_BANDS):
        band = np.asarray(
            [
                by_rank[rank]
                for rank in range(start, end + 1)
                if by_rank[rank] not in excluded
            ],
            dtype=object,
        )
        if len(band) < FREEBASE_SAMPLES_PER_BAND:
            raise ValueError(f"Freebase exploration band {start}-{end} has too few eligible members")
        positions = rng.choice(len(band), size=FREEBASE_SAMPLES_PER_BAND, replace=False)
        selected = sorted(str(band[position]) for position in positions.tolist())
        for candidate_hash in selected:
            rank = next(rank for rank in range(start, end + 1) if by_rank[rank] == candidate_hash)
            output.append(
                {
                    "candidate_set_hash": candidate_hash,
                    "classification_rank": rank,
                    "rank_band_index": band_index,
                    "rank_band_start": start,
                    "rank_band_end": end,
                    "selection_role": "FIXED_RANDOM_EXPLORATION",
                    "exploration_seed": int(exploration_seed),
                    "truth_fields_read": False,
                    "clustering_fields_read": False,
                    "control_identity_read": False,
                }
            )
    if len(output) != len(FREEBASE_EXPLORATION_BANDS) * FREEBASE_SAMPLES_PER_BAND:
        raise RuntimeError("Freebase exploration did not produce exactly forty members")
    return sorted(output, key=lambda row: (int(row["rank_band_index"]), int(row["classification_rank"])))


def _attach_controls(
    selected: Sequence[Mapping[str, Any]],
    full_cohort: Sequence[Mapping[str, Any]],
    dataset: str,
    *,
    selected_role: str,
) -> list[dict[str, Any]]:
    by_hash = {str(row["candidate_set_hash"]): dict(row) for row in full_cohort}
    roles: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        roles[str(row["candidate_set_hash"])].add(selected_role)
    for name, (_, candidate_hash) in CONTROLS[dataset].items():
        if candidate_hash not in by_hash:
            raise ValueError(f"missing {dataset} control: {name}")
        roles[candidate_hash].add(name.upper())
    output: list[dict[str, Any]] = []
    control_hashes = {value[1] for value in CONTROLS[dataset].values()}
    for candidate_hash, role_set in roles.items():
        source = by_hash[candidate_hash]
        output.append(
            {
                "dataset": dataset,
                "candidate_set_id": str(source["candidate_set_id"]),
                "candidate_set_hash": candidate_hash,
                "candidate_set_ordinal": int(source["candidate_set_ordinal"]),
                "roles_json": json.dumps(sorted(role_set), ensure_ascii=False),
                "consumes_scientific_slot": (
                    selected_role in role_set and candidate_hash not in control_hashes
                ),
                "controls_outside_slots": True,
            }
        )
    return sorted(output, key=lambda row: (int(row["candidate_set_ordinal"]), str(row["candidate_set_hash"])))


def build_r1_pool(
    *,
    acm_candidates: Sequence[Mapping[str, Any]],
    freebase_ranking: Sequence[Mapping[str, Any]],
    freebase_exploration: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(acm_candidates) != 10 or len({str(row["candidate_set_hash"]) for row in acm_candidates}) != 10:
        raise ValueError("H5 must retain all ten ACM candidates")
    acm = _attach_controls(acm_candidates, acm_candidates, "acm", selected_role="ALL10_RETAINED")
    if len(freebase_ranking) != 220:
        raise ValueError("H5 Freebase ranking must cover all 220 candidates")
    control_hashes = {value[1] for value in CONTROLS["freebase"].values()}
    exploitation = [
        dict(row)
        for row in freebase_ranking
        if str(row["candidate_set_hash"]) not in control_hashes
    ][:FREEBASE_EXPLOITATION_COUNT]
    if len(exploitation) != FREEBASE_EXPLOITATION_COUNT:
        raise ValueError("H5 Freebase exploitation lacks eighty non-control candidates")
    exploitation_hashes = {str(row["candidate_set_hash"]) for row in exploitation}
    explored_hashes = {str(row["candidate_set_hash"]) for row in freebase_exploration}
    if len(explored_hashes) != 40:
        raise ValueError("H5 requires forty unique exploration candidates")
    if explored_hashes & (exploitation_hashes | control_hashes):
        raise ValueError("H5 exploration overlaps exploitation or a control identity")
    by_hash = {str(row["candidate_set_hash"]): dict(row) for row in freebase_ranking}
    selected: list[dict[str, Any]] = []
    for row in exploitation:
        value = dict(row)
        value["pool_source_role"] = "EXPLOITATION_TOP80"
        selected.append(value)
    for candidate_hash in sorted(explored_hashes):
        value = dict(by_hash[candidate_hash])
        value["pool_source_role"] = "FOUR_BAND_EXPLORATION"
        selected.append(value)
    
    freebase = _attach_controls(selected, freebase_ranking, "freebase", selected_role="H5_R1_POOL")
    if not (120 <= len(freebase) <= 122):
        raise RuntimeError(f"unexpected H5 Freebase pool size after control dedup: {len(freebase)}")
    return acm + freebase


def recall_count(
    pool_hashes: Iterable[str],
    truth_rows: Sequence[Mapping[str, Any]],
    *,
    rank_field: str,
    top_k: int,
) -> tuple[int, list[str]]:
    pool = set(pool_hashes)
    ordered = sorted(
        truth_rows,
        key=lambda row: (_finite(row[rank_field]), str(row["candidate_set_hash"])),
    )[:top_k]
    captured = [str(row["candidate_set_hash"]) for row in ordered if str(row["candidate_set_hash"]) in pool]
    return len(captured), captured


def r1_recall_audit(
    pool_rows: Sequence[Mapping[str, Any]], truth_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = [
        {
            "dataset": "acm",
            "metric": "all10",
            "top_k": 10,
            "captured": 10,
            "threshold": 10,
            "captured_hashes_json": json.dumps(
                sorted(
                    str(row["candidate_set_hash"])
                    for row in pool_rows
                    if str(row["dataset"]) == "acm"
                ),
                ensure_ascii=False,
            ),
            "status": "PASS",
        }
    ]
    freebase_pool = {
        str(row["candidate_set_hash"])
        for row in pool_rows
        if str(row["dataset"]) == "freebase"
    }
    freebase_truth = [row for row in truth_rows if str(row["dataset"]) == "freebase"]
    requirements = (
        ("classification_top20", "classification_rank", 20, R1_RECALL_THRESHOLDS["freebase_classification_top20"]),
        ("joint_top20", "joint_rank", 20, R1_RECALL_THRESHOLDS["freebase_joint_top20"]),
        ("joint_top10", "joint_rank", 10, R1_RECALL_THRESHOLDS["freebase_joint_top10"]),
        ("clustering_top20", "clustering_rank", 20, R1_RECALL_THRESHOLDS["freebase_clustering_top20"]),
    )
    for metric, rank_field, top_k, threshold in requirements:
        captured, hashes = recall_count(
            freebase_pool, freebase_truth, rank_field=rank_field, top_k=top_k
        )
        output.append(
            {
                "dataset": "freebase",
                "metric": metric,
                "top_k": top_k,
                "captured": captured,
                "threshold": threshold,
                "captured_hashes_json": json.dumps(hashes, ensure_ascii=False),
                "status": "PASS" if captured >= threshold else "FAIL",
            }
        )
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
    stage = {
        str(row["candidate_set_hash"]): _finite(row[stage_field])
        for row in stage_rows
        if str(row["dataset"]) == dataset
    }
    truth = {
        str(row["candidate_set_hash"]): _finite(row[truth_field])
        for row in truth_rows
        if str(row["dataset"]) == dataset
    }
    hashes = sorted(set(stage) & set(truth))
    if len(hashes) < 2 or set(stage) != set(hashes):
        raise ValueError(f"incomplete {dataset}/{task} truth fidelity cohort")
    left = [stage[candidate] for candidate in hashes]
    right = [truth[candidate] for candidate in hashes]
    return {
        "dataset": dataset,
        "task": task,
        "candidate_count": len(hashes),
        "spearman_to_3000_truth": _safe_spearman(left, right),
        "kendall_to_3000_truth": _safe_kendall(left, right),
        "stage_field": stage_field,
        "truth_field": truth_field,
    }


def classification_bridge_decision(
    stability_rows: Sequence[Mapping[str, Any]],
    fidelity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_dataset_stability = {str(row["dataset"]): row for row in stability_rows}
    by_dataset_fidelity = {str(row["dataset"]): row for row in fidelity_rows}
    datasets = ("acm", "freebase")
    reasons: list[str] = []
    for dataset in datasets:
        stability = by_dataset_stability.get(dataset)
        fidelity = by_dataset_fidelity.get(dataset)
        if stability is None or fidelity is None:
            reasons.append(f"{dataset}:missing_evidence")
            continue
        if stability["status"] != "RELIABLE":
            reasons.append(f"{dataset}:classification_unreliable")
        if _finite(fidelity["spearman_to_3000_truth"]) < R2_CLASSIFICATION_TRUTH_RHO_MIN:
            reasons.append(f"{dataset}:truth_spearman")
        if _finite(fidelity["kendall_to_3000_truth"]) <= R2_CLASSIFICATION_TRUTH_KENDALL_MIN_EXCLUSIVE:
            reasons.append(f"{dataset}:truth_kendall")
    return {
        "status": "H5_R2_BRIDGE_PASS" if not reasons else "H5_R2_CLASSIFICATION_BRIDGE_FAIL",
        "r2_role": "BRIDGE_FIDELITY_NO_PRUNING",
        "r3_started": False,
        "all_r2_sets_continue_if_pass": True,
        "clustering_used_for_pruning": False,
        "reasons": reasons,
    }


def clustering_model_reliability(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    model_seeds: Sequence[int] = SEARCH_MODEL_SEEDS,
) -> dict[str, Any]:
    by_key = {
        (str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in summary_rows
        if str(row["dataset"]) == dataset
    }
    hashes = sorted({candidate for candidate, _ in by_key})
    seeds = tuple(int(seed) for seed in model_seeds)
    if len(seeds) != 2 or any((candidate, seed) not in by_key for candidate in hashes for seed in seeds):
        raise ValueError(f"incomplete {dataset} two-seed clustering cohort")
    rho = _safe_spearman(
        [_finite(by_key[(candidate, seeds[0])]["mean_F_clu"]) for candidate in hashes],
        [_finite(by_key[(candidate, seeds[1])]["mean_F_clu"]) for candidate in hashes],
    )
    stability = median(
        _finite(by_key[(candidate, seed)]["assignment_stability_mean"])
        for candidate in hashes
        for seed in seeds
    )
    return {
        "dataset": dataset,
        "candidate_count": len(hashes),
        "rho_model_clu": rho,
        "assignment_stability_mean_median": stability,
        "rho_threshold": R3_CLUSTER_MODEL_RHO_MIN,
        "assignment_stability_threshold": R3_ASSIGNMENT_STABILITY_MEDIAN_MIN,
        "status": (
            "RELIABLE"
            if rho >= R3_CLUSTER_MODEL_RHO_MIN
            and stability >= R3_ASSIGNMENT_STABILITY_MEDIAN_MIN
            else "UNRELIABLE"
        ),
    }


def aggregate_stage_objectives(
    classification_by_seed: Sequence[Mapping[str, Any]],
    clustering_summary_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    model_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    cls = {
        (str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in classification_by_seed
        if str(row["dataset"]) == dataset
    }
    clu = {
        (str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in clustering_summary_rows
        if str(row["dataset"]) == dataset
    }
    hashes = sorted({candidate for candidate, _ in cls})
    seeds = tuple(int(seed) for seed in model_seeds)
    output: list[dict[str, Any]] = []
    for candidate in hashes:
        if any((candidate, seed) not in cls or (candidate, seed) not in clu for seed in seeds):
            raise ValueError(f"candidate lacks complete task/seed metrics: {candidate}")
        cls_values = [_finite(cls[(candidate, seed)]["mean_F_cls"]) for seed in seeds]
        clu_values = [_finite(clu[(candidate, seed)]["mean_F_clu"]) for seed in seeds]
        first = cls[(candidate, seeds[0])]
        output.append(
            {
                "dataset": dataset,
                "candidate_set_id": str(first["candidate_set_id"]),
                "candidate_set_hash": candidate,
                "candidate_set_ordinal": int(first["candidate_set_ordinal"]),
                "seed_count": len(seeds),
                "Q_cls": median(cls_values),
                "Q_clu": median(clu_values),
                "W_cls": min(cls_values),
                "W_clu": min(clu_values),
                "assignment_stability_mean": median(
                    _finite(clu[(candidate, seed)]["assignment_stability_mean"])
                    for seed in seeds
                ),
                "per_seed_json": json.dumps(
                    [
                        {
                            "model_seed": seed,
                            "F_cls": cls_values[index],
                            "F_clu": clu_values[index],
                        }
                        for index, seed in enumerate(seeds)
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return output


def joint_normalized_midrank(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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


def r3_gate_decision(
    *,
    classification_stability_rows: Sequence[Mapping[str, Any]],
    clustering_stability_rows: Sequence[Mapping[str, Any]],
    classification_fidelity_rows: Sequence[Mapping[str, Any]],
    clustering_fidelity_rows: Sequence[Mapping[str, Any]],
    joint_fidelity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    for dataset in ("acm", "freebase"):
        cls_stability = next((row for row in classification_stability_rows if str(row["dataset"]) == dataset), None)
        clu_stability = next((row for row in clustering_stability_rows if str(row["dataset"]) == dataset), None)
        cls_fidelity = next((row for row in classification_fidelity_rows if str(row["dataset"]) == dataset), None)
        clu_fidelity = next((row for row in clustering_fidelity_rows if str(row["dataset"]) == dataset), None)
        joint_fidelity = next((row for row in joint_fidelity_rows if str(row["dataset"]) == dataset), None)
        if any(
            value is None
            for value in (
                cls_stability,
                clu_stability,
                cls_fidelity,
                clu_fidelity,
                joint_fidelity,
            )
        ):
            reasons.append(f"{dataset}:missing_evidence")
            continue
        if cls_stability["status"] != "RELIABLE":
            reasons.append(f"{dataset}:classification_reliability")
        if clu_stability["status"] != "RELIABLE":
            reasons.append(f"{dataset}:clustering_reliability")
        if _finite(cls_fidelity["spearman_to_3000_truth"]) < R2_CLASSIFICATION_TRUTH_RHO_MIN:
            reasons.append(f"{dataset}:classification_truth_spearman")
        if _finite(cls_fidelity["kendall_to_3000_truth"]) <= R3_TRUTH_KENDALL_MIN_EXCLUSIVE:
            reasons.append(f"{dataset}:classification_truth_kendall")
        if _finite(clu_fidelity["spearman_to_3000_truth"]) < R3_CLUSTER_TRUTH_RHO_MIN:
            reasons.append(f"{dataset}:clustering_truth_spearman")
        if _finite(joint_fidelity["spearman_to_3000_truth"]) < R3_JOINT_TRUTH_RHO_MIN:
            reasons.append(f"{dataset}:joint_truth_spearman")
        if _finite(joint_fidelity["kendall_to_3000_truth"]) <= R3_TRUTH_KENDALL_MIN_EXCLUSIVE:
            reasons.append(f"{dataset}:joint_truth_kendall")
    return {
        "status": (
            "TC_LPMSO_H5_R3_GATE_PASS"
            if not reasons
            else "TC_LPMSO_H5_R3_RELIABILITY_OR_FIDELITY_FAIL"
        ),
        "third_seed_started": False,
        "reasons": reasons,
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        _finite(left["Q_cls"]) >= _finite(right["Q_cls"])
        and _finite(left["Q_clu"]) >= _finite(right["Q_clu"])
        and (
            _finite(left["Q_cls"]) > _finite(right["Q_cls"])
            or _finite(left["Q_clu"]) > _finite(right["Q_clu"])
        )
    )


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
    count = max(len(output) - 1, 1)
    cls_ranks = rankdata([-_finite(row["Q_cls"]) for row in output], method="average")
    clu_ranks = rankdata([-_finite(row["Q_clu"]) for row in output], method="average")
    per_seed = {
        str(row["candidate_set_hash"]): json.loads(str(row["per_seed_json"]))
        for row in output
    }
    seeds = sorted({int(item["model_seed"]) for values in per_seed.values() for item in values})
    worst: dict[tuple[str, str], float] = {}
    for task in ("cls", "clu"):
        field = f"F_{task}"
        for seed in seeds:
            cohort = [
                (candidate_hash, item)
                for candidate_hash, values in per_seed.items()
                for item in values
                if int(item["model_seed"]) == seed
            ]
            ranks = rankdata(
                [-_finite(item[field]) for _, item in cohort], method="average"
            )
            for (candidate_hash, _), rank in zip(cohort, ranks):
                key = (candidate_hash, task)
                worst[key] = max(float(rank), worst.get(key, 0.0))
    for row, cls_rank, clu_rank in zip(output, cls_ranks, clu_ranks):
        candidate = str(row["candidate_set_hash"])
        row["normalized_rank_cls"] = (float(cls_rank) - 1.0) / count
        row["normalized_rank_clu"] = (float(clu_rank) - 1.0) / count
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
            -_finite(row["assignment_stability_mean"]),
            str(row["candidate_set_hash"]),
        )
    )
    for index, row in enumerate(output, start=1):
        row["pareto_order"] = index
    return output


def promotion_with_controls(
    ordered: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    count: int,
    role: str,
) -> list[dict[str, Any]]:
    full = [dict(row) for row in ordered if str(row["dataset"]) == dataset]
    control_hashes = {value[1] for value in CONTROLS[dataset].values()}
    eligible = [row for row in full if str(row["candidate_set_hash"]) not in control_hashes]
    return _attach_controls(eligible[:count], full, dataset, selected_role=role)


def r3_audit_pool(ordered: Sequence[Mapping[str, Any]], dataset: str) -> list[dict[str, Any]]:
    return promotion_with_controls(
        ordered,
        dataset=dataset,
        count=R3_AUDIT_COUNTS[dataset],
        role="R3_TWO_SEED_AUDIT_POOL",
    )


def three_seed_pairwise_reliability(
    per_seed_rows: Sequence[Mapping[str, Any]], *, dataset: str, task_field: str
) -> dict[str, Any]:
    by_key = {
        (str(row["candidate_set_hash"]), int(row["model_seed"])): _finite(row[task_field])
        for row in per_seed_rows
        if str(row["dataset"]) == dataset
    }
    hashes = sorted({candidate for candidate, _ in by_key})
    seeds = ALL_DEVELOPMENT_MODEL_SEEDS
    if any((candidate, seed) not in by_key for candidate in hashes for seed in seeds):
        raise ValueError(f"incomplete three-seed cohort for {dataset}/{task_field}")
    pairwise = [
        _safe_spearman(
            [by_key[(candidate, left)] for candidate in hashes],
            [by_key[(candidate, right)] for candidate in hashes],
        )
        for left_index, left in enumerate(seeds)
        for right in seeds[left_index + 1 :]
    ]
    value = median(pairwise)
    return {
        "dataset": dataset,
        "task_field": task_field,
        "candidate_count": len(hashes),
        "model_seed_count": 3,
        "pairwise_spearman_json": json.dumps(pairwise),
        "median_pairwise_rho": value,
        "threshold": CLASSIFICATION_MODEL_RHO_MIN,
        "status": "RELIABLE" if value >= CLASSIFICATION_MODEL_RHO_MIN else "UNRELIABLE",
    }


def r4_promotion(ordered: Sequence[Mapping[str, Any]], dataset: str) -> list[dict[str, Any]]:
    return promotion_with_controls(
        ordered,
        dataset=dataset,
        count=R4_PROMOTION_COUNTS[dataset],
        role="R4_PROMOTED",
    )


def manual_safety(
    candidate_per_seed: Mapping[int, Mapping[str, float]],
    manual_per_seed: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    seeds = ALL_DEVELOPMENT_MODEL_SEEDS
    if set(candidate_per_seed) != set(seeds) or set(manual_per_seed) != set(seeds):
        raise ValueError("manual safety requires exactly the three development seeds")
    output: dict[str, Any] = {"seed_count": 3}
    all_safe = True
    positive_joint_means: list[float] = []
    for task in ("cls", "clu"):
        field = f"F_{task}"
        deltas = [
            _finite(candidate_per_seed[seed][field]) - _finite(manual_per_seed[seed][field])
            for seed in seeds
        ]
        paired_se = stdev(deltas) / math.sqrt(len(deltas))
        mean_delta = mean(deltas)
        safe = mean_delta >= -paired_se and all(delta > -R4_PER_SEED_DROP_LIMIT for delta in deltas)
        output[f"delta_{task}_per_seed_json"] = json.dumps(deltas)
        output[f"mean_delta_{task}"] = mean_delta
        output[f"paired_se_delta_{task}"] = paired_se
        output[f"every_seed_above_minus_2pp_{task}"] = all(
            delta > -R4_PER_SEED_DROP_LIMIT for delta in deltas
        )
        output[f"safe_{task}"] = safe
        positive_joint_means.append(mean_delta)
        all_safe = all_safe and safe
    output["mean_joint_delta"] = mean(positive_joint_means)
    output["status"] = "SAFE" if all_safe else "UNSAFE"
    return output


def choose_single_r4_set(
    ordered: Sequence[Mapping[str, Any]],
    safety_by_hash: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
) -> dict[str, Any]:
    cohort = [dict(row) for row in ordered if str(row["dataset"]) == dataset]
    safe = [
        row
        for row in cohort
        if safety_by_hash.get(str(row["candidate_set_hash"]), {}).get("status") == "SAFE"
    ]
    if not safe:
        return {"dataset": dataset, "status": "TC_LPMSO_H5_DEVELOPMENT_FAIL", "reason": "NO_SAFE_SET"}
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
    manual_hash = CONTROLS[dataset]["manual"][1]
    return {
        "dataset": dataset,
        "status": "SELECTED",
        "candidate_set_id": selected["candidate_set_id"],
        "candidate_set_hash": selected["candidate_set_hash"],
        "candidate_set_ordinal": selected["candidate_set_ordinal"],
        "selected_manual": str(selected["candidate_set_hash"]) == manual_hash,
        "complete_set_only": True,
        "taskwise_set_splicing": False,
        "pareto_front": selected["pareto_front"],
        "pareto_order": selected["pareto_order"],
        "manual_safety": dict(safety_by_hash[str(selected["candidate_set_hash"])]),
    }
