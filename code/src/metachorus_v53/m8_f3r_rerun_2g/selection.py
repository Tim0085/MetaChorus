"""Ablation selection from search-stage summaries."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .common import canonical_json_hash
from .constants import ABLATION_VARIANTS, FROZEN_SETS, SEARCH_MODEL_SEEDS, SELECTION_FIELDS

OBJECTIVES = ("Q_cls", "Q_clu", "W_cls", "W_clu")


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite search objective: {field}")
    return result


def _field(row: Mapping[str, Any], name: str) -> float:
    aliases = {
        "Q_cls": ("Q_cls", "Q_cls_R4"),
        "Q_clu": ("Q_clu", "Q_clu_R4"),
        "W_cls": ("W_cls", "W_cls_R4"),
        "W_clu": ("W_clu", "W_clu_R4"),
        "classification_sd": ("classification_sd", "sample_sd_F_cls", "sample_sd_F_cls_R4"),
        "assignment_stability": ("assignment_stability", "assignment_stability_mean"),
    }
    for key in aliases.get(name, (name,)):
        if key in row and str(row[key]) != "":
            return _finite(row[key], key)
    raise ValueError(f"search objective is missing: {name}")


def _identity(row: Mapping[str, Any]) -> str:
    value = str(row.get("set_hash", row.get("candidate_set_hash", ""))).casefold()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("candidate set hash is invalid")
    return value


def _set_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("set_id", row.get("candidate_set_id", "")))
    if not value:
        raise ValueError("candidate set id is missing")
    return value


def normalized_midranks(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    """Descending normalized mid-rank in [0, 1], where zero is best."""
    if not rows:
        raise ValueError("cannot rank an empty candidate pool")
    ordered = sorted(((_identity(row), _field(row, field)) for row in rows), key=lambda item: (-item[1], item[0]))
    output: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        midrank = ((index + 1) + end) / 2.0
        normalized = 0.0 if len(ordered) == 1 else (midrank - 1.0) / (len(ordered) - 1.0)
        for position in range(index, end):
            output[ordered[position][0]] = normalized
        index = end
    return output


def first_pareto_front(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    """Return the deterministic first front for objectives that are maximized."""
    result: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        values = [_field(candidate, field) for field in fields]
        for other in rows:
            if _identity(candidate) == _identity(other):
                continue
            other_values = [_field(other, field) for field in fields]
            if all(left >= right for left, right in zip(other_values, values)) and any(
                left > right for left, right in zip(other_values, values)
            ):
                dominated = True
                break
        if not dominated:
            result.append(dict(candidate))
    return sorted(result, key=_identity)


def _safe_rows(rows: Sequence[Mapping[str, Any]], safety: Mapping[str, str]) -> list[dict[str, Any]]:
    safe = [dict(row) for row in rows if safety.get(_identity(row)) == "PASS"]
    if not safe:
        raise ValueError("Manual safety mask contains no PASS candidate")
    return safe


def _robust_choice(rows: Sequence[Mapping[str, Any]], *, pareto: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranks = {field: normalized_midranks(rows, field) for field in OBJECTIVES}
    enriched = []
    for row in rows:
        identity = _identity(row)
        r_median = max(ranks["Q_cls"][identity], ranks["Q_clu"][identity])
        r_worst = max(ranks["W_cls"][identity], ranks["W_clu"][identity])
        enriched.append(
            {
                **dict(row),
                "R_Q_cls": ranks["Q_cls"][identity],
                "R_Q_clu": ranks["Q_clu"][identity],
                "R_W_cls": ranks["W_cls"][identity],
                "R_W_clu": ranks["W_clu"][identity],
                "R_median": r_median,
                "R_worst": r_worst,
                "R_robust": max(r_median, r_worst),
            }
        )
    eligible = first_pareto_front(enriched, OBJECTIVES) if pareto else enriched
    ordered = sorted(
        eligible,
        key=lambda row: (
            float(row["R_robust"]),
            statistics.fmean((float(row["R_median"]), float(row["R_worst"]))),
            -min(_field(row, "W_cls"), _field(row, "W_clu")),
            -_field(row, "assignment_stability"),
            _identity(row),
        ),
    )
    return dict(ordered[0]), enriched


def _per_seed(row: Mapping[str, Any]) -> dict[int, dict[str, float]]:
    payload = row.get("per_seed")
    if payload is None:
        payload = row.get("per_seed_json")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, Mapping):
        items = [dict(value, model_seed=int(key)) for key, value in payload.items()]
    elif isinstance(payload, Sequence):
        items = [dict(item) for item in payload]
    else:
        raise ValueError("candidate lacks frozen per-seed search metrics")
    result: dict[int, dict[str, float]] = {}
    for item in items:
        seed = int(item["model_seed"])
        result[seed] = {
            "F_cls": _finite(item.get("F_cls", item.get("mean_F_cls")), "F_cls"),
            "F_clu": _finite(item.get("F_clu", item.get("mean_F_clu")), "F_clu"),
            "assignment_stability": _finite(
                item.get("assignment_stability", item.get("assignment_stability_mean", row.get("assignment_stability_mean"))),
                "assignment_stability",
            ),
        }
    return result


def _single_seed_choice(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_rows = []
    for row in rows:
        values = _per_seed(row)
        if seed not in values:
            raise ValueError("single-seed ablation lacks its exact frozen seed")
        seed_rows.append({**dict(row), **values[seed], "Q_cls": values[seed]["F_cls"], "Q_clu": values[seed]["F_clu"]})
    front = first_pareto_front(seed_rows, ("Q_cls", "Q_clu"))
    cls_rank = normalized_midranks(front, "Q_cls")
    clu_rank = normalized_midranks(front, "Q_clu")
    enriched = []
    for row in front:
        identity = _identity(row)
        enriched.append(
            {
                **row,
                "R_cls": cls_rank[identity],
                "R_clu": clu_rank[identity],
                "R_single": max(cls_rank[identity], clu_rank[identity]),
            }
        )
    enriched.sort(
        key=lambda row: (
            row["R_single"],
            statistics.fmean((row["R_cls"], row["R_clu"])),
            -_field(row, "assignment_stability"),
            _identity(row),
        )
    )
    return dict(enriched[0]), enriched


def _median_only_choice(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    front = first_pareto_front(rows, ("Q_cls", "Q_clu"))
    cls_rank = normalized_midranks(front, "Q_cls")
    clu_rank = normalized_midranks(front, "Q_clu")
    enriched = []
    for row in front:
        identity = _identity(row)
        enriched.append(
            {
                **dict(row),
                "R_Q_cls": cls_rank[identity],
                "R_Q_clu": clu_rank[identity],
                "R_median": max(cls_rank[identity], clu_rank[identity]),
            }
        )
    enriched.sort(key=lambda row: (row["R_median"], statistics.fmean((row["R_Q_cls"], row["R_Q_clu"])), _identity(row)))
    return dict(enriched[0]), enriched


def _scalar_choice(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranks = {field: normalized_midranks(rows, field) for field in OBJECTIVES}
    enriched = []
    for row in rows:
        identity = _identity(row)
        values = [ranks[field][identity] for field in OBJECTIVES]
        enriched.append({**dict(row), **{f"R_{field}": ranks[field][identity] for field in OBJECTIVES}, "R_scalar": max(values), "rank_mean": statistics.fmean(values)})
    enriched.sort(key=lambda row: (row["R_scalar"], row["rank_mean"], -min(_field(row, "W_cls"), _field(row, "W_clu")), _identity(row)))
    return dict(enriched[0]), enriched


def _classification_choice(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (-_field(row, "Q_cls"), -_field(row, "W_cls"), _field(row, "classification_sd"), _identity(row)))
    return dict(ordered[0]), [dict(row) for row in ordered]


def _clustering_choice(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (-_field(row, "Q_clu"), -_field(row, "W_clu"), -_field(row, "assignment_stability"), _identity(row)))
    return dict(ordered[0]), [dict(row) for row in ordered]


def _selection_row(dataset: str, variant: str, chosen: Mapping[str, Any], pool_identity: str, **flags: Any) -> dict[str, Any]:
    paths = chosen.get("path_ids", ())
    if isinstance(paths, str):
        try:
            paths = json.loads(paths)
        except json.JSONDecodeError:
            paths = [paths]
    row = {
        "dataset": dataset,
        "ablation_variant": variant,
        "selection_seed_role": flags.pop("selection_seed_role", "THREE_SEED_FROZEN_SEARCH"),
        "set_id": _set_id(chosen),
        "set_hash": _identity(chosen),
        "path_ids": json.dumps(list(paths), ensure_ascii=False, separators=(",", ":")),
        "selection_pool_identity": pool_identity,
        "manual_safety_used": flags.pop("manual_safety_used", True),
        "random_exploration_used": flags.pop("random_exploration_used", True),
        "Pareto_used": flags.pop("Pareto_used", True),
        "median_objectives_used": flags.pop("median_objectives_used", True),
        "worst_case_objectives_used": flags.pop("worst_case_objectives_used", True),
        "classification_used": flags.pop("classification_used", True),
        "clustering_used": flags.pop("clustering_used", True),
        "search_metric_source": flags.pop("search_metric_source", "FROZEN_SEARCH_STAGE_ONLY"),
        "status": flags.pop("status", "PASS"),
    }
    if flags:
        raise ValueError(f"unknown selection flags: {sorted(flags)}")
    return row


def select_dataset_variants(
    dataset: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    manual_safety_rows: Sequence[Mapping[str, Any]],
    *,
    exploration_only_hashes: Iterable[str] = (),
    no_exploration_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select all ten variants from frozen search evidence only."""
    dataset = dataset.casefold()
    if dataset not in SEARCH_MODEL_SEEDS:
        raise ValueError("unsupported ablation dataset")
    rows = [dict(row) for row in candidate_rows if str(row.get("dataset", dataset)).casefold() == dataset]
    identities = [_identity(row) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError("candidate pool is empty or contains duplicate set hashes")
    safety = {
        _identity(row): str(row.get("status", ""))
        for row in manual_safety_rows
        if str(row.get("dataset", dataset)).casefold() == dataset
    }
    if set(safety) != set(identities):
        raise ValueError("Manual safety mask must cover the complete R4 pool")
    safe = _safe_rows(rows, safety)
    pool_identity = canonical_json_hash(sorted(identities))
    by_hash = {_identity(row): row for row in rows}
    frozen = FROZEN_SETS[dataset]["FULL_FROZEN"]
    full = by_hash.get(str(frozen["set_hash"]), dict(frozen))
    selected: list[dict[str, Any]] = [
        _selection_row(dataset, "FULL_FROZEN", full, pool_identity, search_metric_source="FINAL_FROZEN_REGISTRY_IDENTITY_ONLY")
    ]
    components: list[dict[str, Any]] = []
    for ordinal, seed in enumerate(SEARCH_MODEL_SEEDS[dataset]):
        choice, detail = _single_seed_choice(safe, seed)
        variant = f"SINGLE_SEED_RANK_S{ordinal}"
        selected.append(
            _selection_row(
                dataset,
                variant,
                choice,
                pool_identity,
                selection_seed_role=f"SEARCH_SEED_{seed}",
                median_objectives_used=False,
                worst_case_objectives_used=False,
            )
        )
        components.extend({"dataset": dataset, "ablation_variant": variant, "component": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)} for row in detail)
    excluded = {str(value).casefold() for value in exploration_only_hashes}
    no_exploration_pool = [dict(row) for row in (no_exploration_rows or rows) if _identity(row) not in excluded]
    no_exploration_safe = _safe_rows(no_exploration_pool, safety)
    choice, detail = _robust_choice(no_exploration_safe)
    selected.append(_selection_row(dataset, "NO_RANDOM_EXPLORATION", choice, canonical_json_hash(sorted(_identity(row) for row in no_exploration_pool)), random_exploration_used=False))
    components.extend({"dataset": dataset, "ablation_variant": "NO_RANDOM_EXPLORATION", "component": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)} for row in detail)
    choice, detail = _median_only_choice(safe)
    selected.append(_selection_row(dataset, "MEDIAN_ONLY_NO_WORST_CASE", choice, pool_identity, worst_case_objectives_used=False))
    components.extend({"dataset": dataset, "ablation_variant": "MEDIAN_ONLY_NO_WORST_CASE", "component": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)} for row in detail)
    choice, detail = _scalar_choice(safe)
    selected.append(_selection_row(dataset, "BALANCED_SCALAR_ONLY_NO_PARETO", choice, pool_identity, Pareto_used=False))
    components.extend({"dataset": dataset, "ablation_variant": "BALANCED_SCALAR_ONLY_NO_PARETO", "component": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)} for row in detail)
    choice, detail = _classification_choice(safe)
    selected.append(_selection_row(dataset, "CLASSIFICATION_ONLY", choice, pool_identity, Pareto_used=False, clustering_used=False))
    components.extend({"dataset": dataset, "ablation_variant": "CLASSIFICATION_ONLY", "component": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)} for row in detail)
    choice, detail = _clustering_choice(safe)
    selected.append(_selection_row(dataset, "CLUSTERING_ONLY", choice, pool_identity, Pareto_used=False, classification_used=False))
    components.extend({"dataset": dataset, "ablation_variant": "CLUSTERING_ONLY", "component": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)} for row in detail)
    choice, detail = _robust_choice(rows)
    selected.append(_selection_row(dataset, "NO_MANUAL_SAFETY", choice, pool_identity, manual_safety_used=False))
    components.extend({"dataset": dataset, "ablation_variant": "NO_MANUAL_SAFETY", "component": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)} for row in detail)
    if [row["ablation_variant"] for row in selected] != list(ABLATION_VARIANTS):
        raise AssertionError("internal variant ordering drift")
    if any(tuple(row) != SELECTION_FIELDS for row in selected):
        raise AssertionError("selection CSV schema drift")
    return selected, components


def validate_selection_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    expected = {(dataset, variant) for dataset in SEARCH_MODEL_SEEDS for variant in ABLATION_VARIANTS}
    observed = [(str(row.get("dataset", "")).casefold(), str(row.get("ablation_variant", ""))) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("exact ablation variant coverage is invalid")
    if any(str(row.get("search_metric_source", "")) == "" or "LOCKED" in str(row.get("search_metric_source", "")).upper() for row in rows):
        raise ValueError("selection attempted to use locked report evidence")
    return {dataset: sum(identity[0] == dataset for identity in observed) for dataset in SEARCH_MODEL_SEEDS}


__all__ = [name for name in globals() if not name.startswith("_")]

