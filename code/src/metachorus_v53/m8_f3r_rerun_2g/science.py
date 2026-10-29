"""Scientific-table construction and strict consistency checks."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .constants import (
    ABLATION_DATASETS,
    ABLATION_VARIANTS,
    EVALUATION_CONTRACTS,
    EVALUATION_MODEL_SEEDS,
)

RAW_METRICS = ("Micro_F1", "Macro_F1", "NMI", "ARI")
DERIVED_METRICS = ("F_cls", "F_clu", "F_joint")
PAIRED_METRICS = (*RAW_METRICS, *DERIVED_METRICS)


def metric_values(micro_f1: Any, macro_f1: Any, nmi: Any, ari: Any) -> dict[str, float]:
    values = {
        "Micro_F1": float(micro_f1),
        "Macro_F1": float(macro_f1),
        "NMI": float(nmi),
        "ARI": float(ari),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("metric values must be finite")
    values["F_cls"] = 0.5 * (values["Micro_F1"] + values["Macro_F1"])
    values["F_clu"] = 0.5 * (values["NMI"] + values["ARI"])
    values["F_joint"] = 0.5 * (values["F_cls"] + values["F_clu"])
    return values


def validate_metric_formulas(rows: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-10) -> None:
    for row in rows:
        expected = metric_values(row["Micro_F1"], row["Macro_F1"], row["NMI"], row["ARI"])
        for metric in DERIVED_METRICS:
            observed = float(row[metric])
            if not math.isfinite(observed) or abs(observed - expected[metric]) > tolerance:
                raise ValueError(f"metric formula mismatch: {metric}")


def validate_metric_matrix(rows: Sequence[Mapping[str, Any]], *, require_manual: bool = True) -> dict[str, Any]:
    variants = set(ABLATION_VARIANTS) | ({"MANUAL"} if require_manual else set())
    identities = [
        (str(row.get("dataset", "")).casefold(), str(row.get("ablation_variant", "")), int(row.get("model_seed", -1)))
        for row in rows
    ]
    expected = {(dataset, variant, seed) for dataset in ABLATION_DATASETS for variant in variants for seed in EVALUATION_MODEL_SEEDS}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("M8_F3R_RERUN_2G_FAILED_PROTOCOL: metric dataset/variant/seed matrix is invalid")
    validate_metric_formulas(rows)
    for row in rows:
        if not str(row.get("GPU_UUID", "")) or not str(row.get("runtime_domain", "")) or float(row.get("training_seconds", -1)) < 0:
            raise ValueError("metric row lacks runtime provenance")
        if str(row.get("result_source", "")) not in {"EXACT_REUSE", "NEW_RUN"}:
            raise ValueError("metric row has an invalid result source")
    return {"metric_rows": len(rows), "evaluation_seeds": list(EVALUATION_MODEL_SEEDS), "formula_check": "PASS"}


def recompute_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]).casefold(), str(row["ablation_variant"]))].append(row)
    output = []
    for (dataset, variant), cohort in sorted(grouped.items()):
        if len(cohort) != len(EVALUATION_MODEL_SEEDS) or {int(row["model_seed"]) for row in cohort} != set(EVALUATION_MODEL_SEEDS):
            raise ValueError("summary cohort does not contain the exact evaluation seeds")
        for metric in PAIRED_METRICS:
            values = [float(row[metric]) for row in cohort]
            output.append(
                {
                    "dataset": dataset,
                    "ablation_variant": variant,
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "sample_sd": statistics.stdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "seed_count": len(values),
                }
            )
    return output


def validate_summary(summary_rows: Sequence[Mapping[str, Any]], metric_rows: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-10) -> None:
    expected = {
        (str(row["dataset"]), str(row["ablation_variant"]), str(row["metric"])): row
        for row in recompute_summary(metric_rows)
    }
    observed = {
        (str(row["dataset"]).casefold(), str(row["ablation_variant"]), str(row["metric"])): row
        for row in summary_rows
    }
    if set(observed) != set(expected):
        raise ValueError("summary identities do not match the metric matrix")
    for identity, reference in expected.items():
        for field in ("mean", "sample_sd", "minimum", "maximum", "seed_count"):
            value = float(observed[identity][field])
            if not math.isfinite(value) or abs(value - float(reference[field])) > tolerance:
                raise ValueError(f"summary mismatch: {identity}:{field}")


def recompute_paired_deltas(rows: Sequence[Mapping[str, Any]], *, reference_variant: str) -> list[dict[str, Any]]:
    if reference_variant not in {"FULL_FROZEN", "MANUAL"}:
        raise ValueError("unsupported paired reference")
    indexed = {
        (str(row["dataset"]).casefold(), str(row["ablation_variant"]), int(row["model_seed"])): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("paired metric input contains duplicate identities")
    output = []
    for dataset in ABLATION_DATASETS:
        for variant in ABLATION_VARIANTS:
            if variant == reference_variant:
                continue
            for metric in PAIRED_METRICS:
                deltas = []
                for seed in EVALUATION_MODEL_SEEDS:
                    candidate = indexed.get((dataset, variant, seed))
                    reference = indexed.get((dataset, reference_variant, seed))
                    if candidate is None or reference is None:
                        raise ValueError("paired delta rows are not aligned by the exact seed")
                    deltas.append(float(candidate[metric]) - float(reference[metric]))
                output.append(
                    {
                        "dataset": dataset,
                        "ablation_variant": variant,
                        "reference_variant": reference_variant,
                        "metric": metric,
                        "model_seeds": json.dumps(list(EVALUATION_MODEL_SEEDS), separators=(",", ":")),
                        "paired_deltas": json.dumps(deltas, separators=(",", ":")),
                        "mean_delta": statistics.fmean(deltas),
                        "sample_sd": statistics.stdev(deltas),
                        "non_worse_count": sum(value >= 0 for value in deltas),
                        "minimum_delta": min(deltas),
                        "confirmatory_p_value_claimed": False,
                    }
                )
    return output


def validate_paired_deltas(delta_rows: Sequence[Mapping[str, Any]], metric_rows: Sequence[Mapping[str, Any]], *, reference_variant: str, tolerance: float = 1e-10) -> None:
    expected = {
        (row["dataset"], row["ablation_variant"], row["metric"]): row
        for row in recompute_paired_deltas(metric_rows, reference_variant=reference_variant)
    }
    observed = {
        (str(row["dataset"]).casefold(), str(row["ablation_variant"]), str(row["metric"])): row
        for row in delta_rows
    }
    if set(observed) != set(expected):
        raise ValueError("paired delta identities are incomplete")
    for identity, reference in expected.items():
        row = observed[identity]
        if str(row.get("confirmatory_p_value_claimed", "False")).casefold() not in {"false", "0"}:
            raise ValueError("confirmatory p-value claim is forbidden")
        if [int(value) for value in json.loads(str(row["model_seeds"]))] != list(EVALUATION_MODEL_SEEDS):
            raise ValueError("paired delta seed order changed")
        values = [float(value) for value in json.loads(str(row["paired_deltas"]))]
        expected_values = [float(value) for value in json.loads(str(reference["paired_deltas"]))]
        if len(values) != 3 or any(abs(left - right) > tolerance for left, right in zip(values, expected_values)):
            raise ValueError("paired per-seed values changed")
        for field in ("mean_delta", "sample_sd", "non_worse_count", "minimum_delta"):
            if abs(float(row[field]) - float(reference[field])) > tolerance:
                raise ValueError(f"paired delta summary mismatch: {identity}:{field}")


def metric_rows_for_variants(
    physical_metrics: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project deduplicated physical results back to variant-labelled rows."""
    physical = {
        (str(row["dataset"]).casefold(), str(row["set_hash"]), int(row["model_seed"])): row
        for row in physical_metrics
    }
    if len(physical) != len(physical_metrics):
        raise ValueError("physical metric registry contains duplicate identities")
    output = []
    selections = [dict(row) for row in selection_rows]
    datasets = {
        str(row.get("dataset", "")).casefold() for row in selections
    }
    if not datasets or not datasets <= set(ABLATION_DATASETS):
        raise ValueError("selection projection has an invalid dataset scope")
    for dataset in datasets:
        manual = next((row for row in selections if str(row["dataset"]).casefold() == dataset and str(row["ablation_variant"]) == "MANUAL"), None)
        if manual is None:
            raise ValueError("selection projection lacks Manual identity")
    for selection in selections:
        dataset = str(selection["dataset"]).casefold()
        variant = str(selection["ablation_variant"])
        for seed in EVALUATION_MODEL_SEEDS:
            source = physical.get((dataset, str(selection["set_hash"]), seed))
            if source is None:
                raise ValueError("variant projection lacks a physical metric row")
            values = metric_values(source["Micro_F1"], source["Macro_F1"], source["NMI"], source["ARI"])
            output.append(
                {
                    "dataset": dataset,
                    "ablation_variant": variant,
                    "set_id": str(selection["set_id"]),
                    "set_hash": str(selection["set_hash"]),
                    "model_seed": seed,
                    **values,
                    "assignment_stability": float(source["assignment_stability"]),
                    "GPU_UUID": str(source["GPU_UUID"]),
                    "runtime_domain": str(source["runtime_domain"]),
                    "training_seconds": float(source["training_seconds"]),
                    "result_source": str(source["result_source"]),
                    "evaluated_at_utc": str(source["evaluated_at_utc"]),
                }
            )
    validate_metric_formulas(output)
    return output


def validate_physical_metric_roles(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("physical metric role audit cannot be empty")
    for row in rows:
        dataset = str(row.get("dataset", "")).casefold()
        expected = EVALUATION_CONTRACTS.get(dataset)
        if expected is None or any(
            str(row.get(field, "")) != str(expected[field])
            for field in (
                "classification_fit_role",
                "classification_metric_role",
                "clustering_fit_role",
                "clustering_metric_role",
            )
        ):
            raise ValueError("physical metric row changed the frozen evaluation roles")
        if str(row.get("raw_labels_persisted", "False")).casefold() not in {
            "false",
            "0",
        }:
            raise ValueError("physical metric row reports raw labels persisted")


def validate_existing_safety_evidence(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {"acm", "imdb", "aminer"}
    observed = {str(row.get("dataset", "")).casefold() for row in rows}
    if observed != expected:
        raise ValueError("existing safety evidence must cover ACM, IMDb, and AMiner")
    for row in rows:
        new_training = row.get("new_training")
        if isinstance(new_training, str):
            new_training = new_training.strip().casefold() in {"true", "1", "yes"}
        if new_training is not False or str(row.get("evidence_type", "")) != "EXISTING_FROZEN_EVIDENCE":
            raise ValueError("safety evidence attempted new training or changed evidence type")


def validate_progressive_cost(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {"acm", "freebase", "imdb", "dblp", "aminer"}
    observed = {str(row.get("dataset", "")).casefold() for row in rows}
    if observed != expected:
        raise ValueError("progressive cost must cover all five datasets")
    for row in rows:
        status = str(row.get("cost_status", row.get("status", "")))
        measured = row.get("measured_incremental_gpu_seconds", "")
        if status == "NOT_RECOVERABLE" and str(measured).strip() not in {"", "None", "null"}:
            raise ValueError("NOT_RECOVERABLE progressive cost cannot contain an estimate")
        if status not in {"PASS", "NOT_RECOVERABLE"}:
            raise ValueError("progressive cost status is invalid")


__all__ = [name for name in globals() if not name.startswith("_")]
