"""Locked evaluation and final five-dataset freeze for M8-F1."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.m6f_m7f_fast.evaluation import (
    PreliminaryLabelSession,
    evaluate_preliminary_representation,
)
from metachorus_v53.m6f_m7f_fast.protocol import METRICS

from .common import (
    canonical_json_hash,
    envelope,
    read_csv,
    read_json,
    sha256_file,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    CONFIRMATION_MODEL_SEEDS,
    DATASETS,
    DATASET_SPECS,
    FINAL10_ENDPOINT,
    FINAL_METHOD_ROLES,
    FROZEN_SEED_NAMESPACE_IDENTITY,
    KMEANS_SEEDS,
)
from .planning import canonical_plan_row, validate_plan
from .training import completed_marker, validate_completion_marker


T_CRITICAL_DF9_95 = 2.2621571628540993
FINAL_METRICS = tuple(METRICS)


def metric_values(micro: float, macro: float, nmi: float, ari: float) -> dict[str, float]:
    f_cls = 0.5 * (float(micro) + float(macro))
    f_clu = 0.5 * (float(nmi) + float(ari))
    return {
        "Micro_F1": float(micro),
        "Macro_F1": float(macro),
        "NMI": float(nmi),
        "ARI": float(ari),
        "F_cls": f_cls,
        "F_clu": f_clu,
        "F_joint": 0.5 * (f_cls + f_clu),
    }


def _governed_file(result: Path, relative: str) -> Path:
    if "\\" in relative or not relative:
        raise ValueError("governed relative path is unsafe")
    candidate = result / Path(*relative.split("/"))
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("governed evidence is not a regular file")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(result)
    except ValueError as exc:
        raise ValueError("governed evidence escapes the result root") from exc
    return resolved


def _load_representation(result: Path, basis: Mapping[str, Any]) -> np.ndarray:
    dataset = str(basis["dataset"])
    spec = DATASET_SPECS[dataset]
    path = _governed_file(result, str(basis["representation_relative_path"]))
    if sha256_file(path) != str(basis["representation_sha256"]):
        raise ValueError("M8-F1 representation hash changed")
    values = np.load(path, allow_pickle=False)
    if (
        values.ndim != 2
        or values.shape[0] != int(spec["target_node_count"])
        or not np.isfinite(values).all()
    ):
        raise ValueError("M8-F1 representation shape/finiteness is invalid")
    return values


def close_canaries(
    *, result_root: str | Path, rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    plan = validate_plan(rows, seal, logical_stage="CANARY")
    registry: list[dict[str, Any]] = []
    for row in plan:
        completed = completed_marker(result, row, "CANARY")
        if completed is None:
            raise RuntimeError("M8-F1 canary matrix is incomplete")
        basis = validate_completion_marker(completed[1], row, logical_stage="CANARY")
        audit = basis.get("canary_engineering_audit", {})
        if (
            audit.get("status") != "PASS"
            or audit.get("checkpoint_write_pass") is not True
            or audit.get("checkpoint_read_pass") is not True
            or audit.get("checkpoint_state_roundtrip") is not True
            or audit.get("labels_mounted") is not False
            or int(audit.get("downstream_evaluations_during_training", -1)) != 0
        ):
            raise RuntimeError("M8-F1 canary checkpoint/label boundary failed")
        registry.append(basis)
    expected_mapping = {
        ("imdb", int(CONFIRMATION_MODEL_SEEDS[0]), 0),
        ("aminer", int(CONFIRMATION_MODEL_SEEDS[0]), 1),
        ("imdb", int(CONFIRMATION_MODEL_SEEDS[1]), 2),
        ("aminer", int(CONFIRMATION_MODEL_SEEDS[1]), 3),
    }
    observed_mapping = {
        (str(row["dataset"]), int(row["model_seed"]), int(row["worker_index"]))
        for row in registry
    }
    if (
        len(registry) != 4
        or len({row["gpu_uuid"] for row in registry}) != 4
        or observed_mapping != expected_mapping
    ):
        raise RuntimeError("M8-F1 canaries did not cover the frozen four-GPU mapping")
    value = envelope(
        "metachorus.v53.m8-f1.canary-results.v1",
        "PASS",
        {
            "canary_plan_seal_identity": seal["identity"],
            "run_count": 4,
            "run_registry_identity": canonical_json_hash(registry),
            "datasets": list(DATASETS),
            "gpu_uuids": sorted(str(row["gpu_uuid"]) for row in registry),
            "all_checkpoint_roundtrip_pass": True,
            "all_labels_mounted": False,
            "all_downstream_evaluations_during_training": 0,
            "scientific_results_allowed": False,
            "formal_final10_restarts_from_zero": True,
            "fanout_authorized": True,
        },
    )
    write_csv_once(result / "M8_F1_CANARY_RUN_REGISTRY.csv", registry)
    write_json_once(result / "M8_F1_CANARY_RESULTS.json", value)
    return value


def seal_final10_training_registry(
    *, result_root: str | Path, rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve()
    plan = validate_plan(rows, seal, logical_stage="FINAL10")
    registry: list[dict[str, Any]] = []
    for row in plan:
        completed = completed_marker(result, row, "FINAL10")
        if completed is None:
            raise RuntimeError("M8-F1 FINAL10 matrix is incomplete")
        basis = validate_completion_marker(completed[1], row, logical_stage="FINAL10")
        marker_path = completed[0]
        registry.append(
            {
                **basis,
                "completion_marker_relative_path": marker_path.relative_to(result).as_posix(),
                "completion_marker_sha256": sha256_file(marker_path),
                "status": "PASS",
            }
        )
    expected_pairs = {
        (dataset, int(seed)) for dataset in DATASETS for seed in CONFIRMATION_MODEL_SEEDS
    }
    observed_pairs = {(str(row["dataset"]), int(row["model_seed"])) for row in registry}
    if (
        len(registry) != 20
        or len({row["physical_run_id"] for row in registry}) != 20
        or observed_pairs != expected_pairs
        or any(int(row["optimizer_steps"]) != FINAL10_ENDPOINT for row in registry)
        or any(row["labels_mounted_during_training"] is not False for row in registry)
    ):
        raise RuntimeError("M8-F1 training registry is not the exact 20-run matrix")
    seal_basis = {
        "final10_matrix_seal_identity": seal["identity"],
        "active_datasets": list(DATASETS),
        "physical_run_count": len(registry),
        "model_seed_count_per_dataset": 10,
        "registry_identity": canonical_json_hash(registry),
        "all_unique_runs_trained_fresh_from_zero": True,
        "all_status_pass": True,
        "labels_mounted": False,
        "preliminary_evaluation_started": False,
        "search_checkpoint_reused": False,
    }
    registry_seal = envelope(
        "metachorus.v53.m8-f1.final10-training-registry-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        seal_basis,
    )
    write_csv_once(result / "M8_F1_RUN_REGISTRY_MINIMAL.csv", registry)
    write_json_once(result / "M8_F1_FINAL10_TRAINING_REGISTRY_SEAL.json", registry_seal)
    return registry, registry_seal


def summarize_final10(
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for dataset in DATASETS:
        cohort = [row for row in metrics if str(row["dataset"]) == dataset]
        if (
            len(cohort) != 10
            or {int(row["model_seed"]) for row in cohort} != set(CONFIRMATION_MODEL_SEEDS)
        ):
            raise ValueError(f"M8-F1 {dataset} summary requires exact ten seeds")
        summary: dict[str, Any] = {
            "dataset": dataset,
            "final_method_role": FINAL_METHOD_ROLES[dataset],
            "set_id": DATASET_SPECS[dataset]["manual_set_id"],
            "set_hash": DATASET_SPECS[dataset]["manual_set_hash"],
            "model_seed_count": 10,
            "status": "PASS",
        }
        for metric in FINAL_METRICS:
            values = [float(row[metric]) for row in cohort]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("M8-F1 final metric is not finite")
            mean = statistics.mean(values)
            sample_sd = statistics.stdev(values)
            se = sample_sd / math.sqrt(len(values))
            lower = mean - T_CRITICAL_DF9_95 * se
            upper = mean + T_CRITICAL_DF9_95 * se
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sample_sd"] = sample_sd
            intervals.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "model_seed_count": 10,
                    "mean": mean,
                    "sample_sd": sample_sd,
                    "se": se,
                    "t_critical_df9_95": T_CRITICAL_DF9_95,
                    "lower_95": lower,
                    "upper_95": upper,
                    "minimum": min(values),
                    "maximum": max(values),
                    "selection_reopened": False,
                    "status": "DESCRIPTIVE_ONLY",
                }
            )
        summaries.append(summary)
    return summaries, intervals


def build_prior3_descriptive_audit(
    final10_summary: Sequence[Mapping[str, Any]], prior3_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    summary = {str(row["dataset"]): row for row in final10_summary}
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        cohort = [row for row in prior3_rows if str(row.get("dataset")) == dataset]
        if len(cohort) != 3 or len({int(row["model_seed"]) for row in cohort}) != 3:
            raise ValueError(f"M8-F1 {dataset} prior3 evidence is not exact")
        for metric in FINAL_METRICS:
            prior_values = [float(row[metric]) for row in cohort]
            prior_mean = statistics.mean(prior_values)
            prior_sd = statistics.stdev(prior_values)
            final_mean = float(summary[dataset][f"{metric}_mean"])
            final_sd = float(summary[dataset][f"{metric}_sample_sd"])
            delta = final_mean - prior_mean
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "final10_model_seed_count": 10,
                    "prior_model_seed_count": 3,
                    "final10_mean": final_mean,
                    "prior3_mean": prior_mean,
                    "final10_mean_minus_prior3_mean": delta,
                    "final10_sample_sd": final_sd,
                    "prior3_sample_sd": prior_sd,
                    "direction": "UP" if delta > 0 else "DOWN" if delta < 0 else "UNCHANGED",
                    "direction_consistent": (final_mean >= 0.0) == (prior_mean >= 0.0),
                    "combined_13seed_mean_forbidden": True,
                    "paired_sample_interpretation_forbidden": True,
                    "method_change_allowed": False,
                    "evidence_level": "DESCRIPTIVE_PRIOR3_ONLY",
                    "status": "PASS",
                }
            )
    return output


def build_rejected_search_set_ablation(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        spec = DATASET_SPECS[dataset]
        matches = [
            row
            for row in rows
            if str(row.get("dataset")) == dataset
            and row.get("candidate_set_hash") == spec["rejected_set_hash"]
        ]
        if len(matches) != 1 or int(matches[0].get("model_seed_count", -1)) != 3:
            raise ValueError(f"M8-F1 {dataset} rejected-set public evidence is not exact three-seed evidence")
        row = matches[0]
        values = {
            metric: float(row.get(metric, row.get(f"{metric}_mean", float("nan"))))
            for metric in FINAL_METRICS
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"M8-F1 {dataset} rejected-set public metric is not finite")
        output.append(
            {
                "dataset": dataset,
                "candidate_set_id": spec["rejected_set_id"],
                "candidate_set_hash": spec["rejected_set_hash"],
                "model_seed_count": 3,
                **values,
                "final_method_role": FINAL_METHOD_ROLES[dataset],
                "ablation_evidence_level": "3_SEED_DEVELOPMENT_OR_PRELIMINARY_ABLATION",
                "final_evidence_eligibility": "NOT_10_SEED_FINAL_EVIDENCE",
                "new_gpu_runs": 0,
                "method_change_allowed": False,
                "status": "REJECTED_SET_ABLATION_ONLY",
            }
        )
    return output


def _markdown_table(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    if not rows:
        return f"# {title}\n"
    fields = list(rows[0])
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def build_five_dataset_tables(
    *,
    parent_rows: Sequence[Mapping[str, Any]],
    parent_stats: Sequence[Mapping[str, Any]],
    final10_summary: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]] | str]:
    parent = {str(row["dataset"]): dict(row) for row in parent_rows}
    stats = {str(row["dataset"]): dict(row) for row in parent_stats}
    new = {str(row["dataset"]): dict(row) for row in final10_summary}
    expected = {"acm", "freebase", "imdb", "dblp", "aminer"}
    if set(parent) != expected or set(new) != set(DATASETS):
        raise ValueError("M8-F1 five-dataset source table is incomplete")
    if not {"acm", "freebase", "dblp"}.issubset(stats):
        raise ValueError("M8-F1 parent ten-seed sample SD evidence is incomplete")

    status_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    supplementary_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase", "imdb", "dblp", "aminer"):
        source = parent[dataset]
        role = FINAL_METHOD_ROLES[dataset]
        if dataset in DATASETS:
            row = new[dataset]
            mean = {metric: float(row[f"{metric}_mean"]) for metric in FINAL_METRICS}
            sd = {metric: float(row[f"{metric}_sample_sd"]) for metric in FINAL_METRICS}
            final_set_id = DATASET_SPECS[dataset]["manual_set_id"]
            final_set_hash = DATASET_SPECS[dataset]["manual_set_hash"]
            evidence = "10_SEED_FINAL_FIXED_METHOD_INTERNAL_REPORT"
        else:
            row = stats[dataset]
            mean = {metric: float(source[metric]) for metric in FINAL_METRICS}
            sd = {metric: float(row[f"{metric}_sample_sd"]) for metric in FINAL_METRICS}
            final_set_id = source["final_set_id"]
            final_set_hash = source["final_set_hash"]
            evidence = str(source.get("confirmation_evidence", "10_SEED_PARENT_FROZEN"))
        source_seed_count = int(source.get("model_seed_count", -1))
        if (
            (dataset in DATASETS and source_seed_count not in {3, 10})
            or (dataset not in DATASETS and source_seed_count != 10)
        ):
            raise ValueError("M8-F1 parent seed count is invalid")
        claim = (
            "BUDGET_LIMITED_SEARCH;NO_FULL_THREE_SEED_SEARCH_SPACE_OPTIMUM_CLAIM"
            if dataset == "dblp"
            else "FINAL_10_SEED_FIXED_METHOD_RESULT"
        )
        status_rows.append(
            {
                "dataset": dataset,
                "final_method_role": role,
                "final_set_id": final_set_id,
                "final_set_hash": final_set_hash,
                **mean,
                **{f"{metric}_sample_sd": sd[metric] for metric in FINAL_METRICS},
                "model_seed_count": 10,
                "evidence_level": evidence,
                "claim_boundary": claim,
                "status": "FINAL_10SEED_FROZEN",
            }
        )
        main_rows.append(
            {
                "dataset": dataset,
                "final_method_role": role,
                "model_seed_count": 10,
                **{
                    f"{metric}_mean": mean[metric]
                    for metric in ("Micro_F1", "Macro_F1", "NMI", "ARI")
                },
                **{
                    f"{metric}_sample_sd": sd[metric]
                    for metric in ("Micro_F1", "Macro_F1", "NMI", "ARI")
                },
                **{
                    f"{metric}_mean_plus_minus_sd": f"{mean[metric]:.6f} ± {sd[metric]:.6f}"
                    for metric in ("Micro_F1", "Macro_F1", "NMI", "ARI")
                },
            }
        )
        supplementary_rows.append(
            {
                "dataset": dataset,
                "final_method_role": role,
                "model_seed_count": 10,
                **{f"{metric}_mean": mean[metric] for metric in ("F_cls", "F_clu", "F_joint")},
                **{
                    f"{metric}_sample_sd": sd[metric]
                    for metric in ("F_cls", "F_clu", "F_joint")
                },
            }
        )
        path_rows.append(
            {
                "dataset": dataset,
                "final_method_role": role,
                "final_set_id": final_set_id,
                "final_set_hash": final_set_hash,
                "path_ids": (
                    list(DATASET_SPECS[dataset]["manual_path_ids"])
                    if dataset in DATASETS
                    else source.get("path_ids", "PARENT_FROZEN")
                ),
                "status": "FROZEN",
            }
        )
        evidence_rows.append(
            {
                "dataset": dataset,
                "model_seed_count": 10,
                "evidence_level": evidence,
                "final_method_role": role,
                "claim_boundary": claim,
            }
        )
    if any(row["model_seed_count"] != 10 for row in status_rows):
        raise RuntimeError("M8-F1 did not close all five datasets at ten seeds")
    claim_md = (
        "# Final Claim Boundary\n\n"
        "- All five datasets have frozen 10-model-seed final results.\n"
        "- IMDb and AMiner remain Manual fallbacks; their roles were not reopened.\n"
        "- DBLP remains BUDGET_LIMITED_SEARCH.\n"
        "- DBLP has NO_FULL_THREE_SEED_SEARCH_SPACE_OPTIMUM_CLAIM.\n"
        "- No additional search, retuning, stopping-point selection, or model seeds are authorized.\n"
    )
    return {
        "status": status_rows,
        "main": main_rows,
        "supplementary": supplementary_rows,
        "paths": path_rows,
        "evidence": evidence_rows,
        "claim_markdown": claim_md,
    }


def evaluate_locked_final10(
    *,
    result_root: str | Path,
    role_manifest_path: str | Path,
    seed_manifest_path: str | Path,
    private_search_mount_root: str | Path,
    label_sources: Mapping[str, str | Path],
    label_source_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    rows = read_csv(result / "FINAL10_COMPLETION_MATRIX.csv")
    seal = read_json(result / "FINAL10_COMPLETION_MATRIX_SEAL.json")
    registry, registry_seal = seal_final10_training_registry(
        result_root=result, rows=rows, seal=seal
    )
    seed_manifest = read_json(seed_manifest_path)
    seed_basis = seed_manifest.get("basis", {})
    if (
        seed_manifest.get("identity") != FROZEN_SEED_NAMESPACE_IDENTITY
        or seed_manifest.get("identity") != canonical_json_hash(seed_basis)
        or tuple(map(int, seed_basis.get("preliminary_kmeans_seeds", []))) != tuple(KMEANS_SEEDS)
        or len(KMEANS_SEEDS) != 20
    ):
        raise ValueError("M8-F1 KMeans seed manifest identity changed")
    session = PreliminaryLabelSession(
        role_manifest_path=role_manifest_path,
        private_search_mount_root=private_search_mount_root,
        label_sources=label_sources,
        label_source_roots=label_source_roots,
        training_registry_seal=registry_seal,
    )
    roles = {dataset: session.open_dataset(dataset) for dataset in DATASETS}
    plan_by_id = {str(row["physical_run_id"]): canonical_plan_row(row) for row in rows}
    metrics: list[dict[str, Any]] = []
    classification: list[dict[str, Any]] = []
    clustering_long: list[dict[str, Any]] = []
    clustering_summary: list[dict[str, Any]] = []
    for registry_row in registry:
        dataset = str(registry_row["dataset"])
        plan_row = plan_by_id[str(registry_row["physical_run_id"])]
        evaluated = evaluate_preliminary_representation(
            _load_representation(result, registry_row), dataset=dataset, roles=roles[dataset]
        )
        recomputed = metric_values(
            evaluated["metrics"]["Micro_F1"],
            evaluated["metrics"]["Macro_F1"],
            evaluated["metrics"]["NMI"],
            evaluated["metrics"]["ARI"],
        )
        if any(abs(recomputed[key] - float(evaluated["metrics"][key])) > 1e-12 for key in FINAL_METRICS):
            raise RuntimeError("M8-F1 metric formula changed")
        metric_row = {
            "dataset": dataset,
            "final_method_role": FINAL_METHOD_ROLES[dataset],
            "physical_run_id": plan_row["physical_run_id"],
            "set_id": plan_row["candidate_set_id"],
            "set_hash": plan_row["candidate_set_hash"],
            "model_seed": int(plan_row["model_seed"]),
            **recomputed,
            "assignment_stability": evaluated["assignment_stability_mean"],
            "assignment_stability_p10": evaluated["assignment_stability_p10"],
            "GPU_UUID": registry_row["gpu_uuid"],
            "gpu_uuid": registry_row["gpu_uuid"],
            "gpu_pci_bus_id": registry_row["gpu_pci_bus_id"],
            "runtime_domain": registry_row["runtime_domain"],
            "training_seconds": float(registry_row["invocation_wall_seconds"]),
            "classification_fit_role": evaluated["classification_fit_role"],
            "classification_metric_role": evaluated["classification_metric_role"],
            "clustering_fit_role": evaluated["clustering_fit_role"],
            "clustering_metric_role": evaluated["clustering_metric_role"],
            "labels_mounted_during_training": False,
            "status": "PASS",
        }
        metrics.append(metric_row)
        classification.append(
            {
                key: metric_row[key]
                for key in (
                    "dataset", "final_method_role", "physical_run_id", "set_id", "set_hash",
                    "model_seed", "Micro_F1", "Macro_F1", "F_cls", "classification_fit_role",
                    "classification_metric_role", "gpu_uuid", "runtime_domain", "status",
                )
            }
        )
        clustering_summary.append(
            {
                key: metric_row[key]
                for key in (
                    "dataset", "final_method_role", "physical_run_id", "set_id", "set_hash",
                    "model_seed", "NMI", "ARI", "F_clu", "assignment_stability",
                    "assignment_stability_p10", "clustering_fit_role", "clustering_metric_role",
                    "gpu_uuid", "runtime_domain", "status",
                )
            }
        )
        for repeat in evaluated["clustering_repeats"]:
            clustering_long.append(
                {
                    "dataset": dataset,
                    "physical_run_id": plan_row["physical_run_id"],
                    "set_hash": plan_row["candidate_set_hash"],
                    "model_seed": int(plan_row["model_seed"]),
                    **repeat,
                }
            )
    if (
        session.access_count != 2
        or any(session.per_dataset_access_count[dataset] != 1 for dataset in DATASETS)
        or len(metrics) != 20
        or len(clustering_long) != 20 * len(KMEANS_SEEDS)
    ):
        raise RuntimeError("M8-F1 locked evaluation matrix/access count is incomplete")

    summaries, intervals = summarize_final10(metrics)
    prior3 = build_prior3_descriptive_audit(
        summaries, read_csv(result / "M8_F1_PRIOR3_METRICS_BY_SEED.csv")
    )
    ablation = build_rejected_search_set_ablation(
        read_csv(result / "M8_F1_REJECTED_SET_EXISTING_METRICS.csv")
    )
    tables = build_five_dataset_tables(
        parent_rows=read_csv(result / "FIVE_DATASET_STATUS_TABLE_PRE_FINAL10.csv"),
        parent_stats=read_csv(result / "M8_F1_PARENT_FINAL_METRIC_STATS.csv"),
        final10_summary=summaries,
    )
    final_freeze = envelope(
        "metachorus.v53.m8-f1.final-method-freeze.v1",
        "FROZEN_AFTER_FINAL10",
        {
            "training_registry_seal_identity": registry_seal["identity"],
            "final10_metrics_identity": canonical_json_hash(metrics),
            "final10_summary_identity": canonical_json_hash(summaries),
            "five_dataset_status_identity": canonical_json_hash(tables["status"]),
            "final_method_roles": FINAL_METHOD_ROLES,
            "all_five_datasets_have_10_model_seed_final_results": True,
            "search_reopened": False,
            "second_choice_used": False,
            "algorithm_retuning": False,
            "additional_path_search": False,
            "additional_model_seed_runs": False,
            "core_model_experiments_complete": True,
        },
    )
    write_csv_once(result / "FINAL10_LOCKED_CLASSIFICATION_METRICS.csv", classification)
    write_csv_once(result / "FINAL10_LOCKED_CLUSTERING_METRICS_LONG.csv", clustering_long)
    write_csv_once(result / "FINAL10_LOCKED_CLUSTERING_METRICS_SUMMARY.csv", clustering_summary)
    write_csv_once(result / "FINAL10_METRICS_BY_MODEL_SEED.csv", metrics)
    write_csv_once(result / "FINAL10_SUMMARY.csv", summaries)
    write_csv_once(result / "FINAL10_INTERVALS.csv", intervals)
    write_csv_once(result / "FINAL10_VS_PRIOR3_DESCRIPTIVE_AUDIT.csv", prior3)
    write_csv_once(result / "REJECTED_SEARCH_SET_ABLATION_TABLE.csv", ablation)
    write_text_once(
        result / "REJECTED_SEARCH_SET_ABLATION_TABLE.md",
        _markdown_table(ablation, "Rejected Search-Set Ablation (Existing Three-Seed Evidence Only)"),
    )
    write_csv_once(result / "FIVE_DATASET_FINAL_10SEED_STATUS_TABLE.csv", tables["status"])
    write_text_once(
        result / "FIVE_DATASET_FINAL_10SEED_STATUS_TABLE.md",
        _markdown_table(tables["status"], "Five-Dataset Final 10-Seed Status"),
    )
    write_csv_once(result / "FIVE_DATASET_PAPER_MAIN_TABLE.csv", tables["main"])
    write_text_once(
        result / "FIVE_DATASET_PAPER_MAIN_TABLE.md",
        _markdown_table(tables["main"], "Five-Dataset Paper Main Table"),
    )
    write_csv_once(
        result / "FIVE_DATASET_SUPPLEMENTARY_COMPOSITE_TABLE.csv",
        tables["supplementary"],
    )
    write_csv_once(result / "FINAL_PATH_SET_REGISTRY_5DATASETS.csv", tables["paths"])
    write_csv_once(result / "FINAL_EVIDENCE_LEVEL_TABLE.csv", tables["evidence"])
    write_text_once(result / "FINAL_CLAIM_BOUNDARY.md", str(tables["claim_markdown"]))
    write_json_once(result / "M8_F1_FINAL_METHOD_FREEZE.json", final_freeze)
    write_json_once(
        result / "M8_F1_LOCKED_EVALUATION_ACCESS_AUDIT.json",
        envelope(
            "metachorus.v53.m8-f1.locked-evaluation-access-audit.v1",
            "PASS",
            {
                "training_registry_seal_identity": registry_seal["identity"],
                "label_source_access_count": session.access_count,
                "per_dataset_access_count": {
                    dataset: session.per_dataset_access_count[dataset] for dataset in DATASETS
                },
                "classification_fit_role": "GLOBAL_SEARCH_POOL",
                "classification_metric_role": "PRELIM_LOCKED_REPORT",
                "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
                "clustering_metric_role": "PRELIM_LOCKED_REPORT",
                "raw_labels_persisted": False,
            },
        ),
    )
    return final_freeze


__all__ = [
    "metric_values",
    "close_canaries",
    "seal_final10_training_registry",
    "summarize_final10",
    "build_prior3_descriptive_audit",
    "build_rejected_search_set_ablation",
    "build_five_dataset_tables",
    "evaluate_locked_final10",
]
