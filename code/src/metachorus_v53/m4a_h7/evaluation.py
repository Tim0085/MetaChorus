"""H7 R4 evaluation under the immutable H5 GLOBAL_SEARCH_POOL contract."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file
from metachorus_v53.m4a_h5.evaluation import evaluate_classification, evaluate_clustering
from metachorus_v53.m4a_h5.labels import verify_global_label_role_seal

from .common import envelope, read_csv, write_csv_once, write_json_once
from .constants import (
    EVALUATOR_CONTRACT_SHA256,
    EVALUATOR_SEAL_SHA256,
    EXPECTED_R4_RUNS,
    ROLE_MANIFEST_SHA256,
    ROLE_SEAL_SHA256,
)
from .planning import canonical_plan_row, validate_r4_plan_rows


STAGE_PREFIX = {"R4": "TC_LPMSO_H7_R4"}

_PARENT_EVALUATOR_FILES = {
    "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json": ROLE_MANIFEST_SHA256,
    "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_SEAL.json": ROLE_SEAL_SHA256,
    "TC_LPMSO_H5_SEARCH_EVALUATOR_CONTRACT.yaml": EVALUATOR_CONTRACT_SHA256,
    "TC_LPMSO_H5_SEARCH_EVALUATOR_SEAL.json": EVALUATOR_SEAL_SHA256,
}


def validate_sealed_stage_inputs(
    *, stage: str, plan_path: Path, result_root: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    if stage != "R4" or not rows:
        raise ValueError(stage)
    validate_r4_plan_rows(rows)
    prefix = STAGE_PREFIX[stage]
    plan_seal_path = result_root / f"{prefix}_PLAN_SEAL.json"
    registry_seal_path = result_root / f"{prefix}_REGISTRY_SEAL.json"
    run_registry_path = result_root / f"{prefix}_RUN_REGISTRY.csv"
    checkpoint_registry_path = result_root / f"{prefix}_CHECKPOINT_REGISTRY.csv"
    for path in (plan_seal_path, registry_seal_path, run_registry_path, checkpoint_registry_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing sealed H7 R4 input: {path.name}")
    plan_seal = read_json(plan_seal_path)
    plan_basis = plan_seal.get("basis", {})
    canonical_rows = [canonical_plan_row(row) for row in rows]
    if (
        plan_seal.get("schema_version") != "metachorus.v53.m4a-h7.execution-plan-seal.v1"
        or plan_seal.get("status") != "SEALED_BEFORE_GPU_TRAINING"
        or plan_seal.get("identity") != canonical_json_hash(plan_basis)
        or plan_basis.get("stage") != stage
        or int(plan_basis.get("row_count", -1)) != len(rows)
        or plan_basis.get("rows_identity") != canonical_json_hash(canonical_rows)
        or any(str(row.get("plan_seal_sha256")) != plan_seal.get("identity") for row in rows)
        or plan_basis.get("labels_mounted") is not False
        or plan_basis.get("h5_locked_report_access") is not False
        or plan_basis.get("s_eval_started") is not False
    ):
        raise RuntimeError("H7 R4 execution plan seal mismatch")
    registry_seal = read_json(registry_seal_path)
    registry_basis = registry_seal.get("basis", {})
    expected = {
        "stage": stage,
        "endpoint_update": int(rows[0]["endpoint_update"]),
        "plan_sha256": sha256_file(plan_path),
        "plan_seal_sha256": sha256_file(plan_seal_path),
        "run_registry_sha256": sha256_file(run_registry_path),
        "checkpoint_registry_sha256": sha256_file(checkpoint_registry_path),
        "run_count": len(rows),
        "checkpoint_count": len(rows),
        "all_status_pass": True,
    }
    if (
        registry_seal.get("schema_version")
        != "metachorus.v53.m4a-h7.stage-registry-seal.v1"
        or registry_seal.get("status") != "SEALED_BEFORE_STAGE_EVALUATION"
        or registry_seal.get("identity") != canonical_json_hash(registry_basis)
        or any(registry_basis.get(key) != value for key, value in expected.items())
        or registry_basis.get("labels_mounted") is not False
        or registry_basis.get("h5_locked_report_access") is not False
        or registry_basis.get("s_eval_started") is not False
    ):
        raise RuntimeError("H7 R4 registry seal mismatch")
    run_registry = read_csv(run_registry_path)
    checkpoint_registry = read_csv(checkpoint_registry_path)
    plan_by_id = {str(row["physical_run_id"]): row for row in rows}
    run_by_id = {str(row["physical_run_id"]): row for row in run_registry}
    checkpoint_by_id = {str(row["physical_run_id"]): row for row in checkpoint_registry}
    if (
        len(plan_by_id) != EXPECTED_R4_RUNS
        or len(run_by_id) != EXPECTED_R4_RUNS
        or len(checkpoint_by_id) != EXPECTED_R4_RUNS
        or set(plan_by_id) != set(run_by_id)
        or set(plan_by_id) != set(checkpoint_by_id)
    ):
        raise RuntimeError("H7 R4 plan/registry physical identities do not match")
    identity_fields = (
        "dataset",
        "candidate_set_id",
        "candidate_set_hash",
        "candidate_set_ordinal",
        "model_seed",
        "endpoint_update",
    )
    integer_fields = {"candidate_set_ordinal", "model_seed", "endpoint_update"}
    for run_id, plan_row in plan_by_id.items():
        for registry_row in (run_by_id[run_id], checkpoint_by_id[run_id]):
            for field in identity_fields:
                observed = int(registry_row[field]) if field in integer_fields else str(registry_row[field])
                expected_value = int(plan_row[field]) if field in integer_fields else str(plan_row[field])
                if observed != expected_value:
                    raise RuntimeError(f"H7 R4 registry identity drift: {run_id}/{field}")
            if registry_row.get("status") != "PASS":
                raise RuntimeError(f"H7 R4 registry is not PASS: {run_id}")


def _verify_parent_evaluator(h5_result_root: Path, private_mount_root: Path) -> dict[str, Any]:
    for name, expected in _PARENT_EVALUATOR_FILES.items():
        path = h5_result_root / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise RuntimeError(f"H7 frozen parent evaluator identity mismatch: {name}")
    manifest = verify_global_label_role_seal(
        result_root=h5_result_root,
        private_mount_root=private_mount_root,
    )
    return manifest


def _resolve_representation(row: Mapping[str, Any], *, result_root: Path) -> tuple[Path, str]:
    registry = read_csv(result_root / "TC_LPMSO_H7_R4_CHECKPOINT_REGISTRY.csv")
    matches = [
        item
        for item in registry
        if str(item["physical_run_id"]) == str(row["physical_run_id"])
        and str(item["candidate_set_hash"]) == str(row["candidate_set_hash"])
        and int(item["model_seed"]) == int(row["model_seed"])
        and int(item["endpoint_update"]) == int(row["endpoint_update"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"H7 R4 representation registry mismatch: {row['physical_run_id']}")
    registry_row = matches[0]
    if registry_row.get("source_root") != "H7_RESULT_ROOT":
        raise RuntimeError("H7 evaluation accepts only H7 result-root representations")
    root = result_root.resolve(strict=True)
    path = (result_root / str(registry_row["representation_relative_path"])).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("H7 representation escapes result root") from error
    expected_sha = str(registry_row["representation_sha256"])
    if path.is_symlink() or sha256_file(path) != expected_sha:
        raise RuntimeError(f"H7 R4 representation changed: {row['physical_run_id']}")
    return path, expected_sha


def _evaluate_one(
    row: Mapping[str, Any],
    *,
    result_root: Path,
    private_mount_root: Path,
    split_seeds_by_dataset: Mapping[str, Sequence[int]],
    clustering_seeds: Sequence[int],
    label_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path, representation_sha = _resolve_representation(row, result_root=result_root)
    representation = np.load(path, allow_pickle=False)
    if representation.ndim != 2 or representation.size == 0 or not np.isfinite(representation).all():
        raise RuntimeError(f"H7 R4 representation invalid: {row['physical_run_id']}")
    dataset = str(row["dataset"])
    classification = evaluate_classification(
        representation,
        dataset=dataset,
        private_mount_root=private_mount_root,
        split_seeds=split_seeds_by_dataset[dataset],
        label_manifest=label_manifest,
    )
    clustering_long, clustering_summary = evaluate_clustering(
        representation,
        dataset=dataset,
        private_mount_root=private_mount_root,
        kmeans_seeds=clustering_seeds,
        label_manifest=label_manifest,
    )
    basis = {
        "stage": "R4",
        "physical_run_id": str(row["physical_run_id"]),
        "dataset": dataset,
        "candidate_set_id": str(row["candidate_set_id"]),
        "candidate_set_hash": str(row["candidate_set_hash"]),
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "model_seed": int(row["model_seed"]),
        "update": int(row["endpoint_update"]),
        "representation_sha256": representation_sha,
        "classification": classification,
        "clustering_long": clustering_long,
        "clustering_summary": clustering_summary,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_access": False,
    }
    return envelope("metachorus.v53.m4a-h7.stage-evaluation.v1", "PASS", basis)


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    row, result_root, mount_root, split_seeds, clustering_seeds, manifest = payload
    return _evaluate_one(
        row,
        result_root=Path(result_root),
        private_mount_root=Path(mount_root),
        split_seeds_by_dataset=split_seeds,
        clustering_seeds=clustering_seeds,
        label_manifest=manifest,
    )


def _validate_existing_payload(
    value: Mapping[str, Any], row: Mapping[str, Any], representation_sha256: str
) -> None:
    basis = value.get("basis")
    if (
        value.get("schema_version") != "metachorus.v53.m4a-h7.stage-evaluation.v1"
        or value.get("status") != "PASS"
        or not isinstance(basis, Mapping)
        or value.get("identity") != canonical_json_hash(basis)
    ):
        raise RuntimeError(f"invalid H7 R4 evaluation: {row['physical_run_id']}")
    expected = {
        "stage": "R4",
        "physical_run_id": str(row["physical_run_id"]),
        "dataset": str(row["dataset"]),
        "candidate_set_id": str(row["candidate_set_id"]),
        "candidate_set_hash": str(row["candidate_set_hash"]),
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "model_seed": int(row["model_seed"]),
        "update": int(row["endpoint_update"]),
        "representation_sha256": representation_sha256,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_access": False,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": basis.get(key)}
        for key, expected_value in expected.items()
        if basis.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"stale H7 R4 evaluation: {row['physical_run_id']}: {mismatches}")


def _record_evaluation_failure(
    *, result_root: Path, row: Mapping[str, Any], exc: BaseException
) -> None:
    message_basis = {"failure_kind": type(exc).__name__, "message_hash": canonical_json_hash(str(exc))}
    basis = {
        "stage": "R4_EVALUATION",
        "physical_run_id": str(row["physical_run_id"]),
        "attempt_index": -1,
        "candidate_set_hash": str(row["candidate_set_hash"]),
        "model_seed": int(row["model_seed"]),
        "failure_kind": type(exc).__name__,
        "failure_message_identity": canonical_json_hash(message_basis),
        "root_cause_identity": canonical_json_hash(message_basis),
        "scientific_metrics_persisted": False,
        "engineering_retry_permitted": False,
        "metric_based_retry_forbidden": True,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_access": False,
    }
    failure_root = result_root / "failed_attempts" / "evaluation" / "r4"
    failure_root.mkdir(parents=True, exist_ok=True)
    write_json_once(
        failure_root / f"{row['physical_run_id']}__evaluation_failed.json",
        envelope(
            "metachorus.v53.m4a-h7.stage-evaluation-failed.v1",
            "FAILED_PRESERVED",
            basis,
        ),
    )


def _flatten(payloads: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    classification: list[dict[str, Any]] = []
    clustering_long: list[dict[str, Any]] = []
    clustering_summary: list[dict[str, Any]] = []
    search: list[dict[str, Any]] = []
    for payload in payloads:
        basis = payload["basis"]
        identity = {
            "stage": basis["stage"],
            "physical_run_id": basis["physical_run_id"],
            "dataset": basis["dataset"],
            "candidate_set_id": basis["candidate_set_id"],
            "candidate_set_hash": basis["candidate_set_hash"],
            "candidate_set_ordinal": basis["candidate_set_ordinal"],
            "model_seed": basis["model_seed"],
            "update": basis["update"],
            "representation_sha256": basis["representation_sha256"],
        }
        cls_values: list[float] = []
        for row in basis["classification"]:
            cls_values.append(float(row["F_cls"]))
            classification.append(
                {
                    **identity,
                    "classification_split_index": row["classification_split_index"],
                    "classification_split_seed": row["classification_split_seed"],
                    "micro_f1": row["micro_f1"],
                    "macro_f1": row["macro_f1"],
                    "F_cls": row["F_cls"],
                    "h5_locked_report_access": False,
                    "s_eval_access": False,
                }
            )
        for row in basis["clustering_long"]:
            clustering_long.append(
                {
                    **identity,
                    "kmeans_seed_index": row["kmeans_seed_index"],
                    "kmeans_seed": row["kmeans_seed"],
                    "nmi": row["nmi"],
                    "ari": row["ari"],
                    "F_clu": row["F_clu"],
                    "assignment_sha256": row["assignment_sha256"],
                    "h5_locked_report_access": False,
                    "s_eval_access": False,
                }
            )
        summary = {
            **identity,
            **dict(basis["clustering_summary"]),
            "h5_locked_report_access": False,
            "s_eval_access": False,
        }
        clustering_summary.append(summary)
        search.append(
            {
                **identity,
                "mean_F_cls": mean(cls_values),
                "sample_sd_F_cls": stdev(cls_values),
                "mean_F_clu": summary["mean_F_clu"],
                "sample_sd_F_clu": summary["sample_sd_F_clu"],
                "SE_clu": summary["SE_clu"],
                "LCB_clu": summary["LCB_clu"],
                "assignment_stability_mean": summary["assignment_stability_mean"],
                "assignment_stability_p10": summary["assignment_stability_p10"],
                "h5_locked_report_access": False,
                "s_eval_access": False,
            }
        )
    key = lambda row: (
        str(row["dataset"]),
        int(row["candidate_set_ordinal"]),
        int(row["model_seed"]),
    )
    classification.sort(key=lambda row: (*key(row), int(row["classification_split_index"])))
    clustering_long.sort(key=lambda row: (*key(row), int(row["kmeans_seed_index"])))
    clustering_summary.sort(key=key)
    search.sort(key=key)
    return {
        "classification": classification,
        "clustering_long": clustering_long,
        "clustering_summary": clustering_summary,
        "search": search,
    }


def run_stage_evaluation(
    *,
    plan_path: Path,
    result_root: Path,
    h5_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    rows = read_csv(plan_path)
    if not rows or any(row["stage"] != "R4" for row in rows):
        raise RuntimeError("invalid or empty H7 R4 plan")
    validate_sealed_stage_inputs(stage="R4", plan_path=plan_path, result_root=result_root, rows=rows)
    manifest = _verify_parent_evaluator(h5_result_root, private_mount_root)
    split_seeds = {
        dataset: [
            int(value)
            for value in manifest["seed_bank"]["per_dataset"][dataset][
                "within_pool_classification_split_seeds"
            ]
        ]
        for dataset in ("acm", "freebase")
    }
    clustering_seeds = [int(value) for value in manifest["seed_bank"]["clustering_search_seeds"]]
    if any(len(values) != 5 for values in split_seeds.values()) or len(clustering_seeds) != 20:
        raise RuntimeError("H7 frozen evaluator seed bank drifted")
    job_root = result_root / "stage_evaluation" / "r4"
    job_root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[Mapping[str, Any]] = []
    for row in rows:
        path = job_root / f"{row['physical_run_id']}.json"
        if path.is_file():
            value = read_json(path)
            representation_sha = _resolve_representation(row, result_root=result_root)[1]
            _validate_existing_payload(value, row, representation_sha)
            payloads[str(row["physical_run_id"])] = value
        else:
            missing.append(row)
    if max(1, int(workers)) == 1:
        for row in missing:
            try:
                value = _evaluate_one(
                    row,
                    result_root=result_root,
                    private_mount_root=private_mount_root,
                    split_seeds_by_dataset=split_seeds,
                    clustering_seeds=clustering_seeds,
                    label_manifest=manifest,
                )
            except Exception as exc:
                _record_evaluation_failure(result_root=result_root, row=row, exc=exc)
                raise
            write_json_once(job_root / f"{row['physical_run_id']}.json", value)
            payloads[str(row["physical_run_id"])] = value
    else:
        jobs = [
            (row, str(result_root), str(private_mount_root), split_seeds, clustering_seeds, manifest)
            for row in missing
        ]
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            future_to_row = {executor.submit(_worker, job): job[0] for job in jobs}
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    value = future.result()
                except Exception as exc:
                    _record_evaluation_failure(result_root=result_root, row=row, exc=exc)
                    failures.append(str(row["physical_run_id"]))
                    continue
                write_json_once(job_root / f"{row['physical_run_id']}.json", value)
                payloads[str(row["physical_run_id"])] = value
        if failures:
            raise RuntimeError(f"H7 R4 evaluation failures preserved for {len(failures)} identities")
    if len(payloads) != EXPECTED_R4_RUNS:
        raise RuntimeError(f"H7 R4 evaluation incomplete: {len(payloads)}/{EXPECTED_R4_RUNS}")
    metrics = _flatten(list(payloads.values()))
    prefix = STAGE_PREFIX["R4"]
    destinations = {
        "classification": result_root / f"{prefix}_CLASSIFICATION_METRICS.csv",
        "clustering_long": result_root / f"{prefix}_CLUSTERING_METRICS_LONG.csv",
        "clustering_summary": result_root / f"{prefix}_CLUSTERING_METRICS_SUMMARY.csv",
        "search": result_root / f"{prefix}_SEARCH_METRICS.csv",
    }
    for key, path in destinations.items():
        write_csv_once(path, metrics[key], fieldnames=list(metrics[key][0]))
    return metrics
