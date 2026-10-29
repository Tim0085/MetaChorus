"""Resumable H5 R2/R3/third-seed/R4 search-pool evaluation."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import atomic_write_csv, atomic_write_json, canonical_json_hash, read_json, sha256_file

from .evaluation import evaluate_classification, evaluate_clustering
from .labels import verify_global_label_role_seal


STAGE_ARTIFACT_PREFIX = {
    "R2": "TC_LPMSO_H5_R2",
    "R3": "TC_LPMSO_H5_R3",
    "THIRD_SEED_R3_AUDIT": "TC_LPMSO_H5_THIRD_SEED",
    "R4": "TC_LPMSO_H5_R4",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("plan_seal_sha256", None)
    for field in (
        "candidate_set_ordinal",
        "model_seed",
        "continuation_from_update",
        "endpoint_update",
        "gpu_preference",
        "attempt_index",
    ):
        value[field] = int(value[field])
    value["labels_mounted"] = str(value["labels_mounted"]).lower() == "true"
    return value


def _validate_sealed_stage_inputs(
    *, stage: str, plan_path: Path, result_root: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    prefix = STAGE_ARTIFACT_PREFIX[stage]
    plan_seal_path = result_root / f"{prefix}_PLAN_SEAL.json"
    registry_seal_path = result_root / f"{prefix}_REGISTRY_SEAL.json"
    run_registry_path = result_root / f"{prefix}_RUN_REGISTRY.csv"
    checkpoint_registry_path = result_root / f"{prefix}_CHECKPOINT_REGISTRY.csv"
    for path in (
        plan_seal_path,
        registry_seal_path,
        run_registry_path,
        checkpoint_registry_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"missing sealed H5 stage input: {path.name}")
    plan_seal = read_json(plan_seal_path)
    plan_basis = plan_seal.get("basis", {})
    canonical_rows = [_canonical_plan_row(row) for row in rows]
    if (
        plan_seal.get("schema_version")
        != "metachorus.v53.m4a-h5.execution-plan-seal.v1"
        or plan_seal.get("status") != "SEALED_BEFORE_GPU_TRAINING"
        or plan_seal.get("identity") != canonical_json_hash(plan_basis)
        or plan_basis.get("stage") != stage
        or int(plan_basis.get("row_count", -1)) != len(rows)
        or plan_basis.get("rows_identity") != canonical_json_hash(canonical_rows)
        or any(str(row.get("plan_seal_sha256")) != plan_seal.get("identity") for row in rows)
    ):
        raise RuntimeError(f"H5 {stage} execution plan seal mismatch")
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
        != "metachorus.v53.m4a-h5.stage-registry-seal.v1"
        or registry_seal.get("status") != "SEALED_BEFORE_STAGE_EVALUATION"
        or registry_seal.get("identity") != canonical_json_hash(registry_basis)
        or any(registry_basis.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError(f"H5 {stage} registry seal mismatch")


def _resolve_representation(
    row: Mapping[str, Any], *, result_root: Path, h4_result_root: Path
) -> tuple[Path, str]:
    stage = str(row["stage"])
    endpoint = int(row["endpoint_update"])
    prefix = STAGE_ARTIFACT_PREFIX[stage]
    registry_path = result_root / f"{prefix}_CHECKPOINT_REGISTRY.csv"
    if not registry_path.is_file():
        raise RuntimeError(
            f"H5 {stage} checkpoint registry must be sealed before evaluation"
        )
    matches = [
        item
        for item in _read_csv(registry_path)
        if str(item["physical_run_id"]) == str(row["physical_run_id"])
        and str(item["candidate_set_hash"]) == str(row["candidate_set_hash"])
        and int(item["model_seed"]) == int(row["model_seed"])
        and int(item["endpoint_update"]) == endpoint
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"H5 {stage} representation registry identity mismatch: "
            f"{row['physical_run_id']} ({len(matches)} matches)"
        )
    registry = matches[0]
    source_root = str(registry["source_root"])
    if source_root == "H4_PARENT_IMMUTABLE":
        path = h4_result_root / str(registry["representation_relative_path"])
    elif source_root == "H5_RESULT_ROOT":
        path = result_root / str(registry["representation_relative_path"])
    else:
        raise RuntimeError(f"unsupported H5 representation source root: {source_root}")
    expected_sha = str(registry["representation_sha256"])
    if stage == "R2" and str(row["execution_action"]) == "EVALUATION_ONLY_REUSE":
        if (
            str(registry["representation_relative_path"])
            != str(row["parent_representation_relative_path"])
            or expected_sha != str(row["parent_representation_sha256"])
        ):
            raise RuntimeError(
                f"H5 R2 reuse registry disagrees with the sealed plan: {row['physical_run_id']}"
            )
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise RuntimeError(f"H5 stage representation missing or changed: {row['physical_run_id']}")
    return path, expected_sha


def _evaluate_one(
    row: Mapping[str, Any],
    *,
    result_root: Path,
    h4_result_root: Path,
    private_mount_root: Path,
    split_seeds_by_dataset: Mapping[str, Sequence[int]],
    clustering_seeds: Sequence[int],
    label_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path, representation_sha = _resolve_representation(
        row, result_root=result_root, h4_result_root=h4_result_root
    )
    representation = np.load(path, allow_pickle=False)
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
        "stage": row["stage"],
        "physical_run_id": row["physical_run_id"],
        "dataset": dataset,
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "model_seed": int(row["model_seed"]),
        "update": int(row["endpoint_update"]),
        "representation_sha256": representation_sha,
        "classification": classification,
        "clustering_long": clustering_long,
        "clustering_summary": clustering_summary,
        "h5_locked_report_access": False,
    }
    return {
        "schema_version": "metachorus.v53.m4a-h5.stage-evaluation.v1",
        "status": "PASS",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }


def _worker(payload):
    row, result_root, h4_root, mount_root, split_seeds, clustering_seeds, label_manifest = payload
    return _evaluate_one(
        row,
        result_root=Path(result_root),
        h4_result_root=Path(h4_root),
        private_mount_root=Path(mount_root),
        split_seeds_by_dataset=split_seeds,
        clustering_seeds=clustering_seeds,
        label_manifest=label_manifest,
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        observed = read_json(path)
        if observed.get("identity") != value.get("identity"):
            raise RuntimeError(f"append-only stage evaluation conflict: {path}")
        return
    atomic_write_json(path, dict(value))


def _write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to publish an empty H5 metric table: {path.name}")
    fieldnames = list(rows[0])
    if path.is_file():
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        buffer.seek(0)
        expected = [dict(row) for row in csv.DictReader(buffer)]
        observed = _read_csv(path)
        if observed != expected:
            raise RuntimeError(f"append-only H5 metric table conflict: {path.name}")
        return
    atomic_write_csv(path, rows, fieldnames=fieldnames)


def _validate_existing_payload(
    value: Mapping[str, Any], row: Mapping[str, Any], representation_sha256: str
) -> None:
    basis = value.get("basis")
    if (
        value.get("schema_version") != "metachorus.v53.m4a-h5.stage-evaluation.v1"
        or value.get("status") != "PASS"
        or not isinstance(basis, Mapping)
        or value.get("identity") != canonical_json_hash(basis)
    ):
        raise RuntimeError(f"invalid H5 stage-evaluation payload: {row['physical_run_id']}")
    expected = {
        "stage": str(row["stage"]),
        "physical_run_id": str(row["physical_run_id"]),
        "dataset": str(row["dataset"]),
        "candidate_set_id": str(row["candidate_set_id"]),
        "candidate_set_hash": str(row["candidate_set_hash"]),
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "model_seed": int(row["model_seed"]),
        "update": int(row["endpoint_update"]),
        "representation_sha256": representation_sha256,
        "h5_locked_report_access": False,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": basis.get(key)}
        for key, expected_value in expected.items()
        if basis.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            f"stale or mismatched H5 stage evaluation {row['physical_run_id']}: {mismatches}"
        )


def _flatten(payloads: Sequence[Mapping[str, Any]]):
    classification_rows: list[dict[str, Any]] = []
    clustering_rows: list[dict[str, Any]] = []
    clustering_summary_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
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
        cls_values = []
        for row in basis["classification"]:
            cls_values.append(float(row["F_cls"]))
            classification_rows.append(
                {
                    **identity,
                    "classification_split_index": row["classification_split_index"],
                    "classification_split_seed": row["classification_split_seed"],
                    "micro_f1": row["micro_f1"],
                    "macro_f1": row["macro_f1"],
                    "F_cls": row["F_cls"],
                    "h5_locked_report_access": False,
                }
            )
        for row in basis["clustering_long"]:
            clustering_rows.append(
                {
                    **identity,
                    "kmeans_seed_index": row["kmeans_seed_index"],
                    "kmeans_seed": row["kmeans_seed"],
                    "nmi": row["nmi"],
                    "ari": row["ari"],
                    "F_clu": row["F_clu"],
                    "assignment_sha256": row["assignment_sha256"],
                    "h5_locked_report_access": False,
                }
            )
        summary = {**identity, **dict(basis["clustering_summary"])}
        clustering_summary_rows.append(summary)
        search_rows.append(
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
            }
        )
    key = lambda row: (
        str(row["dataset"]),
        int(row["candidate_set_ordinal"]),
        int(row["model_seed"]),
    )
    classification_rows.sort(key=lambda row: (*key(row), int(row["classification_split_index"])))
    clustering_rows.sort(key=lambda row: (*key(row), int(row["kmeans_seed_index"])))
    clustering_summary_rows.sort(key=key)
    search_rows.sort(key=key)
    return classification_rows, clustering_rows, clustering_summary_rows, search_rows


def run_stage_evaluation(
    *,
    stage: str,
    plan_path: Path,
    result_root: Path,
    h4_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    if stage not in STAGE_ARTIFACT_PREFIX:
        raise ValueError(stage)
    rows = _read_csv(plan_path)
    if not rows or any(row["stage"] != stage for row in rows):
        raise RuntimeError(f"invalid or empty H5 {stage} plan")
    _validate_sealed_stage_inputs(
        stage=stage, plan_path=plan_path, result_root=result_root, rows=rows
    )
    manifest = verify_global_label_role_seal(
        result_root=result_root, private_mount_root=private_mount_root
    )
    split_seeds = {
        dataset: [int(value) for value in manifest["seed_bank"]["per_dataset"][dataset]["within_pool_classification_split_seeds"]]
        for dataset in ("acm", "freebase")
    }
    clustering_seeds = [int(value) for value in manifest["seed_bank"]["clustering_search_seeds"]]
    job_root = result_root / "stage_evaluation" / stage.lower()
    job_root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    missing = []
    for row in rows:
        path = job_root / f"{row['physical_run_id']}.json"
        if path.is_file():
            value = read_json(path)
            representation_sha = _resolve_representation(
                row, result_root=result_root, h4_result_root=h4_result_root
            )[1]
            _validate_existing_payload(value, row, representation_sha)
            payloads[row["physical_run_id"]] = value
        else:
            missing.append(row)
    if max(1, int(workers)) == 1:
        for row in missing:
            value = _evaluate_one(
                row,
                result_root=result_root,
                h4_result_root=h4_result_root,
                private_mount_root=private_mount_root,
                split_seeds_by_dataset=split_seeds,
                clustering_seeds=clustering_seeds,
                label_manifest=manifest,
            )
            _write_once(job_root / f"{row['physical_run_id']}.json", value)
            payloads[row["physical_run_id"]] = value
    else:
        jobs = [
            (
                row,
                str(result_root),
                str(h4_result_root),
                str(private_mount_root),
                split_seeds,
                clustering_seeds,
                manifest,
            )
            for row in missing
        ]
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            future_to_row = {executor.submit(_worker, job): job[0] for job in jobs}
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                value = future.result()
                _write_once(job_root / f"{row['physical_run_id']}.json", value)
                payloads[row["physical_run_id"]] = value
    if len(payloads) != len(rows):
        raise RuntimeError(f"H5 {stage} evaluation incomplete: {len(payloads)}/{len(rows)}")
    classification, clustering, clustering_summary, search = _flatten(list(payloads.values()))
    prefix = STAGE_ARTIFACT_PREFIX[stage]
    if stage == "THIRD_SEED_R3_AUDIT":
        _write_csv_once(result_root / f"{prefix}_METRICS.csv", search)
        _write_csv_once(result_root / f"{prefix}_CLASSIFICATION_METRICS.csv", classification)
        _write_csv_once(result_root / f"{prefix}_CLUSTERING_METRICS_LONG.csv", clustering)
        _write_csv_once(result_root / f"{prefix}_CLUSTERING_METRICS_SUMMARY.csv", clustering_summary)
    else:
        _write_csv_once(result_root / f"{prefix}_CLASSIFICATION_METRICS.csv", classification)
        _write_csv_once(result_root / f"{prefix}_CLUSTERING_METRICS_LONG.csv", clustering)
        _write_csv_once(result_root / f"{prefix}_CLUSTERING_METRICS_SUMMARY.csv", clustering_summary)
        _write_csv_once(result_root / f"{prefix}_SEARCH_METRICS.csv", search)
    return {
        "classification": classification,
        "clustering_long": clustering,
        "clustering_summary": clustering_summary,
        "search": search,
    }
