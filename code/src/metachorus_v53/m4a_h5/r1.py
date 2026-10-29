"""Resumable H5 R1 evaluation-only pass and frozen pool construction."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import (
    atomic_write_csv,
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
    utc_now,
)

from .constants import CONTROLS, FREEBASE_EXPLOITATION_COUNT
from .evaluation import evaluate_classification, evaluate_clustering
from .failure import write_scientific_failure
from .labels import verify_global_label_role_seal
from .protocol import (
    aggregate_classification_by_seed,
    build_r1_pool,
    classification_stability,
    freebase_classification_ranking,
    r1_recall_audit,
    select_freebase_exploration,
)
from .truth import bind_development_truth


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


def _evaluate_one(
    inventory_row: Mapping[str, Any],
    *,
    h1_result_root: Path,
    private_mount_root: Path,
    split_seeds_by_dataset: Mapping[str, Sequence[int]],
    clustering_seeds: Sequence[int],
    label_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(inventory_row)
    representation_path = h1_result_root / str(row["representation_relative_path"])
    if sha256_file(representation_path) != str(row["representation_sha256"]):
        raise RuntimeError(f"R1 representation hash changed: {row['physical_run_id']}")
    representation = np.load(representation_path, allow_pickle=False)
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
        "physical_run_id": row["physical_run_id"],
        "dataset": dataset,
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "model_seed": int(row["model_seed"]),
        "update": 300,
        "representation_sha256": row["representation_sha256"],
        "classification": classification,
        "clustering_long": clustering_long,
        "clustering_summary": clustering_summary,
        "new_training_count": 0,
        "h5_locked_report_access": False,
    }
    return {
        "schema_version": "metachorus.v53.m4a-h5.r1-evaluation.v1",
        "status": "PASS",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }


def _worker(payload: tuple[dict[str, Any], str, str, dict[str, list[int]], list[int], dict[str, Any]]) -> dict[str, Any]:
    row, h1_root, mount_root, split_seeds, clustering_seeds, label_manifest = payload
    return _evaluate_one(
        row,
        h1_result_root=Path(h1_root),
        private_mount_root=Path(mount_root),
        split_seeds_by_dataset=split_seeds,
        clustering_seeds=clustering_seeds,
        label_manifest=label_manifest,
    )


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        observed = read_json(path)
        if observed.get("identity") != payload.get("identity"):
            raise RuntimeError(f"append-only R1 evaluation conflict: {path.name}")
        return
    atomic_write_json(path, dict(payload))


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise RuntimeError(f"append-only R1 JSON conflict: {path.name}")
        return
    atomic_write_json(path, dict(payload))


def _write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to publish an empty R1 table: {path.name}")
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
        if _read_csv(path) != expected:
            raise RuntimeError(f"append-only R1 CSV conflict: {path.name}")
        return
    atomic_write_csv(path, rows, fieldnames=fieldnames)


def _validate_existing_payload(
    payload: Mapping[str, Any], inventory_row: Mapping[str, Any]
) -> None:
    basis = payload.get("basis")
    if (
        payload.get("schema_version") != "metachorus.v53.m4a-h5.r1-evaluation.v1"
        or payload.get("status") != "PASS"
        or not isinstance(basis, Mapping)
        or payload.get("identity") != canonical_json_hash(basis)
    ):
        raise RuntimeError(f"invalid R1 payload: {inventory_row['physical_run_id']}")
    expected = {
        "physical_run_id": str(inventory_row["physical_run_id"]),
        "dataset": str(inventory_row["dataset"]),
        "candidate_set_id": str(inventory_row["candidate_set_id"]),
        "candidate_set_hash": str(inventory_row["candidate_set_hash"]),
        "candidate_set_ordinal": int(inventory_row["candidate_set_ordinal"]),
        "model_seed": int(inventory_row["model_seed"]),
        "update": 300,
        "representation_sha256": str(inventory_row["representation_sha256"]),
        "new_training_count": 0,
        "h5_locked_report_access": False,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": basis.get(key)}
        for key, expected_value in expected.items()
        if basis.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            f"stale or mismatched R1 payload {inventory_row['physical_run_id']}: {mismatches}"
        )


def _collect_rows(payloads: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for payload in payloads:
        basis = payload["basis"]
        identity = {
            "physical_run_id": basis["physical_run_id"],
            "dataset": basis["dataset"],
            "candidate_set_id": basis["candidate_set_id"],
            "candidate_set_hash": basis["candidate_set_hash"],
            "candidate_set_ordinal": basis["candidate_set_ordinal"],
            "model_seed": basis["model_seed"],
            "update": 300,
            "representation_sha256": basis["representation_sha256"],
        }
        classification_values: list[float] = []
        for row in basis["classification"]:
            classification_values.append(float(row["F_cls"]))
            long_rows.append(
                {
                    **identity,
                    "task": "classification",
                    "repeat_index": int(row["classification_split_index"]),
                    "repeat_seed": int(row["classification_split_seed"]),
                    "micro_f1": row["micro_f1"],
                    "macro_f1": row["macro_f1"],
                    "nmi": "",
                    "ari": "",
                    "F_cls": row["F_cls"],
                    "F_clu": "",
                    "assignment_sha256": "",
                    "h5_locked_report_access": False,
                }
            )
        for row in basis["clustering_long"]:
            long_rows.append(
                {
                    **identity,
                    "task": "clustering",
                    "repeat_index": int(row["kmeans_seed_index"]),
                    "repeat_seed": int(row["kmeans_seed"]),
                    "micro_f1": "",
                    "macro_f1": "",
                    "nmi": row["nmi"],
                    "ari": row["ari"],
                    "F_cls": "",
                    "F_clu": row["F_clu"],
                    "assignment_sha256": row["assignment_sha256"],
                    "h5_locked_report_access": False,
                }
            )
        summary = basis["clustering_summary"]
        score_rows.append(
            {
                **identity,
                "mean_F_cls": mean(classification_values),
                "sample_sd_F_cls": stdev(classification_values),
                "mean_F_clu": summary["mean_F_clu"],
                "sample_sd_F_clu": summary["sample_sd_F_clu"],
                "SE_clu": summary["SE_clu"],
                "LCB_clu": summary["LCB_clu"],
                "assignment_stability_mean": summary["assignment_stability_mean"],
                "assignment_stability_p10": summary["assignment_stability_p10"],
                "new_training_count": 0,
            }
        )
    long_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["candidate_set_ordinal"]),
            int(row["model_seed"]),
            str(row["task"]),
            int(row["repeat_index"]),
        )
    )
    score_rows.sort(
        key=lambda row: (
            str(row["dataset"]), int(row["candidate_set_ordinal"]), int(row["model_seed"])
        )
    )
    return long_rows, score_rows


def run_r1(
    *,
    result_root: Path,
    h1_result_root: Path,
    h2_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    preflight = read_json(result_root / "M4A_H5_PREFLIGHT_COMPLETE.json")
    if preflight.get("status") != "READY_FOR_R1_EVALUATION":
        raise RuntimeError("H5 preflight is not ready for R1 evaluation")
    label_manifest = verify_global_label_role_seal(
        result_root=result_root, private_mount_root=private_mount_root
    )
    split_seeds = {
        dataset: [int(value) for value in label_manifest["seed_bank"]["per_dataset"][dataset]["within_pool_classification_split_seeds"]]
        for dataset in ("acm", "freebase")
    }
    clustering_seeds = [int(value) for value in label_manifest["seed_bank"]["clustering_search_seeds"]]
    exploration_seed = int(label_manifest["seed_bank"]["freebase_exploration_seed"])
    inventory = _read_csv(result_root / "TC_LPMSO_H5_R1_REPRESENTATION_INVENTORY.csv")
    if len(inventory) != 460:
        raise RuntimeError("H5 R1 inventory is not exactly 460 representations")
    job_root = result_root / "r1_evaluation"
    job_root.mkdir(exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for row in inventory:
        path = job_root / f"{row['physical_run_id']}.json"
        if path.is_file():
            payload = read_json(path)
            _validate_existing_payload(payload, row)
            payloads[row["physical_run_id"]] = payload
        else:
            missing.append(dict(row))
    _append_log(
        result_root / "M4A_H5_EXECUTION_LOG.jsonl",
        {
            "timestamp_utc": utc_now(),
            "stage": "R1_EVALUATION",
            "status": "STARTED",
            "completed_before_start": len(payloads),
            "remaining": len(missing),
            "workers": int(workers),
            "gpu_required": False,
        },
    )
    worker_count = max(1, int(workers))
    if worker_count == 1:
        for row in missing:
            payload = _evaluate_one(
                row,
                h1_result_root=h1_result_root,
                private_mount_root=private_mount_root,
                split_seeds_by_dataset=split_seeds,
                clustering_seeds=clustering_seeds,
                label_manifest=label_manifest,
            )
            _write_result(job_root / f"{row['physical_run_id']}.json", payload)
            payloads[str(row["physical_run_id"])] = payload
    else:
        jobs = [
            (
                row,
                str(h1_result_root),
                str(private_mount_root),
                split_seeds,
                clustering_seeds,
                label_manifest,
            )
            for row in missing
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_row = {executor.submit(_worker, job): job[0] for job in jobs}
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                payload = future.result()
                _write_result(job_root / f"{row['physical_run_id']}.json", payload)
                payloads[str(row["physical_run_id"])] = payload
    if len(payloads) != 460:
        raise RuntimeError(f"H5 R1 evaluation incomplete: {len(payloads)}/460")
    long_rows, score_rows = _collect_rows(list(payloads.values()))
    _write_csv_once(result_root / "TC_LPMSO_H5_R1_SEARCH_METRICS_LONG.csv", long_rows)
    _write_csv_once(result_root / "TC_LPMSO_H5_R1_SCORE_TABLE.csv", score_rows)
    classification_repeat_rows = [
        {
            **{key: row[key] for key in (
                "dataset", "candidate_set_id", "candidate_set_hash", "candidate_set_ordinal", "model_seed"
            )},
            "classification_split_index": row["repeat_index"],
            "F_cls": row["F_cls"],
        }
        for row in long_rows
        if row["task"] == "classification"
    ]
    stability_rows = [
        classification_stability(classification_repeat_rows, dataset=dataset, fidelity=300)
        for dataset in ("acm", "freebase")
    ]
    _write_csv_once(result_root / "TC_LPMSO_H5_CLASSIFICATION_STABILITY.csv", stability_rows)
    _write_json_once(
        result_root / "TC_LPMSO_H5_CLASSIFICATION_STABILITY_DECISION.json",
        {
            "schema_version": "metachorus.v53.m4a-h5.classification-stability-decision.v1",
            "status": "PASS" if all(row["status"] == "RELIABLE" for row in stability_rows) else "FAIL",
            "basis": {"rows": stability_rows, "raw_sd_snr_primary_gate": False},
        },
    )
    by_seed_rows = aggregate_classification_by_seed(classification_repeat_rows, "freebase")
    ranking = freebase_classification_ranking(by_seed_rows)
    control_hashes = {value[1] for value in CONTROLS["freebase"].values()}
    exploitation = [
        {**row, "selection_role": "EXPLOITATION_TOP80"}
        for row in ranking
        if str(row["candidate_set_hash"]) not in control_hashes
    ][:FREEBASE_EXPLOITATION_COUNT]
    exploration_input = [
        {
            "candidate_set_hash": row["candidate_set_hash"],
            "classification_rank": row["classification_rank"],
        }
        for row in ranking
    ]
    exploration = select_freebase_exploration(
        exploration_input,
        exploration_seed,
        excluded_hashes={str(row["candidate_set_hash"]) for row in exploitation},
    )
    acm_candidates = []
    seen_acm: set[str] = set()
    for row in score_rows:
        if row["dataset"] == "acm" and row["candidate_set_hash"] not in seen_acm:
            seen_acm.add(str(row["candidate_set_hash"]))
            acm_candidates.append(row)
    pool = build_r1_pool(
        acm_candidates=acm_candidates,
        freebase_ranking=ranking,
        freebase_exploration=exploration,
    )
    rank_by_hash = {str(row["candidate_set_hash"]): int(row["classification_rank"]) for row in ranking}
    exploitation_hashes = {str(row["candidate_set_hash"]) for row in exploitation}
    exploration_hashes = {str(row["candidate_set_hash"]) for row in exploration}
    for row in pool:
        candidate_hash = str(row["candidate_set_hash"])
        row["classification_rank"] = rank_by_hash.get(candidate_hash, "")
        row["exploitation_member"] = candidate_hash in exploitation_hashes
        row["exploration_member"] = candidate_hash in exploration_hashes
    _write_csv_once(result_root / "TC_LPMSO_H5_R1_EXPLOITATION_TABLE.csv", exploitation)
    _write_csv_once(result_root / "TC_LPMSO_H5_R1_EXPLORATION_TABLE.csv", exploration)
    _write_csv_once(result_root / "TC_LPMSO_H5_R1_POOL_TABLE.csv", pool)
    pool_basis = {
        "score_table_sha256": sha256_file(result_root / "TC_LPMSO_H5_R1_SCORE_TABLE.csv"),
        "exploitation_table_sha256": sha256_file(result_root / "TC_LPMSO_H5_R1_EXPLOITATION_TABLE.csv"),
        "exploration_table_sha256": sha256_file(result_root / "TC_LPMSO_H5_R1_EXPLORATION_TABLE.csv"),
        "pool_table_sha256": sha256_file(result_root / "TC_LPMSO_H5_R1_POOL_TABLE.csv"),
        "exploration_seed": exploration_seed,
        "acm_unique_set_count": sum(row["dataset"] == "acm" for row in pool),
        "freebase_unique_set_count": sum(row["dataset"] == "freebase" for row in pool),
        "freebase_exploitation_slot_count": 80,
        "freebase_exploration_slot_count": 40,
        "controls_outside_slots": True,
        "truth_opened_before_seal": False,
        "sealed_before_truth_recall": True,
    }
    pool_seal = {
        "schema_version": "metachorus.v53.m4a-h5.r1-pool-seal.v1",
        "status": "SEALED_BEFORE_TRUTH_RECALL",
        "basis": pool_basis,
        "identity": canonical_json_hash(pool_basis),
    }
    _write_json_once(result_root / "TC_LPMSO_H5_R1_POOL_SEAL.json", pool_seal)
    truth_rows, truth_binding = bind_development_truth(
        h2_result_root=h2_result_root, result_root=result_root
    )
    recall_rows = r1_recall_audit(pool, truth_rows)
    _write_csv_once(result_root / "TC_LPMSO_H5_R1_RECALL_AUDIT.csv", recall_rows)
    recall_pass = all(row["status"] == "PASS" for row in recall_rows)
    decision_basis = {
        "pool_seal_identity": pool_seal["identity"],
        "truth_binding_identity": truth_binding["identity"],
        "recall_audit_sha256": sha256_file(result_root / "TC_LPMSO_H5_R1_RECALL_AUDIT.csv"),
        "recall_pass": recall_pass,
        "exploration_resampling_permitted": False,
        "new_training_count": 0,
    }
    decision = {
        "schema_version": "metachorus.v53.m4a-h5.r1-decision.v1",
        "status": "H5_R1_POOL_RECALL_PASS" if recall_pass else "H5_R1_POOL_RECALL_FAIL",
        "basis": decision_basis,
        "identity": canonical_json_hash(decision_basis),
    }
    _write_json_once(result_root / "TC_LPMSO_H5_R1_DECISION.json", decision)
    if not recall_pass:
        write_scientific_failure(
            result_root=result_root,
            failure_code=decision["status"],
            decision_path=result_root / "TC_LPMSO_H5_R1_DECISION.json",
            counts={"evaluated_representation_count": 460, "new_training_count": 0},
        )
    _append_log(
        result_root / "M4A_H5_EXECUTION_LOG.jsonl",
        {
            "timestamp_utc": utc_now(),
            "stage": "R1_EVALUATION_AND_POOL",
            "status": decision["status"],
            "evaluated_representation_count": 460,
            "new_training_count": 0,
            "gpu_required": False,
        },
    )
    return decision
