"""Append-only scientific failure closure for H5 stage gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from metachorus_v53.audit.common import (
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
    utc_now,
)


ALLOWED_FAILURES = {
    "H5_R1_POOL_RECALL_FAIL": "R1_RECALL",
    "H5_R2_CLASSIFICATION_BRIDGE_FAIL": "R2_EVALUATION",
    "TC_LPMSO_H5_R3_RELIABILITY_OR_FIDELITY_FAIL": "R3_EVALUATION",
    "H5_THREE_SEED_RELIABILITY_FAIL": "THIRD_SEED_EVALUATION",
    "TC_LPMSO_H5_DEVELOPMENT_FAIL": "R4_EVALUATION",
}


def write_scientific_failure(
    *,
    result_root: Path,
    failure_code: str,
    decision_path: Path,
    counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if failure_code not in ALLOWED_FAILURES:
        raise ValueError(f"unsupported H5 scientific failure code: {failure_code}")
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    basis = {
        "failed_at_utc": utc_now(),
        "failed_stage": ALLOWED_FAILURES[failure_code],
        "failure_code": failure_code,
        "decision_artifact": decision_path.name,
        "decision_sha256": sha256_file(decision_path),
        "scientific_gate_failure": True,
        "engineering_retry_permitted": False,
        "metric_based_retry_forbidden": True,
        "completed_runs_preserved": True,
        "global_label_pool_changed": False,
        "exploration_set_changed": False,
        "thresholds_changed": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
        "downstream_training_started": False,
        "counts": dict(counts or {}),
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h5.failure.v1",
        "status": "TC_LPMSO_H5_DEVELOPMENT_FAIL",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }
    path = result_root / "M4A_H5_FAILURE.json"
    if path.is_file():
        observed = read_json(path)
        immutable_keys = (
            "failed_stage",
            "failure_code",
            "decision_artifact",
            "decision_sha256",
        )
        if any(
            observed.get("basis", {}).get(key) != basis[key]
            for key in immutable_keys
        ):
            raise RuntimeError("append-only H5 failure receipt conflict")
        return observed
    atomic_write_json(path, payload)
    return payload

