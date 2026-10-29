"""Append-only H6 failure and stop receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from metachorus_v53.audit.common import canonical_json_hash, read_json, utc_now

from .common import write_json_once


ALLOWED_FAILURES = {
    "H6_CANARY_FAIL_STOP_BEFORE_FANOUT": "CANARY",
    "H6_THIRD_SEED_ENGINEERING_FAILURE": "THIRD_SEED_TRAINING",
    "TC_LPMSO_H6_THREE_SEED_R3_FAIL": "THREE_SEED_R3_EVALUATION",
    "TC_LPMSO_H6_R4_DEVELOPMENT_FAIL": "R4_EVALUATION",
    "PARTIAL_RESOURCE_PAUSE": "RESOURCE_AUDIT",
    "IMMEDIATE_COMMON_ROOT_CAUSE_PAUSE": "COMMON_ROOT_CAUSE_AUDIT",
}


def write_failure(
    result_root: Path,
    *,
    failure_code: str,
    evidence_artifact: str,
    evidence_sha256: str,
    counts: Mapping[str, Any],
    scientific_gate_failure: bool,
    engineering_retry_permitted: bool,
) -> dict[str, Any]:
    if failure_code not in ALLOWED_FAILURES:
        raise ValueError(failure_code)
    target = result_root / "M4A_H6_FAILURE.json"
    if target.is_file():
        observed = read_json(target)
        basis = observed.get("basis", {})
        expected = {
            "failure_code": failure_code,
            "failed_stage": ALLOWED_FAILURES[failure_code],
            "evidence_artifact": evidence_artifact,
            "evidence_sha256": evidence_sha256,
        }
        if (
            observed.get("schema_version") != "metachorus.v53.m4a-h6.failure.v1"
            or observed.get("status") != "TC_LPMSO_H6_DEVELOPMENT_FAIL"
            or observed.get("identity") != canonical_json_hash(basis)
            or any(basis.get(key) != value for key, value in expected.items())
        ):
            raise FileExistsError("append-only H6 failure receipt conflict")
        return observed
    basis = {
        "failed_at_utc": utc_now(),
        "failure_code": failure_code,
        "failed_stage": ALLOWED_FAILURES[failure_code],
        "evidence_artifact": evidence_artifact,
        "evidence_sha256": evidence_sha256,
        "counts": dict(counts),
        "scientific_gate_failure": bool(scientific_gate_failure),
        "engineering_retry_permitted": bool(engineering_retry_permitted),
        "metric_based_retry_forbidden": bool(scientific_gate_failure),
        "completed_runs_preserved": True,
        "h5_failure_and_resource_pause_preserved": True,
        "h5_120_failed_attempts_preserved": True,
        "uncertainty_pool_changed": False,
        "thresholds_changed": False,
        "third_seed_changed": False,
        "r4_started": failure_code == "TC_LPMSO_H6_R4_DEVELOPMENT_FAIL",
        "downstream_training_started": failure_code == "TC_LPMSO_H6_R4_DEVELOPMENT_FAIL",
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h6.failure.v1",
        "status": "TC_LPMSO_H6_DEVELOPMENT_FAIL",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }
    write_json_once(target, payload)
    return payload
