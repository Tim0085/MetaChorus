"""Append-only H7 terminal failure receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from metachorus_v53.audit.common import canonical_json_hash, read_json, utc_now

from .common import write_json_once


ALLOWED_FAILURES = {
    "H7_CANARY_FAIL_STOP_BEFORE_FANOUT": "CANARY",
    "H7_R4_ENGINEERING_FAILURE": "R4_TRAINING",
    "TC_LPMSO_H7_NO_SAFE_SET": "R4_EVALUATION",
    "TC_LPMSO_H7_SELECTION_FAIL": "R4_SELECTION",
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
    if scientific_gate_failure and engineering_retry_permitted:
        raise ValueError("H7 scientific failures cannot authorize engineering retries")
    target = result_root / "M4A_H7_FAILURE.json"
    expected = {
        "failure_code": failure_code,
        "failed_stage": ALLOWED_FAILURES[failure_code],
        "evidence_artifact": evidence_artifact,
        "evidence_sha256": evidence_sha256,
    }
    if target.is_file():
        observed = read_json(target)
        basis = observed.get("basis", {})
        if (
            observed.get("schema_version") != "metachorus.v53.m4a-h7.failure.v1"
            or observed.get("status") != "TC_LPMSO_H7_FAILED"
            or observed.get("identity") != canonical_json_hash(basis)
            or any(basis.get(key) != value for key, value in expected.items())
        ):
            raise FileExistsError("append-only H7 failure receipt conflict")
        return observed
    basis = {
        "failed_at_utc": utc_now(),
        **expected,
        "counts": dict(counts),
        "scientific_gate_failure": bool(scientific_gate_failure),
        "engineering_retry_permitted": bool(engineering_retry_permitted),
        "metric_based_retry_forbidden": bool(scientific_gate_failure),
        "completed_runs_preserved": True,
        "h6_parent_failure_preserved": True,
        "shortlist_rules_changed": False,
        "candidate_sets_replaced": False,
        "development_seeds_changed": False,
        "manual_safety_gate_changed": False,
        "fallback_to_second_ranked_set": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h7.failure.v1",
        "status": "TC_LPMSO_H7_FAILED",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }
    write_json_once(target, payload)
    return payload
