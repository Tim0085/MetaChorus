"""Canonical append-only helpers shared by H8 stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import canonical_json_hash
from metachorus_v53.m4a_h6.common import (
    append_jsonl_once,
    envelope,
    read_csv,
    safe_relative,
    write_csv_once,
    write_json_once,
    write_text_once,
)


def seal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    extra_basis: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seal typed H8 plan rows before any GPU process sees them."""

    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value.pop("plan_seal_sha256", None)
        canonical_rows.append(value)
    basis: dict[str, Any] = {
        "stage": stage,
        "row_count": len(canonical_rows),
        "rows_identity": canonical_json_hash(canonical_rows),
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "run_level_parallel_only": True,
        "maximum_gpu_count": 2,
        "multi_gpu_single_run": False,
        "h5_locked_report_access": False,
        "locked_evaluation_started": False,
    }
    if extra_basis:
        basis.update(dict(extra_basis))
    identity = canonical_json_hash(basis)
    sealed = [{**row, "plan_seal_sha256": identity} for row in canonical_rows]
    return sealed, envelope(
        "metachorus.v53.m4a-h8.execution-plan-seal.v1",
        "SEALED_BEFORE_GPU_TRAINING",
        basis,
    )


def validate_envelope(
    value: Mapping[str, Any], *, schema_version: str, status: str | None = None
) -> dict[str, Any]:
    basis = value.get("basis")
    if (
        value.get("schema_version") != schema_version
        or not isinstance(basis, Mapping)
        or value.get("identity") != canonical_json_hash(basis)
        or (status is not None and value.get("status") != status)
    ):
        raise ValueError(f"invalid H8 envelope: {schema_version}")
    return dict(value)


__all__ = [
    "append_jsonl_once",
    "canonical_json_hash",
    "envelope",
    "read_csv",
    "safe_relative",
    "seal_rows",
    "validate_envelope",
    "write_csv_once",
    "write_json_once",
    "write_text_once",
]
