"""Append-only helpers for H7 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.m4a_h6.common import (
    append_jsonl_once,
    envelope,
    read_csv,
    safe_relative,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from metachorus_v53.audit.common import canonical_json_hash


def seal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seal an H7 execution plan using typed canonical rows."""

    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value.pop("plan_seal_sha256", None)
        canonical_rows.append(value)
    basis = {
        "stage": stage,
        "row_count": len(canonical_rows),
        "rows_identity": canonical_json_hash(canonical_rows),
        "labels_mounted": False,
        "run_level_parallel_only": True,
        "maximum_gpu_count": 2,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    identity = canonical_json_hash(basis)
    sealed = [{**row, "plan_seal_sha256": identity} for row in canonical_rows]
    return sealed, envelope(
        "metachorus.v53.m4a-h7.execution-plan-seal.v1",
        "SEALED_BEFORE_GPU_TRAINING",
        basis,
    )


__all__ = [
    "append_jsonl_once",
    "canonical_json_hash",
    "envelope",
    "read_csv",
    "safe_relative",
    "seal_rows",
    "write_csv_once",
    "write_json_once",
    "write_text_once",
]
