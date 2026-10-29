"""Append-only and canonical helpers shared by H6 stages."""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_json_hash,
    read_json,
)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def envelope(schema_version: str, status: str, basis: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(basis)
    return {
        "schema_version": schema_version,
        "status": status,
        "basis": value,
        "identity": canonical_json_hash(value),
    }


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    if target.exists():
        if read_json(target) != dict(value):
            raise FileExistsError(f"append-only JSON conflict: {target}")
        return target
    atomic_write_json(target, dict(value))
    return target


def write_csv_once(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> Path:
    target = Path(path)
    normalized = [{field: row.get(field, "") for field in fieldnames} for row in rows]
    if target.exists():
        observed = read_csv(target)
        expected = [
            {field: str(row.get(field, "")) for field in fieldnames}
            for row in normalized
        ]
        if observed != expected:
            raise FileExistsError(f"append-only CSV conflict: {target}")
        return target
    atomic_write_csv(target, normalized, fieldnames=fieldnames)
    return target


def write_text_once(path: str | Path, value: str) -> Path:
    target = Path(path)
    if target.exists():
        if target.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"append-only text conflict: {target}")
        return target
    atomic_write_text(target, value)
    return target


def append_jsonl_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    event = dict(value)
    event_id = str(event.get("event_id", ""))
    if not event_id:
        raise ValueError("append-only JSONL events require event_id")
    existing: list[dict[str, Any]] = []
    if target.exists():
        existing = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    matches = [row for row in existing if str(row.get("event_id")) == event_id]
    if matches:
        if matches != [event]:
            raise FileExistsError(f"append-only JSONL event conflict: {event_id}")
        return target
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
    return target


def seal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    schema_version: str = "metachorus.v53.m4a-h6.execution-plan-seal.v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    return sealed, envelope(schema_version, "SEALED_BEFORE_GPU_TRAINING", basis)


def safe_relative(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError(f"non-portable relative path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value}")
    return path
