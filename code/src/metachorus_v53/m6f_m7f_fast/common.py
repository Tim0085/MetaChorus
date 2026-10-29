"""Canonical, append-only helpers for the M6F/M7F fast successor."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import (
    canonical_json_hash,
    read_json,
    sha256_file,
)


def envelope(schema_version: str, status: str, basis: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(basis)
    return {
        "schema_version": schema_version,
        "status": status,
        "basis": payload,
        "identity": canonical_json_hash(payload),
    }


def safe_relative(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError(f"non-portable relative path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value}")
    return path


def atomic_replace_bytes(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    payload = (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"append-only JSON conflict: {target}")
        return target
    return atomic_replace_bytes(target, payload)


def write_text_once(path: str | Path, value: str) -> Path:
    target = Path(path)
    payload = value.encode("utf-8")
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"append-only text conflict: {target}")
        return target
    return atomic_replace_bytes(target, payload)


def write_csv_once(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> Path:
    target = Path(path)
    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return str(value)

    expected = [
        {field: cell(row.get(field, "")) for field in fieldnames}
        for row in rows
    ]
    if target.exists():
        with target.open("r", encoding="utf-8-sig", newline="") as stream:
            observed = list(csv.DictReader(stream))
        if observed != expected:
            raise FileExistsError(f"append-only CSV conflict: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            writer.writerows(expected)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not stream.closed:
        raise ValueError(f"CSV is empty or not closed: {source}")
    return rows


def verify_sha256(path: str | Path, expected: str, *, role: str) -> Path:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{role} is not a regular file: {source}")
    observed = sha256_file(source)
    if observed != expected:
        raise ValueError(f"{role} SHA-256 mismatch: expected {expected}, observed {observed}")
    return source


__all__ = [
    "canonical_json_hash",
    "read_json",
    "sha256_file",
    "envelope",
    "safe_relative",
    "write_json_once",
    "write_text_once",
    "write_csv_once",
    "read_csv",
    "verify_sha256",
]
