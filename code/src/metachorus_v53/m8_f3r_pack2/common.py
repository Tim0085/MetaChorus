"""Deterministic, append-only helpers for PACK2."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def envelope(schema: str, status: str, basis: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(basis)
    return {"schema_version": schema, "status": status, "basis": normalized, "identity": canonical_json_hash(normalized)}


def _publish_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"governed output cannot be a symlink: {path}")
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"append-only output conflict: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    if pending.exists():
        raise FileExistsError(f"pending output preserved: {pending}")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(pending, path)
        except OSError:
            with pending.open("rb") as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        if path.read_bytes() != payload:
            raise RuntimeError(f"governed output publication changed bytes: {path}")
    finally:
        if path.is_file() and path.read_bytes() == payload:
            pending.unlink(missing_ok=True)


def write_text_once(path: str | Path, text: str) -> None:
    _publish_once(Path(path), text.encode("utf-8"))


def write_json_once(path: str | Path, value: Any) -> None:
    write_text_once(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_csv_once(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Iterable[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows([dict(row) for row in rows])
    write_text_once(path, buffer.getvalue())


def read_json(path: str | Path) -> Any:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV lacks a header: {source}")
        return list(reader)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"not a boolean: {value!r}")

