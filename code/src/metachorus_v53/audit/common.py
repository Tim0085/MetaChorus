"""Shared deterministic primitives for the v5.3 M0-A audit.

Only Python's standard library is used here so identity and validation scripts
can run before any model dependency is imported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


HASH_CHUNK_BYTES = 1024 * 1024
SHA256_HEX_LENGTH = 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in text
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"append-only destination already exists: {destination}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_text(path: str | Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"append-only destination already exists: {destination}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative_path(path: str | Path, root: str | Path) -> str:
    source = Path(path).resolve()
    base = Path(root).resolve()
    return source.relative_to(base).as_posix()


def path_ref(
    path: str | Path,
    roots: Mapping[str, str | Path],
    *,
    allow_local_absolute: bool,
) -> dict[str, Any]:
    source = Path(path).resolve()
    matches: list[tuple[int, str, Path]] = []
    for root_id, root_path in roots.items():
        root = Path(root_path).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), str(root_id), relative))
    if not matches:
        raise ValueError(f"path is outside every registered source root: {source}")
    _, root_id, relative = max(matches, key=lambda item: item[0])
    result: dict[str, Any] = {
        "source_root_id": root_id,
        "relative_path": relative.as_posix(),
    }
    if allow_local_absolute:
        result["absolute_path_local_only"] = str(source)
    return result


def iter_files(root: str | Path) -> Iterable[Path]:
    base = Path(root).resolve()
    for current, dirs, files in os.walk(base):
        dirs[:] = sorted(name for name in dirs if name != ".git")
        for name in sorted(files):
            yield Path(current) / name


def directory_manifest(
    root: str | Path,
    *,
    include: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in iter_files(base):
        if include is not None and not include(path):
            continue
        try:
            rows.append(
                {
                    "relative_path": path.relative_to(base).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except (OSError, PermissionError) as error:
            failures.append(
                {
                    "relative_path": path.relative_to(base).as_posix(),
                    "error": str(error),
                }
            )
    rows.sort(key=lambda item: item["relative_path"])
    identity_payload = {
        "manifest_rule": "posix_relative_path_sorted_sha256_size_v1",
        "files": rows,
    }
    return {
        **identity_payload,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "directory_manifest_hash": canonical_json_hash(identity_payload),
        "failures": failures,
    }


def csv_semantic_identity(
    path: str | Path,
    *,
    primary_key: Sequence[str],
) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = set(primary_key).difference(fieldnames)
    if missing:
        raise ValueError(f"CSV primary-key columns are missing: {sorted(missing)}")
    rows.sort(key=lambda row: tuple(str(row[key]) for key in primary_key))
    return {
        "raw_sha256": sha256_file(source),
        "columns": fieldnames,
        "row_count": len(rows),
        "primary_key": list(primary_key),
        "semantic_sha256": canonical_json_hash(
            {"columns": fieldnames, "primary_key": list(primary_key), "rows": rows}
        ),
    }


def json_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = read_json(source)
    return {
        "raw_sha256": sha256_file(source),
        "canonical_sha256": canonical_json_hash(value),
        "size_bytes": source.stat().st_size,
    }


def build_checksums(root: str | Path, *, exclude_names: set[str] | None = None) -> str:
    base = Path(root).resolve()
    exclusions = set(exclude_names or ())
    rows: list[str] = []
    for path in iter_files(base):
        relative = path.relative_to(base).as_posix()
        if path.name in exclusions:
            continue
        rows.append(f"{sha256_file(path)}  {relative}")
    return "\n".join(rows) + "\n"


def load_checksums(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or not is_sha256(digest) or not relative:
            raise ValueError(f"invalid checksum line: {line!r}")
        if relative in result:
            raise ValueError(f"duplicate checksum path: {relative}")
        result[relative] = digest
    return result
