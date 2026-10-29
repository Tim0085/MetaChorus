"""Auditable mutable pointers backed by append-only state history."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .common import canonical_json_hash, sha256_file, write_text_once


def _payload(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json_state(path_value: str | Path, value: Mapping[str, Any]) -> None:
    """Replace the canonical state pointer only after preserving its prior bytes."""

    path = Path(path_value)
    payload = _payload(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"governed state pointer cannot be a symlink: {path}")
    if path.exists():
        if not path.is_file():
            raise FileExistsError(path)
        prior = path.read_bytes()
        if prior == payload:
            return
        digest = hashlib.sha256(prior).hexdigest()
        history = path.parent / "state_history" / path.stem / f"{digest}.json"
        write_text_once(history, prior.decode("utf-8"))
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    if pending.exists():
        raise FileExistsError(f"pending state update preserved: {pending}")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    _fsync_directory(path.parent)
    if path.read_bytes() != payload:
        raise RuntimeError(f"governed state pointer changed during publication: {path}")


def publish_snapshot(
    result_root: str | Path,
    *,
    prefix: str,
    member_names: Sequence[str],
) -> dict[str, Any]:
    """Publish a deterministic, append-only remote recovery snapshot."""

    if prefix not in {
        "SNAPSHOT_0_SELECTION",
        "SNAPSHOT_1_FREEBASE",
        "SNAPSHOT_2_DBLP",
    }:
        raise ValueError("unsupported RERUN-2G snapshot prefix")
    result = Path(result_root).resolve(strict=True)
    rows = []
    for name in sorted(set(map(str, member_names))):
        if not name or "\\" in name or "/" in name or name in {".", ".."}:
            raise ValueError("snapshot members must be canonical result-root filenames")
        path = result / name
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise FileNotFoundError(path)
        rows.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    snapshot_basis = {
        "prefix": prefix,
        "members": rows,
        "member_count": len(rows),
    }
    snapshot_identity = canonical_json_hash(snapshot_basis)
    filename = f"{prefix}_{snapshot_identity[:8]}.zip"
    destination = result / filename
    pending = result / f".{filename}.pending-{os.getpid()}"
    if pending.exists():
        raise FileExistsError(f"pending snapshot evidence preserved: {pending}")
    manifest = {
        "schema_version": "metachorus.v53.m8-f3r-rerun-2g.snapshot.v1",
        "status": "PASS",
        "basis": snapshot_basis,
        "identity": snapshot_identity,
    }
    with ZipFile(pending, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for row in rows:
            info = ZipInfo(
                str(row["relative_path"]), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(
                info,
                (result / str(row["relative_path"])).read_bytes(),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )
        info = ZipInfo("SNAPSHOT_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100600 << 16
        archive.writestr(
            info,
            _payload(manifest),
            compress_type=ZIP_DEFLATED,
            compresslevel=9,
        )
    with pending.open("r+b") as stream:
        os.fsync(stream.fileno())
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != sha256_file(pending)
        ):
            raise FileExistsError(f"snapshot publication conflict: {destination}")
        pending.unlink()
    else:
        os.replace(pending, destination)
        _fsync_directory(result)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("published recovery snapshot failed CRC verification")
        observed = sorted(
            info.filename for info in archive.infolist() if not info.is_dir()
        )
        if observed != sorted([*(str(row["relative_path"]) for row in rows), "SNAPSHOT_MANIFEST.json"]):
            raise RuntimeError("published recovery snapshot member set changed")
        if json.loads(archive.read("SNAPSHOT_MANIFEST.json")) != manifest:
            raise RuntimeError("published recovery snapshot manifest changed")
    return {
        "snapshot_name": filename,
        "snapshot_sha256": sha256_file(destination),
        "snapshot_size_bytes": destination.stat().st_size,
        "snapshot_identity": snapshot_identity,
        "member_count": len(rows),
        "status": "PASS",
    }


__all__ = ["publish_json_state", "publish_snapshot"]
