"""Safe NumPy serialization and semantic array identities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from metachorus_v53.audit.common import sha256_file


ALLOWED_DTYPES = {
    np.dtype("float32").str,
    np.dtype("float64").str,
    np.dtype("int32").str,
    np.dtype("int64").str,
    np.dtype("uint8").str,
    np.dtype("bool").str,
}


def as_safe_array(value: Any, *, dtype: np.dtype[Any] | str | None = None) -> np.ndarray:
    """Return a detached C-contiguous, non-object CPU NumPy array."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("object dtype is forbidden")
    if array.dtype.str not in ALLOWED_DTYPES:
        raise TypeError(f"unsupported array dtype: {array.dtype}")
    return np.ascontiguousarray(array)


def array_semantic_sha256(value: Any) -> str:
    """Hash dtype, shape, storage order, and exact element bytes."""
    array = as_safe_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )
    digest.update(b"C")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def array_summary(value: Any) -> dict[str, Any]:
    array = as_safe_array(value)
    finite = bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else True
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
        "semantic_sha256": array_semantic_sha256(array),
        "finite": finite,
        "minimum": None if array.size == 0 else array.min().item(),
        "maximum": None if array.size == 0 else array.max().item(),
    }


def write_npy_deterministic(path: str | Path, value: Any) -> dict[str, Any]:
    """Atomically write an NPY v2 file without pickle or overwrite."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"append-only destination already exists: {destination}")
    array = as_safe_array(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.lib.format.write_array(
                stream, array, version=(2, 0), allow_pickle=False
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = array_summary(array)
    result.update(
        {
            "raw_file_sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
    )
    return result


def load_npy_safe(path: str | Path, *, mmap_mode: str | None = None) -> np.ndarray:
    array = np.load(Path(path), allow_pickle=False, mmap_mode=mmap_mode)
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise TypeError("NPY payload is not a safe numeric array")
    if array.dtype.str not in ALLOWED_DTYPES:
        raise TypeError(f"unsupported array dtype: {array.dtype}")
    return array


def inspect_npy(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    array = load_npy_safe(source, mmap_mode="r")
    result = array_summary(array)
    result.update(
        {
            "raw_file_sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
    )
    return result
