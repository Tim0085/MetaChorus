"""Canonical append-only helpers for the M7F-H2 workflow."""

from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.m7f_h1.common import *  # noqa: F401,F403
from metachorus_v53.m6f_m7f_fast.common import write_csv_once as _write_csv_once


def write_csv_once(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write an append-only CSV while deriving a stable union header when safe."""

    values = [dict(row) for row in rows]
    if fieldnames is None:
        if not values:
            raise ValueError("empty H2 CSV requires explicit fieldnames")
        ordered: list[str] = []
        for row in values:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = tuple(ordered)
    return _write_csv_once(path, values, fieldnames=fieldnames)


__all__ = [name for name in globals() if not name.startswith("_")]
