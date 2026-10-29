"""Canonical append-only helpers shared by the M7F-H1 workflow."""

from __future__ import annotations

import csv as _csv
from pathlib import Path as _Path

from metachorus_v53.m6f_m7f_fast.common import (  # noqa: F401
    canonical_json_hash,
    envelope,
    read_csv,
    read_json,
    safe_relative,
    sha256_file,
    verify_sha256,
    write_csv_once,
    write_json_once,
    write_text_once,
)


def read_csv_allow_empty(path: str | _Path) -> list[dict[str, str]]:
    """Read a governed CSV whose sealed conditional partition may be empty."""

    source = _Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = _csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV lacks a header: {source}")
        rows = list(reader)
    return rows

__all__ = [name for name in globals() if not name.startswith("_")]
