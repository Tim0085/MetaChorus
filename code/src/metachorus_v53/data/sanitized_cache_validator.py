"""Recursive, fail-closed validation for the frozen sanitized-cache schema."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from metachorus_v53.audit.common import (
    canonical_json_hash,
    iter_files,
    load_checksums,
    read_json,
    sha256_file,
)
from .array_identity import inspect_npy
from .sanitized_cache_schema import (
    FORBIDDEN_SUFFIXES,
    SCHEMA_VERSION,
    allowed_static_files,
    assert_safe_relative_path,
    recursive_forbidden_scan,
)


def _manifest_file_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("manifest.files must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
            raise ValueError("manifest file row is invalid")
        relative = row["relative_path"]
        assert_safe_relative_path(relative)
        if relative in result:
            raise ValueError(f"duplicate manifest path: {relative}")
        result[relative] = row
    return result


def validate_sanitized_cache(
    cache_root: str | Path,
    *,
    label_array_identity: Mapping[str, Any] | None,
    require_complete: bool = True,
) -> dict[str, Any]:
    root = Path(cache_root).resolve()
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        manifest = read_json(root / "SANITIZED_CACHE_MANIFEST.json")
        complete = read_json(root / "SANITIZED_CACHE_COMPLETE.json")
        meta_path_schema = read_json(root / "schema" / "meta_path_schema.json")
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "metachorus.sanitized-cache-validation.v1",
            "status": "FAIL",
            "failures": [{"reason": "required_control_file_invalid", "error": str(error)}],
            "warnings": [],
        }
    if manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append({"reason": "schema_version_mismatch"})
    projection_files = {
        str(row.get("array"))
        for row in meta_path_schema.get("paths", [])
        if isinstance(row, dict)
    }
    try:
        for relative in projection_files:
            assert_safe_relative_path(relative)
    except ValueError as error:
        failures.append({"reason": "unsafe_projection_path", "error": str(error)})
    allowed = allowed_static_files() | projection_files
    observed: set[str] = set()
    json_findings: list[dict[str, str]] = []
    object_dtype_count = 0
    symlink_count = 0
    hardlink_count = 0
    unknown_files: list[str] = []
    arrays: dict[str, dict[str, Any]] = {}
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        observed.add(relative)
        if path.is_symlink():
            symlink_count += 1
            failures.append({"reason": "symlink_forbidden", "path": relative})
            continue
        try:
            if os.stat(path, follow_symlinks=False).st_nlink > 1:
                hardlink_count += 1
                failures.append({"reason": "hardlink_forbidden", "path": relative})
        except OSError as error:
            failures.append(
                {"reason": "link_audit_failed", "path": relative, "error": str(error)}
            )
        try:
            assert_safe_relative_path(relative)
        except ValueError as error:
            failures.append(
                {"reason": "unsafe_relative_path", "path": relative, "error": str(error)}
            )
        if relative not in allowed:
            unknown_files.append(relative)
            failures.append({"reason": "unknown_file", "path": relative})
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            failures.append({"reason": "forbidden_serialization", "path": relative})
        if path.suffix.casefold() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                json_findings.extend(
                    {
                        **finding,
                        "file": relative,
                    }
                    for finding in recursive_forbidden_scan(value)
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                failures.append(
                    {"reason": "invalid_json", "path": relative, "error": str(error)}
                )
        if path.suffix.casefold() == ".npy":
            try:
                arrays[relative] = inspect_npy(path)
            except (TypeError, ValueError, OSError) as error:
                if "object" in str(error).casefold():
                    object_dtype_count += 1
                failures.append(
                    {"reason": "unsafe_npy", "path": relative, "error": str(error)}
                )
    if json_findings:
        failures.append(
            {
                "reason": "forbidden_json_information",
                "count": len(json_findings),
                "examples": json_findings[:10],
            }
        )
    missing = sorted(allowed - observed)
    if missing:
        failures.append({"reason": "required_files_missing", "paths": missing})
    if require_complete and complete.get("status") != "COMPLETE":
        failures.append({"reason": "completion_marker_invalid"})
    try:
        manifest_rows = _manifest_file_rows(manifest)
    except ValueError as error:
        manifest_rows = {}
        failures.append({"reason": "manifest_structure_invalid", "error": str(error)})
    expected_payload = allowed - {
        "SANITIZED_CACHE_MANIFEST.json",
        "SANITIZED_CACHE_CHECKSUMS.sha256",
        "SANITIZED_CACHE_COMPLETE.json",
    }
    if set(manifest_rows) != expected_payload:
        failures.append(
            {
                "reason": "manifest_payload_set_mismatch",
                "missing": sorted(expected_payload - set(manifest_rows)),
                "unexpected": sorted(set(manifest_rows) - expected_payload),
            }
        )
    for relative, row in manifest_rows.items():
        path = root / relative
        if not path.is_file():
            continue
        if row.get("raw_file_sha256") != sha256_file(path):
            failures.append({"reason": "manifest_raw_hash_mismatch", "path": relative})
        if row.get("size_bytes") != path.stat().st_size:
            failures.append({"reason": "manifest_size_mismatch", "path": relative})
        if relative.endswith(".npy"):
            observed_array = arrays.get(relative)
            if observed_array is None:
                continue
            for field in ("shape", "dtype", "semantic_sha256"):
                if row.get(field) != observed_array.get(field):
                    failures.append(
                        {
                            "reason": f"manifest_array_{field}_mismatch",
                            "path": relative,
                        }
                    )
    try:
        checksums = load_checksums(root / "SANITIZED_CACHE_CHECKSUMS.sha256")
        expected_checksum_paths = expected_payload | {"SANITIZED_CACHE_MANIFEST.json"}
        if set(checksums) != expected_checksum_paths:
            failures.append({"reason": "checksum_path_set_mismatch"})
        for relative, digest in checksums.items():
            if not (root / relative).is_file() or sha256_file(root / relative) != digest:
                failures.append({"reason": "checksum_mismatch", "path": relative})
    except (OSError, ValueError) as error:
        failures.append({"reason": "checksum_file_invalid", "error": str(error)})
    identity_basis = manifest.get("cache_identity_basis")
    if (
        not isinstance(identity_basis, dict)
        or canonical_json_hash(identity_basis) != manifest.get("cache_identity")
    ):
        failures.append({"reason": "cache_identity_not_rebuildable"})
    if complete.get("manifest_sha256") != sha256_file(
        root / "SANITIZED_CACHE_MANIFEST.json"
    ):
        failures.append({"reason": "complete_manifest_hash_mismatch"})
    if complete.get("checksums_sha256") != sha256_file(
        root / "SANITIZED_CACHE_CHECKSUMS.sha256"
    ):
        failures.append({"reason": "complete_checksums_hash_mismatch"})
    collision_count = 0
    if label_array_identity is not None:
        for relative, summary in arrays.items():
            if (
                summary.get("shape") == label_array_identity.get("shape")
                and summary.get("semantic_sha256")
                == label_array_identity.get("semantic_sha256")
            ):
                collision_count += 1
                failures.append(
                    {"reason": "label_semantic_hash_collision", "path": relative}
                )
    return {
        "schema_version": "metachorus.sanitized-cache-validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "cache_identity": manifest.get("cache_identity"),
        "file_count": len(observed),
        "array_count": len(arrays),
        "unknown_files": unknown_files,
        "unknown_file_count": len(unknown_files),
        "forbidden_json_finding_count": len(json_findings),
        "object_dtype_count": object_dtype_count,
        "symlink_count": symlink_count,
        "hardlink_count": hardlink_count,
        "label_semantic_hash_collision_count": collision_count,
        "failures": failures,
        "warnings": warnings,
    }
