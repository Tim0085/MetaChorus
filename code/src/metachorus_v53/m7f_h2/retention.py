"""Two-phase, append-only rolling cleanup for sealed H2 search states."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import canonical_json_hash, envelope, read_json, sha256_file, write_json_once


def _plain_run_root(search_root: Path, physical_run_id: str) -> Path:
    if not physical_run_id or "/" in physical_run_id or "\\" in physical_run_id or physical_run_id in {".", ".."}:
        raise ValueError("M7F-H2 cleanup received an unsafe physical run id")
    raw = search_root / physical_run_id
    if raw.is_symlink():
        raise ValueError("M7F-H2 cleanup refuses a symlink run root")
    resolved = raw.resolve()
    resolved.relative_to(search_root.resolve())
    return resolved


def _inventory(root: Path, result: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    if root.is_symlink() or not root.is_dir():
        raise ValueError("M7F-H2 cleanup target is not a regular directory")
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("M7F-H2 cleanup target contains a symlink")
        if path.is_file():
            rows.append(
                {
                    "remote_relative_path": path.relative_to(result).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "retention_status": "DELETE_AFTER_SEALED_STAGE_TRANSITION",
                }
            )
    return rows


def cleanup_search_runs(
    *,
    result_root: str | Path,
    logical_stage: str,
    plan_rows: Sequence[Mapping[str, Any]],
    retained_candidate_hashes: set[str],
    stage_evidence_identity: str,
) -> dict[str, Any]:
    if logical_stage not in {"R1", "R3", "R4"} or len(stage_evidence_identity) != 64:
        raise ValueError("M7F-H2 cleanup requires a sealed R1/R3/R4 transition")
    retained = set(map(str, retained_candidate_hashes))
    if not retained:
        raise ValueError("M7F-H2 cleanup cannot remove every candidate")
    result = Path(result_root).resolve(strict=True)
    search_root = result / "search_runs"
    by_id = {str(row["physical_run_id"]): dict(row) for row in plan_rows}
    if len(by_id) != len(plan_rows):
        raise ValueError("M7F-H2 cleanup plan has duplicate physical identities")
    remove_ids = sorted(
        physical_id
        for physical_id, row in by_id.items()
        if str(row["candidate_set_hash"]) not in retained
    )
    plan_path = result / f"AMINER_H2_{logical_stage}_CLEANUP_PLAN.json"
    complete_path = result / f"AMINER_H2_{logical_stage}_CLEANUP_COMPLETE.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        basis = plan.get("basis", {})
        if (
            plan.get("status") != "SEALED_BEFORE_ROLLING_DELETE"
            or plan.get("identity") != canonical_json_hash(basis)
            or basis.get("logical_stage") != logical_stage
            or basis.get("stage_evidence_identity") != stage_evidence_identity
            or basis.get("remove_physical_run_ids") != remove_ids
            or set(map(str, basis.get("retained_candidate_hashes", []))) != retained
        ):
            raise FileExistsError("M7F-H2 existing cleanup plan conflicts")
    else:
        artifact_rows = []
        for physical_id in remove_ids:
            artifact_rows.extend(_inventory(_plain_run_root(search_root, physical_id), result))
        plan = envelope(
            "metachorus.v53.m7f-h2.rolling-cleanup-plan.v1",
            "SEALED_BEFORE_ROLLING_DELETE",
            {
                "logical_stage": logical_stage,
                "stage_evidence_identity": stage_evidence_identity,
                "retained_candidate_hashes": sorted(retained),
                "remove_physical_run_ids": remove_ids,
                "remove_run_count": len(remove_ids),
                "artifact_count": len(artifact_rows),
                "artifact_rows_identity": canonical_json_hash(artifact_rows),
                "artifacts": artifact_rows,
                "parent_result_or_package_deleted": False,
            },
        )
        write_json_once(plan_path, plan)
    for physical_id in remove_ids:
        target = _plain_run_root(search_root, physical_id)
        if target.exists():
            shutil.rmtree(target)
    remaining = [physical_id for physical_id in remove_ids if _plain_run_root(search_root, physical_id).exists()]
    if remaining:
        raise RuntimeError("M7F-H2 rolling cleanup did not remove every sealed target")
    complete = envelope(
        "metachorus.v53.m7f-h2.rolling-cleanup-complete.v1",
        "PASS",
        {
            "logical_stage": logical_stage,
            "cleanup_plan_identity": plan["identity"],
            "removed_run_count": len(remove_ids),
            "removed_artifact_count": int(plan["basis"]["artifact_count"]),
            "retained_candidate_hashes": sorted(retained),
            "parent_result_or_package_deleted": False,
        },
    )
    write_json_once(complete_path, complete)
    return complete


__all__ = ["cleanup_search_runs"]
