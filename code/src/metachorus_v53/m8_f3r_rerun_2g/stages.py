"""Small append-only stage closers for the two-stage remote lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import canonical_json_hash, envelope, read_csv, read_json, write_json_once
from .training import completed_marker, validate_plan


def close_canary(result_root: str | Path) -> dict[str, Any]:
    result = Path(result_root).resolve(strict=True)
    rows = read_csv(result / "M8_F3R_RERUN_2G_CANARY_PLAN.csv")
    seal = read_json(result / "M8_F3R_RERUN_2G_CANARY_PLAN_SEAL.json")
    plan = validate_plan(rows, seal, stage="CANARY")
    markers = []
    for row in plan:
        completed = completed_marker(result, row, "CANARY")
        if completed is None:
            raise RuntimeError("canary fan-out cannot close before all four exact runs pass")
        basis = completed[1]["basis"]
        if (
            basis.get("checkpoint_roundtrip_status") != "PASS"
            or basis.get("labels_mounted_during_training") is not False
            or int(basis.get("downstream_evaluations_during_training", -1)) != 0
        ):
            raise ValueError("canary checkpoint/no-label boundary is invalid")
        markers.append(
            {
                "physical_run_id": row["physical_run_id"],
                "marker_identity": completed[1]["identity"],
                "marker_relative_path": completed[0].relative_to(result).as_posix(),
                "gpu_uuid": basis["gpu_uuid"],
                "checkpoint_roundtrip_status": "PASS",
            }
        )
    basis = {
        "canary_plan_seal_identity": seal["identity"],
        "run_count": len(markers),
        "markers_identity": canonical_json_hash(markers),
        "checkpoint_write_read_pass": True,
        "labels_mounted_during_training": False,
        "scientific_result": False,
        "formal_runs_restart_from_zero": True,
        "fanout_authorized": True,
    }
    value = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.canary-result.v1", "PASS", basis
    )
    write_json_once(result / "M8_F3R_RERUN_2G_CANARY_RESULT.json", value)
    return value


__all__ = ["close_canary"]
