"""Append-only, zero-GPU correction of the H2 AMiner unsafe result."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import canonical_json_hash, envelope, sha256_file, write_csv_once, write_json_once, write_text_once
from .constants import (
    AMINER_MANUAL_SET_HASH,
    AMINER_MANUAL_SET_ID,
    AMINER_UNSAFE_SET_HASH,
    AMINER_UNSAFE_SET_ID,
    PARENT_AMINER_PACKAGE_FILENAME,
    PARENT_AMINER_PACKAGE_SHA256,
    PARENT_AMINER_PACKAGE_SIZE_BYTES,
    PARENT_H1_PACKAGE_FILENAME,
    PARENT_H1_PACKAGE_SHA256,
    PARENT_H1_PACKAGE_SIZE_BYTES,
)


HISTORICAL_MEMBERS = (
    "AMINER_H2_SELECTED_SET.json",
    "AMINER_H2_SELECTION_SEAL.json",
    "AMINER_H2_FINAL_DECISION.json",
    "AMINER_H2_PRELIM_METRICS_BY_SEED.csv",
    "AMINER_H2_SELECTED_VS_MANUAL.csv",
)


def _verified_zip(path: str | Path, *, filename: str, digest: str, size: int) -> Path:
    raw = Path(path).absolute()
    if raw.is_symlink():
        raise ValueError("H3 parent package cannot be a symlink")
    value = raw.resolve(strict=True)
    if not value.is_file() or value.name != filename:
        raise ValueError("H3 parent package path or filename is invalid")
    if value.stat().st_size != size or sha256_file(value) != digest:
        raise ValueError("H3 parent package hash/size identity mismatch")
    with zipfile.ZipFile(value) as archive:
        if archive.testzip() is not None:
            raise ValueError("H3 parent package CRC check failed")
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise ValueError("H3 parent package has duplicate members")
    return value


def _member_bytes(package: Path, name: str) -> bytes:
    with zipfile.ZipFile(package) as archive:
        matches = [info for info in archive.infolist() if info.filename == name]
        if len(matches) != 1 or matches[0].is_dir() or ((matches[0].external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError(f"H3 parent member is absent or unsafe: {name}")
        return archive.read(matches[0])


def _json_member(package: Path, name: str) -> dict[str, Any]:
    value = json.loads(_member_bytes(package, name).decode("utf-8"))
    basis = value.get("basis", {})
    if not isinstance(basis, dict) or value.get("identity") != canonical_json_hash(basis):
        raise ValueError(f"H3 parent envelope identity is invalid: {name}")
    return value


def _csv_member(package: Path, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(_member_bytes(package, name).decode("utf-8-sig"))))


def _historical_hashes(package: Path) -> dict[str, str]:
    return {name: hashlib.sha256(_member_bytes(package, name)).hexdigest() for name in HISTORICAL_MEMBERS}


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _union_seconds(intervals: Sequence[tuple[datetime, datetime]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[datetime]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("H2 resource registry contains a negative interval")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return sum((end - start).total_seconds() for start, end in merged)


def corrected_resource_rows(registry_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report incremental compute without summing cumulative continuation clocks."""

    rows = [dict(row) for row in registry_rows]
    if not rows:
        raise ValueError("H2 resource registry is empty")
    stage_order = {"CANARY": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "PRELIMINARY": 5}
    rows.sort(key=lambda row: (str(row["physical_run_id"]), stage_order[str(row["logical_stage"])]))
    prior_cumulative: dict[str, float] = {}
    increments: dict[int, float] = {}
    for index, row in enumerate(rows):
        physical = str(row["physical_run_id"])
        cumulative = float(row["model_total_seconds"])
        prior = prior_cumulative.get(physical, 0.0)
        if cumulative < prior:
            
            
            prior = 0.0
        increments[index] = cumulative - prior
        prior_cumulative[physical] = cumulative
    scopes = ["CANARY", "R1", "R2", "R3", "R4", "PRELIMINARY", "TOTAL"]
    output = []
    for scope in scopes:
        selected = [(index, row) for index, row in enumerate(rows) if scope == "TOTAL" or row["logical_stage"] == scope]
        starts = [_iso(str(row["started_at"])) for _index, row in selected]
        ends = [_iso(str(row["ended_at"])) for _index, row in selected]
        per_gpu: defaultdict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
        for _index, row in selected:
            per_gpu[str(row["gpu_uuid"])].append((_iso(str(row["started_at"])), _iso(str(row["ended_at"]))))
        output.append(
            {
                "scope": scope,
                "run_count": len(selected),
                "incremental_model_gpu_hours": sum(increments[index] for index, _row in selected) / 3600.0,
                "sum_of_invocation_wall_hours": sum((end - start).total_seconds() for start, end in zip(starts, ends)) / 3600.0,
                "four_gpu_max_busy_hours": max((_union_seconds(intervals) for intervals in per_gpu.values()), default=0.0) / 3600.0,
                "end_to_end_wall_hours": ((max(ends) - min(starts)).total_seconds() / 3600.0) if starts else 0.0,
                "accounting_rule": "ENDPOINT_INCREMENT_OR_FRESH_INVOCATION; WALL_INTERVALS_FROM_STARTED_AT_ENDED_AT",
                "status": "PASS",
            }
        )
    return output


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def build_aminer_correction(
    *,
    parent_aminer_package: str | Path,
    parent_h1_package: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    result = Path(result_root).resolve(strict=True)
    h2 = _verified_zip(
        parent_aminer_package,
        filename=PARENT_AMINER_PACKAGE_FILENAME,
        digest=PARENT_AMINER_PACKAGE_SHA256,
        size=PARENT_AMINER_PACKAGE_SIZE_BYTES,
    )
    h1 = _verified_zip(
        parent_h1_package,
        filename=PARENT_H1_PACKAGE_FILENAME,
        digest=PARENT_H1_PACKAGE_SHA256,
        size=PARENT_H1_PACKAGE_SIZE_BYTES,
    )
    historical_hashes = _historical_hashes(h2)
    selected = _json_member(h2, "AMINER_H2_SELECTED_SET.json")
    selection = _json_member(h2, "AMINER_H2_SELECTION_SEAL.json")
    decision = _json_member(h2, "AMINER_H2_FINAL_DECISION.json")
    selected_set = selected["basis"]["selected_set"]
    if (
        selected.get("status") != "AMINER_H2_NEW_SET_SELECTED"
        or selected_set.get("candidate_set_id") != AMINER_UNSAFE_SET_ID
        or selected_set.get("candidate_set_hash") != AMINER_UNSAFE_SET_HASH
        or selection.get("status") != "SEALED_BEFORE_PRELIMINARY_EVALUATION"
        or decision.get("status") != "AMINER_FAST_UNSAFE"
        or decision["basis"].get("selected_set", {}).get("candidate_set_hash") != AMINER_UNSAFE_SET_HASH
        or decision["basis"].get("search_reopened") is not False
    ):
        raise ValueError("H2 AMiner unsafe-selection history differs from the correction premise")
    manual_summary = [
        row for row in _csv_member(h2, "AMINER_H2_PRELIM_SUMMARY.csv")
        if row.get("logical_role") == "MANUAL"
    ]
    if len(manual_summary) != 1 or manual_summary[0].get("candidate_set_hash") != AMINER_MANUAL_SET_HASH:
        raise ValueError("H2 AMiner paired Manual summary is not unique")
    manual = manual_summary[0]
    fallback_row = {
        "dataset": "aminer",
        "search_selected_set_id": AMINER_UNSAFE_SET_ID,
        "search_selected_set_hash": AMINER_UNSAFE_SET_HASH,
        "preliminary_decision": "AMINER_FAST_UNSAFE",
        "final_method_role": "MANUAL_FALLBACK",
        "final_set_id": AMINER_MANUAL_SET_ID,
        "final_set_hash": AMINER_MANUAL_SET_HASH,
        "final_paths": [
            "paper_to_author::author_to_paper",
            "paper_to_reference::reference_to_paper",
        ],
        "ten_seed_eligible": False,
        "historical_selection_preserved": True,
    }
    write_csv_once(result / "AMINER_H2_FINAL_FALLBACK_REGISTRY.csv", [fallback_row])
    fallback = envelope(
        "metachorus.v53.m7f-h3.aminer-final-fallback.v1",
        "AMINER_FINAL_FALLBACK_CORRECTED",
        {
            **fallback_row,
            "parent_package_sha256": PARENT_AMINER_PACKAGE_SHA256,
            "parent_selected_set_identity": selected["identity"],
            "parent_selection_seal_identity": selection["identity"],
            "parent_final_decision_identity": decision["identity"],
            "historical_member_sha256": historical_hashes,
            "new_gpu_training_performed": False,
            "scientific_metrics_changed": False,
        },
    )
    write_json_once(result / "AMINER_H2_FINAL_FALLBACK_DECISION.json", fallback)
    write_text_once(
        result / "AMINER_H2_FINAL_FALLBACK_DECISION.md",
        "# AMiner H2 Final Safe-Fallback Correction\n\n"
        f"- Search-selected history remains `{AMINER_UNSAFE_SET_ID}` and `AMINER_FAST_UNSAFE`.\n"
        f"- Final safe method is Manual `{AMINER_MANUAL_SET_ID}`.\n"
        "- This is an append-only zero-GPU correction; no historical scientific artifact was overwritten.\n",
    )
    approval = envelope(
        "metachorus.v53.m7f-h3.aminer-ten-seed-corrected.v1",
        "NOT_ELIGIBLE_UNSAFE",
        {
            "fallback_decision_identity": fallback["identity"],
            "preliminary_decision": "AMINER_FAST_UNSAFE",
            "final_set_id": AMINER_MANUAL_SET_ID,
            "final_set_hash": AMINER_MANUAL_SET_HASH,
            "ten_seed_eligible": False,
            "training_started": False,
            "user_approval_received": False,
        },
    )
    write_json_once(result / "AMINER_10SEED_APPROVAL_STATUS_CORRECTED.json", approval)
    write_csv_once(
        result / "AMINER_10SEED_EXPANSION_MATRIX_CORRECTED.csv",
        [{
            "dataset": "aminer",
            "final_set_id": AMINER_MANUAL_SET_ID,
            "final_set_hash": AMINER_MANUAL_SET_HASH,
            "status": "NOT_ELIGIBLE_UNSAFE",
            "training_started": False,
        }],
    )

    parent_table = _csv_member(h2, "UPDATED_FIVE_DATASET_STATUS_TABLE.csv")
    h1_selector = _csv_member(h1, "IMDB_SELECTOR_SUMMARY.csv")
    imdb_summary = [row for row in h1_selector if row.get("selector") == "FIXED_END_3000" and row.get("path_set_role") == "MANUAL"]
    h1_locked = _csv_member(h1, "IMDB_LOCKED_METRICS_BY_SELECTOR_AND_SEED.csv")
    imdb_identity = [row for row in h1_locked if row.get("selector") == "FIXED_END_3000" and row.get("path_set_role") == "MANUAL"]
    if len(imdb_summary) != 1 or len(imdb_identity) != 3 or len({row["candidate_set_hash"] for row in imdb_identity}) != 1:
        raise ValueError("H1 IMDb fixed-end Manual evidence is incomplete")
    corrected = []
    for original in parent_table:
        dataset = original["dataset"]
        row = {
            "dataset": dataset,
            "method_role": original["method_role"],
            "execution_status": original["execution_status"],
            "search_selection_status": original["selection_status"],
            "preliminary_decision": original["preliminary_decision"],
            "final_freeze_status": "PARENT_FROZEN",
            "Micro_F1": original["Micro_F1"], "Macro_F1": original["Macro_F1"],
            "NMI": original["NMI"], "ARI": original["ARI"],
            "F_cls": original["F_cls"], "F_clu": original["F_clu"], "F_joint": original["F_joint"],
            "model_seed_count": original["model_seed_count"],
            "evidence_level": original["evidence_level"],
        }
        if dataset == "aminer":
            row.update({
                "method_role": "MANUAL",
                "execution_status": "AMINER_H2_FULL_SEARCH_COMPLETE",
                "search_selection_status": "AMINER_H2_NEW_SET_SELECTED",
                "preliminary_decision": "AMINER_FAST_UNSAFE",
                "final_freeze_status": "UNSAFE_NEW_SET_FALLBACK_MANUAL",
                "Micro_F1": manual["mean_Micro_F1"], "Macro_F1": manual["mean_Macro_F1"],
                "NMI": manual["mean_NMI"], "ARI": manual["mean_ARI"],
                "F_cls": manual["mean_F_cls"], "F_clu": manual["mean_F_clu"], "F_joint": manual["mean_F_joint"],
                "model_seed_count": manual["model_seed_count"],
                "evidence_level": "3_SEED_PRELIMINARY_MANUAL_FALLBACK_NONCONFIRMATORY",
            })
        elif dataset == "imdb":
            fixed = imdb_summary[0]
            row.update({
                "method_role": "MANUAL_FIXED_END_3000",
                "search_selection_status": "PARENT_FROZEN",
                "preliminary_decision": "IMDB_MANUAL_FALLBACK_CONFIRMED",
                "final_freeze_status": "MANUAL_FALLBACK",
                "Micro_F1": fixed["Micro_F1_mean"], "Macro_F1": fixed["Macro_F1_mean"],
                "NMI": fixed["NMI_mean"], "ARI": fixed["ARI_mean"],
                "F_cls": fixed["F_cls_mean"], "F_clu": fixed["F_clu_mean"], "F_joint": fixed["F_joint_mean"],
                "model_seed_count": fixed["model_seed_count"],
                "evidence_level": "3_SEED_PRELIMINARY_MANUAL_FIXED_END_3000",
            })
        elif dataset == "dblp":
            row.update({
                "method_role": "MANUAL",
                "execution_status": "MANUAL_PASS",
                "search_selection_status": "M7F_H3_RUNNING",
                "preliminary_decision": "NOT_STARTED",
                "final_freeze_status": "PENDING_M7F_H3",
            })
        corrected.append(row)
    if {row["dataset"] for row in corrected} != {"acm", "freebase", "imdb", "dblp", "aminer"}:
        raise ValueError("H3 corrected five-dataset table is not exact")
    write_csv_once(result / "FIVE_DATASET_STATUS_TABLE_CORRECTED.csv", corrected)
    write_text_once(result / "FIVE_DATASET_STATUS_TABLE_CORRECTED.md", "# Corrected Five-Dataset Status\n\n" + _markdown_table(corrected))

    resource_rows = corrected_resource_rows(_csv_member(h2, "AMINER_H2_RUN_REGISTRY_MINIMAL.csv"))
    write_csv_once(result / "AMINER_H2_RESOURCE_SUMMARY_CORRECTED.csv", resource_rows)
    write_text_once(
        result / "AMINER_H2_RESOURCE_ACCOUNTING_NOTE.md",
        "# AMiner H2 Incremental Resource Accounting\n\n"
        "Continuation stages are accounted by endpoint-clock increments for each physical identity. "
        "Invocation wall time uses `started_at`/`ended_at`; per-GPU busy time uses interval unions; "
        "end-to-end time spans the first start to final end.\n",
    )
    return envelope(
        "metachorus.v53.m7f-h3.aminer-correction-closure.v1",
        "AMINER_FINAL_FALLBACK_CORRECTED",
        {
            "fallback_decision_identity": fallback["identity"],
            "approval_identity": approval["identity"],
            "five_dataset_table_identity": canonical_json_hash(corrected),
            "corrected_resource_identity": canonical_json_hash(resource_rows),
            "historical_member_sha256": historical_hashes,
            "new_gpu_training_performed": False,
        },
    )


__all__ = ["HISTORICAL_MEMBERS", "corrected_resource_rows", "build_aminer_correction"]
