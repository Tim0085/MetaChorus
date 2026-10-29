"""Bind the immutable pre-H5 3000-update development truth landscape."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from metachorus_v53.audit.common import atomic_write_json, canonical_json_hash, sha256_file


ACM_TRUTH_FILENAME = "ACM_DEVELOPMENT_TRUTH_AGGREGATE_M4A_H2.csv"
ACM_TRUTH_SHA256 = "b1248f434cbac50b46ce870fef286d5a9fb3ab118baccb290b80a14b396f9ffb"
FREEBASE_TRUTH_FILENAME = "FREEBASE_LP_SRP_C_HOLDOUT_AUDIT.csv"
FREEBASE_TRUTH_SHA256 = "ebb5d0fb992dd38d63ae57a131fc1a6e1e717733f27d69005a6b67cdd44f35a3"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def bind_development_truth(*, h2_result_root: Path, result_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Standardize historical truth into monotone order scores without reading H5 labels."""

    acm_path = h2_result_root / ACM_TRUTH_FILENAME
    freebase_path = h2_result_root / FREEBASE_TRUTH_FILENAME
    if sha256_file(acm_path) != ACM_TRUTH_SHA256:
        raise ValueError("ACM 3000-update development truth hash mismatch")
    if sha256_file(freebase_path) != FREEBASE_TRUTH_SHA256:
        raise ValueError("Freebase 3000-update development truth hash mismatch")
    acm = _read_csv(acm_path)
    freebase = _read_csv(freebase_path)
    if len(acm) != 10 or len(freebase) != 220:
        raise ValueError(f"development truth candidate coverage mismatch: ACM={len(acm)}, Freebase={len(freebase)}")
    rows: list[dict[str, Any]] = []
    for row in acm:
        rows.append(
            {
                "dataset": "acm",
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "classification_rank": float(row["classification_rank"]),
                "clustering_rank": float(row["clustering_rank"]),
                "joint_rank": float(row["joint_rank"]),
                "classification_truth_order_score": float(row["classification_score"]),
                "clustering_truth_order_score": float(row["clustering_score"]),
                "joint_truth_order_score": -float(row["joint_rank"]),
                "truth_training_budget": 3000,
                "truth_source": ACM_TRUTH_FILENAME,
            }
        )
    for row in freebase:
        rows.append(
            {
                "dataset": "freebase",
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "classification_rank": float(row["classification_rank"]),
                "clustering_rank": float(row["clustering_rank"]),
                "joint_rank": float(row["joint_rank"]),
                "classification_truth_order_score": -float(row["classification_rank"]),
                "clustering_truth_order_score": -float(row["clustering_rank"]),
                "joint_truth_order_score": -float(row["joint_rank"]),
                "truth_training_budget": 3000,
                "truth_source": FREEBASE_TRUTH_FILENAME,
            }
        )
    if len({(row["dataset"], row["candidate_set_hash"]) for row in rows}) != 230:
        raise ValueError("development truth contains duplicate candidate identities")
    basis = {
        "acm": {
            "filename": ACM_TRUTH_FILENAME,
            "sha256": ACM_TRUTH_SHA256,
            "candidate_count": 10,
            "training_budget": 3000,
            "classification_truth_basis": "historical mean F_cls",
            "clustering_truth_basis": "historical mean F_clu",
            "joint_truth_basis": "negative corrected joint midrank",
        },
        "freebase": {
            "filename": FREEBASE_TRUTH_FILENAME,
            "sha256": FREEBASE_TRUTH_SHA256,
            "candidate_count": 220,
            "training_budget": 3000,
            "classification_truth_basis": "negative historical classification midrank",
            "clustering_truth_basis": "negative historical clustering midrank",
            "joint_truth_basis": "negative historical joint midrank",
        },
        "historical_pre_h5_development_truth": True,
        "h5_locked_report_opened": False,
        "raw_h5_labels_opened": False,
        "allowed_roles": ["post_pool_seal_recall_audit", "R2_fidelity", "R3_fidelity"],
        "forbidden_roles": ["exploration_sampling", "pool_replacement", "R4_selection"],
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h5.development-truth-binding.v1",
        "status": "BOUND_IMMUTABLE_PRE_H5_TRUTH",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }
    binding_path = result_root / "TC_LPMSO_H5_3000_TRUTH_BINDING.json"
    if binding_path.is_file():
        import json

        observed = json.loads(binding_path.read_text(encoding="utf-8"))
        if observed.get("identity") != payload["identity"]:
            raise RuntimeError("append-only H5 truth binding identity changed")
    else:
        atomic_write_json(binding_path, payload)
    return rows, payload
