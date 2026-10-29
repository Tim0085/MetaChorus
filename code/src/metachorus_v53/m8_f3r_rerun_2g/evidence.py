"""Read-only adapters for the frozen H5/H7 and H3/H4 evidence packages."""

from __future__ import annotations

import csv
import io
import json
import math
import stat
import statistics
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

from .common import canonical_json_hash, sha256_file
from .constants import PARENT_PACKAGES, SEARCH_MODEL_SEEDS


FREEBASE_MEMBERS = {
    "r1_pool": "evidence/results/TC_LPMSO_H5_R1_POOL_TABLE.csv",
    "metrics": "evidence/results/TC_LPMSO_H7_R4_SEARCH_METRICS.csv",
    "ranking": "evidence/results/TC_LPMSO_H7_R4_ROBUST_RANKING.csv",
    "safety": "evidence/results/TC_LPMSO_H7_R4_MANUAL_SAFETY_AUDIT.csv",
    "inventory": "evidence/results/TC_LPMSO_H7_R4_PARENT_STATE_INVENTORY.csv",
}
DBLP_MEMBERS = {
    "r1_components": "DBLP_H3_R1_POOL_COMPONENTS.csv",
    "objectives": "DBLP_H3_R3_OBJECTIVES.csv",
    "shortlist": "DBLP_H3_R3_SHORTLIST.csv",
    "metrics": "DBLP_H3_R4_METRICS.csv",
    "safety": "DBLP_H3_R4_MANUAL_SAFETY.csv",
    "pareto": "DBLP_H3_R4_PARETO.csv",
}

REQUIRED_PARENT_MEMBERS = {
    "FREEBASE_H5": frozenset({FREEBASE_MEMBERS["r1_pool"]}),
    "FREEBASE_H7_ACM_SAFETY": frozenset(
        {
            FREEBASE_MEMBERS["metrics"],
            FREEBASE_MEMBERS["ranking"],
            FREEBASE_MEMBERS["safety"],
            FREEBASE_MEMBERS["inventory"],
        }
    ),
    "DBLP_H3": frozenset(DBLP_MEMBERS.values()),
}


def _safe_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError("parent ZIP member uses a backslash")
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError("parent ZIP member path is unsafe")
    return value


def _verified_package(path_value: str | Path, *, key: str) -> Path:
    if key not in PARENT_PACKAGES:
        raise KeyError(key)
    expected_name, expected_sha256 = PARENT_PACKAGES[key]
    raw = Path(path_value).absolute()
    if raw.is_symlink():
        raise ValueError(f"parent package cannot be a symlink: {key}")
    path = raw.resolve(strict=True)
    if not path.is_file() or path.name != expected_name or sha256_file(path) != expected_sha256:
        raise ValueError(f"parent package identity mismatch: {key}")
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if archive.testzip() is not None or len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise ValueError(f"parent package CRC/member closure failed: {key}")
        for info in infos:
            _safe_member(info.filename.rstrip("/"))
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"parent package contains a symlink: {key}")
        missing = REQUIRED_PARENT_MEMBERS.get(key, frozenset()) - set(names)
        if missing:
            raise ValueError(f"parent package lacks required evidence: {key}:{sorted(missing)}")
    return path


def _csv_member(path: Path, member: str) -> list[dict[str, str]]:
    _safe_member(member)
    with ZipFile(path) as archive:
        try:
            payload = archive.read(member)
        except KeyError as error:
            raise FileNotFoundError(f"parent package lacks {member}") from error
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        raise ValueError(f"parent CSV lacks a header: {member}")
    return list(reader)


def audit_parent_packages(parent_packages: Mapping[str, str | Path]) -> list[dict[str, Any]]:
    if set(parent_packages) != set(PARENT_PACKAGES):
        missing = sorted(set(PARENT_PACKAGES) - set(parent_packages))
        extra = sorted(set(parent_packages) - set(PARENT_PACKAGES))
        raise ValueError(f"parent package mapping is not exact; missing={missing}, extra={extra}")
    rows = []
    for key in PARENT_PACKAGES:
        path = _verified_package(parent_packages[key], key=key)
        rows.append(
            {
                "parent_role": key,
                "filename": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "crc_check": "PASS",
                "member_path_check": "PASS",
                "symlink_check": "PASS",
                "status": "PASS",
            }
        )
    return rows


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite parent search metric: {field}")
    return result


def _identities_by_hash(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get("candidate_set_hash", ""))
        if len(identity) != 64:
            raise ValueError("parent candidate hash is not SHA-256")
        paths = [str(value) for value in json.loads(str(row["path_ids_json"]))]
        current = {
            "candidate_set_id": str(row.get("candidate_set_id", "")),
            "candidate_set_ordinal": int(row.get("candidate_set_ordinal", -1)),
            "path_ids": paths,
        }
        prior = values.setdefault(identity, current)
        if prior != current:
            raise ValueError("parent candidate identity changed across seeds")
    return values


def _candidate_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_paths = row.get("path_ids", row.get("path_ids_json", "[]"))
    return {
        "candidate_set_id": str(row.get("candidate_set_id", "")),
        "candidate_set_ordinal": int(row.get("candidate_set_ordinal", -1)),
        "path_ids": [str(value) for value in json.loads(str(raw_paths))],
    }


def _freebase_candidates_from_metrics(
    rows: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        identity = str(row.get("candidate_set_hash", ""))
        expected = identities.get(identity)
        if expected is None or expected != {
            "candidate_set_id": str(row.get("candidate_set_id", "")),
            "candidate_set_ordinal": int(row.get("candidate_set_ordinal", -1)),
            "path_ids": list(expected["path_ids"]) if expected is not None else [],
        }:
            raise ValueError("Freebase R4 metric identity does not match the parent inventory")
        if (
            str(row.get("stage", "")) != "R4"
            or int(row.get("update", -1)) != 3000
            or str(row.get("h5_locked_report_access", "")).casefold() != "false"
            or str(row.get("s_eval_access", "")).casefold() != "false"
            or len(str(row.get("representation_sha256", ""))) != 64
        ):
            raise ValueError("Freebase R4 metric row violates the frozen search-only contract")
        grouped.setdefault(identity, []).append(row)
    if set(grouped) != set(identities):
        raise ValueError("Freebase R4 metrics do not cover the complete parent inventory")
    candidates = []
    for identity in sorted(grouped):
        seed_rows = grouped[identity]
        by_seed = {int(row["model_seed"]): row for row in seed_rows}
        if len(seed_rows) != len(by_seed) or set(by_seed) != set(SEARCH_MODEL_SEEDS["freebase"]):
            raise ValueError("Freebase R4 metrics do not contain the frozen three search seeds")
        cls_values = [_finite(by_seed[seed]["mean_F_cls"], "mean_F_cls") for seed in SEARCH_MODEL_SEEDS["freebase"]]
        clu_values = [_finite(by_seed[seed]["mean_F_clu"], "mean_F_clu") for seed in SEARCH_MODEL_SEEDS["freebase"]]
        stability = [
            _finite(by_seed[seed]["assignment_stability_mean"], "assignment_stability_mean")
            for seed in SEARCH_MODEL_SEEDS["freebase"]
        ]
        expected = identities[identity]
        candidates.append(
            {
                "dataset": "freebase",
                "candidate_set_id": expected["candidate_set_id"],
                "candidate_set_hash": identity,
                "candidate_set_ordinal": int(expected["candidate_set_ordinal"]),
                "path_ids": list(expected["path_ids"]),
                "Q_cls": statistics.median(cls_values),
                "Q_clu": statistics.median(clu_values),
                "W_cls": min(cls_values),
                "W_clu": min(clu_values),
                "classification_sd": statistics.stdev(cls_values),
                "assignment_stability": statistics.mean(stability),
                "per_seed": [
                    {
                        "model_seed": seed,
                        "F_cls": cls_values[index],
                        "F_clu": clu_values[index],
                        "assignment_stability": stability[index],
                    }
                    for index, seed in enumerate(SEARCH_MODEL_SEEDS["freebase"])
                ],
                "search_metric_source": "FREEBASE_H7_R4_SEARCH_ONLY",
            }
        )
    return candidates


def load_freebase_selection_inputs(
    *,
    h5_package: str | Path,
    h7_package: str | Path,
) -> dict[str, Any]:
    h5 = _verified_package(h5_package, key="FREEBASE_H5")
    h7 = _verified_package(h7_package, key="FREEBASE_H7_ACM_SAFETY")
    metrics = [row for row in _csv_member(h7, FREEBASE_MEMBERS["metrics"]) if row.get("dataset") == "freebase"]
    ranking = [row for row in _csv_member(h7, FREEBASE_MEMBERS["ranking"]) if row.get("dataset") == "freebase"]
    safety_raw = [row for row in _csv_member(h7, FREEBASE_MEMBERS["safety"]) if row.get("dataset") == "freebase"]
    inventory = [row for row in _csv_member(h7, FREEBASE_MEMBERS["inventory"]) if row.get("dataset") == "freebase"]
    identities = _identities_by_hash(inventory)
    candidates = _freebase_candidates_from_metrics(metrics, identities)
    safety = [
        {
            "dataset": "freebase",
            "candidate_set_id": row["candidate_set_id"],
            "candidate_set_hash": row["candidate_set_hash"],
            "status": row["status"],
        }
        for row in safety_raw
    ]
    safety_by_hash = {str(row["candidate_set_hash"]): row for row in safety_raw}
    if (
        not candidates
        or len(safety_by_hash) != len(safety_raw)
        or set(safety_by_hash) != set(identities)
        or {str(row.get("status", "")) for row in safety_raw} - {"PASS", "FAIL"}
    ):
        raise ValueError("Freebase R4 Manual-safety coverage is incomplete")
    for identity, raw in safety_by_hash.items():
        expected = identities[identity]
        if (
            str(raw.get("candidate_set_id", "")) != expected["candidate_set_id"]
            or int(raw.get("candidate_set_ordinal", -1)) != int(expected["candidate_set_ordinal"])
        ):
            raise ValueError("Freebase Manual-safety identity does not match the parent inventory")
    candidate_by_hash = {row["candidate_set_hash"]: row for row in candidates}
    ranking_by_hash = {str(row["candidate_set_hash"]): row for row in ranking}
    pass_hashes = {
        identity for identity, row in safety_by_hash.items() if str(row.get("status")) == "PASS"
    }
    if len(ranking_by_hash) != len(ranking) or set(ranking_by_hash) != pass_hashes:
        raise ValueError("Freebase robust ranking is not the exact Manual-safety PASS subset")
    for identity, row in ranking_by_hash.items():
        candidate = candidate_by_hash[identity]
        if (
            str(row.get("candidate_set_id", "")) != candidate["candidate_set_id"]
            or int(row.get("candidate_set_ordinal", -1)) != candidate["candidate_set_ordinal"]
            or int(row.get("seed_count", -1)) != len(SEARCH_MODEL_SEEDS["freebase"])
        ):
            raise ValueError("Freebase robust ranking identity differs from the full R4 metric pool")
        for ranking_field, candidate_field in (
            ("Q_cls_R4", "Q_cls"),
            ("Q_clu_R4", "Q_clu"),
            ("W_cls_R4", "W_cls"),
            ("W_clu_R4", "W_clu"),
            ("sample_sd_F_cls_R4", "classification_sd"),
            ("assignment_stability_mean", "assignment_stability"),
        ):
            if not math.isclose(
                _finite(row[ranking_field], ranking_field),
                float(candidate[candidate_field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Freebase robust ranking metrics differ from the full R4 metric pool")
    pool = [row for row in _csv_member(h5, FREEBASE_MEMBERS["r1_pool"]) if row.get("dataset") == "freebase"]
    exploration_only = sorted(
        str(row["candidate_set_hash"])
        for row in pool
        if str(row.get("exploration_member", "")).casefold() == "true"
        and str(row.get("exploitation_member", "")).casefold() != "true"
    )
    basis = {
        "dataset": "freebase",
        "candidate_count": len(candidates),
        "candidate_hashes": sorted(row["candidate_set_hash"] for row in candidates),
        "manual_safety_hashes": sorted(row["candidate_set_hash"] for row in safety),
        "exploration_only_hashes": exploration_only,
        "h5_package_sha256": sha256_file(h5),
        "h7_package_sha256": sha256_file(h7),
        "search_only": True,
    }
    return {"candidates": candidates, "safety": safety, "exploration_only_hashes": exploration_only, "identity": canonical_json_hash(basis), "basis": basis}


def load_dblp_selection_inputs(*, h3_package: str | Path) -> dict[str, Any]:
    h3 = _verified_package(h3_package, key="DBLP_H3")
    objectives = _csv_member(h3, DBLP_MEMBERS["objectives"])
    shortlist = _csv_member(h3, DBLP_MEMBERS["shortlist"])
    pareto = _csv_member(h3, DBLP_MEMBERS["pareto"])
    metrics = _csv_member(h3, DBLP_MEMBERS["metrics"])
    safety_raw = _csv_member(h3, DBLP_MEMBERS["safety"])
    components = _csv_member(h3, DBLP_MEMBERS["r1_components"])
    objective_by_hash = {str(row["candidate_set_hash"]): row for row in objectives}
    shortlist_by_hash = {str(row["candidate_set_hash"]): row for row in shortlist}
    if (
        len(objective_by_hash) != len(objectives)
        or len(shortlist_by_hash) != len(shortlist)
        or not shortlist_by_hash
        or not set(shortlist_by_hash).issubset(objective_by_hash)
    ):
        raise ValueError("DBLP R3 objective/shortlist identity closure is invalid")
    per_hash: dict[str, list[dict[str, Any]]] = {}
    for row in metrics:
        if row.get("dataset") != "dblp" or row.get("metric_role") != "GLOBAL_SEARCH_POOL_ONLY" or str(row.get("prelim_locked_report_access", "")).casefold() not in {"false", "0"}:
            raise ValueError("DBLP R4 metric row is not search-only")
        per_hash.setdefault(str(row["candidate_set_hash"]), []).append(row)
    candidates = []
    for row in pareto:
        identity = str(row["candidate_set_hash"])
        shortlist_row = shortlist_by_hash.get(identity)
        objective_row = objective_by_hash.get(identity)
        if shortlist_row is None or objective_row is None:
            raise ValueError("DBLP R4 candidate is absent from the sealed R3 shortlist")
        expected_identity = _candidate_identity(row)
        if _candidate_identity(shortlist_row) != expected_identity or _candidate_identity(objective_row) != expected_identity:
            raise ValueError("DBLP R3/R4 candidate identity chain changed")
        cohort = per_hash.get(identity, [])
        if len(cohort) != 3 or {int(value["model_seed"]) for value in cohort} != set(SEARCH_MODEL_SEEDS["dblp"]):
            raise ValueError("DBLP R4 row does not contain the frozen three search seeds")
        per_seed = [
            {
                "model_seed": int(value["model_seed"]),
                "F_cls": _finite(value["F_cls"], "F_cls"),
                "F_clu": _finite(value["F_clu"], "F_clu"),
                "assignment_stability": _finite(value["assignment_stability_mean"], "assignment_stability_mean"),
            }
            for value in sorted(cohort, key=lambda value: int(value["model_seed"]))
        ]
        candidates.append(
            {
                "dataset": "dblp",
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": identity,
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "path_ids": [str(value) for value in json.loads(row["path_ids"])],
                "Q_cls": _finite(row["Q_cls"], "Q_cls"),
                "Q_clu": _finite(row["Q_clu"], "Q_clu"),
                "W_cls": _finite(row["W_cls"], "W_cls"),
                "W_clu": _finite(row["W_clu"], "W_clu"),
                "classification_sd": statistics.stdev(value["F_cls"] for value in per_seed),
                "assignment_stability": _finite(row["assignment_stability_mean"], "assignment_stability_mean"),
                "per_seed": per_seed,
                "search_metric_source": "DBLP_H3_R4_SEARCH_ONLY",
            }
        )
    safety = [
        {
            "dataset": "dblp",
            "candidate_set_id": row["candidate_set_id"],
            "candidate_set_hash": row["candidate_set_hash"],
            "status": row["status"],
        }
        for row in safety_raw
    ]
    if not candidates or {row["candidate_set_hash"] for row in candidates} != {row["candidate_set_hash"] for row in safety}:
        raise ValueError("DBLP R4 Manual-safety coverage is incomplete")
    exploration_only = []
    for row in components:
        values = json.loads(row["components"])
        if len(values) == 1 and str(values[0]).startswith("FROZEN_EXPLORATION_BAND_"):
            exploration_only.append(str(row["candidate_set_hash"]))
    basis = {
        "dataset": "dblp",
        "candidate_count": len(candidates),
        "candidate_hashes": sorted(row["candidate_set_hash"] for row in candidates),
        "manual_safety_hashes": sorted(row["candidate_set_hash"] for row in safety),
        "exploration_only_hashes": sorted(exploration_only),
        "h3_package_sha256": sha256_file(h3),
        "search_only": True,
    }
    return {"candidates": candidates, "safety": safety, "exploration_only_hashes": sorted(exploration_only), "identity": canonical_json_hash(basis), "basis": basis}


def existing_safety_rows(parent_audit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_role = {str(row["parent_role"]): row for row in parent_audit_rows}
    mapping = {"acm": "FREEBASE_H7_ACM_SAFETY", "imdb": "IMDB_SAFETY", "aminer": "AMINER_SAFETY"}
    return [
        {
            "dataset": dataset,
            "evidence_type": "EXISTING_FROZEN_EVIDENCE",
            "parent_role": role,
            "parent_filename": by_role[role]["filename"],
            "parent_sha256": by_role[role]["sha256"],
            "new_training": False,
            "status": "PASS",
        }
        for dataset, role in mapping.items()
    ]


__all__ = [
    "audit_parent_packages",
    "load_freebase_selection_inputs",
    "load_dblp_selection_inputs",
    "existing_safety_rows",
]
