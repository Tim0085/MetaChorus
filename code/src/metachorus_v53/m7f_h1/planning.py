"""Frozen plan construction and existing-evidence inventory for M7F-H1."""

from __future__ import annotations

import csv
from io import StringIO
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence
import zipfile

from metachorus_v53.audit.common import read_json
from metachorus_v53.m6f_m7f_fast.planning import (
    load_cache_resolution as _load_cache_resolution,
)

from .common import (
    canonical_json_hash,
    envelope,
    read_csv,
    safe_relative,
    sha256_file,
)
from .constants import (
    BUDGET_CANARY_CHECKPOINT_SCHEDULE,
    BUDGET_CANARY_REPRESENTATION_SCHEDULE,
    BUDGET_CANARY_UPDATES,
    BUDGET_MODEL_SEEDS,
    EVALUATION_MODEL_SEEDS,
    IMDB_FINAL_SET,
    IMDB_MANUAL_SET,
    MANUAL_CHECKPOINT_SCHEDULE,
    MANUAL_DATASETS,
    MANUAL_REPRESENTATION_SCHEDULE,
    MANUAL_SET_IDENTITIES,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    TRAJECTORY_CHECKPOINT_SCHEDULE,
    TRAJECTORY_REPRESENTATION_SCHEDULE,
    TRAJECTORY_UPDATES,
)


PLAN_FIELDS = (
    "stage",
    "physical_run_id",
    "dataset",
    "path_set_role",
    "cache_identity",
    "path_library_identity",
    "all_sets_identity",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "path_ids",
    "model_seed",
    "model_seed_ordinal",
    "continuation_from_update",
    "endpoint_update",
    "checkpoint_schedule",
    "representation_schedule",
    "training_from_zero",
    "labels_mounted",
    "downstream_evaluations_during_training",
    "eligible_for_scientific_result",
    "eligible_for_selector_development",
    "budget_canary_not_scientific_selection",
    "attempt_index",
    "status",
    "plan_seal_identity",
)


def load_cache_resolution(cache_root: str | Path) -> dict[str, Any]:
    value = _load_cache_resolution(cache_root)
    manifest = read_json(Path(cache_root) / "SANITIZED_CACHE_MANIFEST.json")
    files = manifest.get("files", {})
    members = files.values() if isinstance(files, Mapping) else files
    cache_size_bytes = sum(int(row.get("size_bytes", 0)) for row in members)
    if cache_size_bytes <= 0:
        raise ValueError("frozen cache manifest has no positive member bytes")
    return {**value, "manifest": manifest, "cache_size_bytes": cache_size_bytes}


def _json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        payload = archive.read(name)
    except KeyError as exc:
        raise ValueError(f"parent package lacks {name}") from exc
    value = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"parent member is not a JSON object: {name}")
    return value


def _csv_member(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    try:
        payload = archive.read(name)
    except KeyError as exc:
        raise ValueError(f"parent package lacks {name}") from exc
    return list(csv.DictReader(StringIO(payload.decode("utf-8-sig"))))


def audit_parent_package(path: str | Path) -> dict[str, Any]:
    """Verify the exact M6F/M7F parent without extracting it."""

    source = Path(path)
    if source.name != PARENT_PACKAGE_FILENAME or source.is_symlink() or not source.is_file():
        raise ValueError("M7F-H1 parent package path/name is not exact")
    if source.stat().st_size != PARENT_PACKAGE_SIZE_BYTES or sha256_file(source) != PARENT_PACKAGE_SHA256:
        raise ValueError("M7F-H1 parent package bytes changed")
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError("M7F-H1 parent package CRC failed")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            safe_relative(name).as_posix() != name for name in names
        ):
            raise ValueError("M7F-H1 parent package member paths are unsafe")
        complete = _json_member(archive, "M6F_M7F_FAST_COMPLETE.json")
        git_identity = _json_member(archive, "FAST_GIT_AND_CONFIG_IDENTITY.json")
        registry = _csv_member(archive, "FAST_FINAL_FROZEN_SET_REGISTRY.csv")
        preliminary_matrix = _csv_member(
            archive, "FAST_PRELIMINARY_EXPERIMENT_MATRIX.csv"
        )
    imdb_selected = [
        row for row in registry
        if row.get("dataset") == "imdb"
    ]
    imdb_manual_rows = [
        row for row in preliminary_matrix
        if row.get("dataset") == "imdb"
        and row.get("logical_role") == "MANUAL"
        and row.get("availability") == "AVAILABLE"
    ]
    imdb_manual_identities = {
        (
            row.get("candidate_set_id"),
            row.get("candidate_set_hash"),
            int(row.get("candidate_set_ordinal", -1)),
        )
        for row in imdb_manual_rows
    }
    if (
        complete.get("status") != "FAST_3DATASET_PARTIAL"
        or complete.get("identity") != canonical_json_hash(complete.get("basis", {}))
        or git_identity.get("basis", {}).get("source_commit") != PARENT_SOURCE_COMMIT
        or git_identity.get("basis", {}).get("source_tree") != PARENT_SOURCE_TREE
        or len(imdb_selected) != 1
        or len(imdb_manual_rows) != len(EVALUATION_MODEL_SEEDS)
        or {int(row.get("model_seed", -1)) for row in imdb_manual_rows}
        != set(EVALUATION_MODEL_SEEDS)
        or len(imdb_manual_identities) != 1
    ):
        raise ValueError("M7F-H1 parent identity chain is incomplete")
    selected = imdb_selected[0]
    manual = imdb_manual_rows[0]
    for expected, observed in ((IMDB_FINAL_SET, selected), (IMDB_MANUAL_SET, manual)):
        for key in ("candidate_set_id", "candidate_set_hash", "candidate_set_ordinal"):
            if str(observed.get(key)) != str(expected[key]):
                raise ValueError(f"parent IMDb {key} changed")
    basis = {
        "parent_package_filename": source.name,
        "parent_package_sha256": PARENT_PACKAGE_SHA256,
        "parent_package_size_bytes": PARENT_PACKAGE_SIZE_BYTES,
        "parent_complete_identity": complete["identity"],
        "parent_git_identity": git_identity["identity"],
        "parent_source_commit": PARENT_SOURCE_COMMIT,
        "parent_source_tree": PARENT_SOURCE_TREE,
        "parent_final_registry_rows_identity": canonical_json_hash(registry),
        "parent_preliminary_matrix_rows_identity": canonical_json_hash(
            preliminary_matrix
        ),
        "imdb_final_set": dict(IMDB_FINAL_SET),
        "imdb_manual_set": dict(IMDB_MANUAL_SET),
    }
    return envelope(
        "metachorus.v53.m7f-h1.parent-package-audit.v1", "PASS", basis
    )


def _candidate(cache: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row for row in cache["all_sets"]["sets"]
        if row.get("candidate_set_id") == expected["candidate_set_id"]
        and row.get("candidate_set_hash") == expected["candidate_set_hash"]
        and int(row.get("candidate_set_ordinal", -1))
        == int(expected["candidate_set_ordinal"])
    ]
    if len(rows) != 1:
        raise ValueError("frozen candidate is missing or duplicated in all_sets")
    return dict(rows[0])


def total_hop_count(candidate: Mapping[str, Any]) -> int:
    paths = candidate.get("path_ids", [])
    if not isinstance(paths, list) or not paths:
        raise ValueError("candidate path_ids are not explicit")
    return sum(len(str(path).split("::")) for path in paths)


def select_budget_candidates(cache: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    dataset = str(cache["dataset"])
    if dataset not in MANUAL_DATASETS:
        raise ValueError("budget canaries are defined only for DBLP and AMiner")
    manual = _candidate(cache, MANUAL_SET_IDENTITIES[dataset])
    non_manual = [
        dict(row) for row in cache["all_sets"]["sets"]
        if row.get("candidate_set_hash") != manual["candidate_set_hash"]
    ]
    if not non_manual:
        raise ValueError("budget canary requires non-Manual candidates")
    minimum = min(total_hop_count(row) for row in non_manual)
    maximum = max(total_hop_count(row) for row in non_manual)
    shortest = min(
        (row for row in non_manual if total_hop_count(row) == minimum),
        key=lambda row: str(row["candidate_set_hash"]),
    )
    longest = min(
        (row for row in non_manual if total_hop_count(row) == maximum),
        key=lambda row: str(row["candidate_set_hash"]),
    )
    return {
        "MANUAL": {**manual, "total_hop_count": total_hop_count(manual)},
        "SHORTEST": {**shortest, "total_hop_count": minimum},
        "LONGEST": {**longest, "total_hop_count": maximum},
    }


def _base_row(
    cache: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    stage: str,
    physical_run_id: str,
    path_set_role: str,
    seed: int,
    seed_ordinal: int,
    endpoint: int,
    checkpoints: Sequence[int],
    representations: Sequence[int],
    scientific: bool,
    selector: bool,
    budget_canary: bool,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "physical_run_id": physical_run_id,
        "dataset": cache["dataset"],
        "path_set_role": path_set_role,
        "cache_identity": cache["cache_identity"],
        "path_library_identity": cache["path_library_identity"],
        "all_sets_identity": cache["all_sets_identity"],
        "candidate_set_id": candidate["candidate_set_id"],
        "candidate_set_hash": candidate["candidate_set_hash"],
        "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
        "path_ids": list(candidate["path_ids"]),
        "model_seed": int(seed),
        "model_seed_ordinal": int(seed_ordinal),
        "continuation_from_update": 0,
        "endpoint_update": int(endpoint),
        "checkpoint_schedule": list(map(int, checkpoints)),
        "representation_schedule": list(map(int, representations)),
        "training_from_zero": True,
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "eligible_for_scientific_result": bool(scientific),
        "eligible_for_selector_development": bool(selector),
        "budget_canary_not_scientific_selection": bool(budget_canary),
        "attempt_index": 0,
        "status": "PENDING_GPU",
        "plan_seal_identity": "",
    }


def _seal_plan(
    rows: Sequence[Mapping[str, Any]], *, stage: str, schema: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unsealed = [
        {key: value for key, value in dict(row).items() if key != "plan_seal_identity"}
        for row in rows
    ]
    basis = {
        "stage": stage,
        "row_count": len(unsealed),
        "rows_identity": canonical_json_hash(unsealed),
        "physical_run_ids_unique": len({row["physical_run_id"] for row in unsealed})
        == len(unsealed),
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "training_from_zero": True,
        "maximum_gpu_count": 2,
    }
    if not unsealed or basis["physical_run_ids_unique"] is not True:
        raise ValueError("M7F-H1 plan is empty or has duplicate physical identities")
    seal = envelope(schema, "SEALED_BEFORE_GPU_TRAINING", basis)
    return [
        {**dict(row), "plan_seal_identity": seal["identity"]} for row in rows
    ], seal


def build_imdb_trajectory_plan(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cache.get("dataset") != "imdb":
        raise ValueError("IMDb trajectory plan requires the IMDb cache")
    candidates = (
        ("FINAL_FROZEN_METHOD", _candidate(cache, IMDB_FINAL_SET)),
        ("MANUAL", _candidate(cache, IMDB_MANUAL_SET)),
    )
    rows: list[dict[str, Any]] = []
    for role, candidate in candidates:
        for ordinal, seed in enumerate(EVALUATION_MODEL_SEEDS):
            rows.append(
                _base_row(
                    cache,
                    candidate,
                    stage="IMDB_TRAJECTORY",
                    physical_run_id=(
                        "M7F-H1-IMDB-TRAJECTORY-"
                        f"C{int(candidate['candidate_set_ordinal']):03d}-E{ordinal:02d}"
                    ),
                    path_set_role=role,
                    seed=seed,
                    seed_ordinal=ordinal,
                    endpoint=TRAJECTORY_UPDATES,
                    checkpoints=TRAJECTORY_CHECKPOINT_SCHEDULE,
                    representations=TRAJECTORY_REPRESENTATION_SCHEDULE,
                    scientific=True,
                    selector=True,
                    budget_canary=False,
                )
            )
    return _seal_plan(
        rows,
        stage="IMDB_TRAJECTORY",
        schema="metachorus.v53.m7f-h1.imdb-trajectory-plan-seal.v1",
    )


def build_manual_plan(caches: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(caches) != set(MANUAL_DATASETS):
        raise ValueError("Manual plan requires exactly DBLP and AMiner caches")
    rows: list[dict[str, Any]] = []
    for dataset in MANUAL_DATASETS:
        cache = caches[dataset]
        candidate = _candidate(cache, MANUAL_SET_IDENTITIES[dataset])
        token = dataset.upper()
        for ordinal, seed in enumerate(EVALUATION_MODEL_SEEDS):
            rows.append(
                _base_row(
                    cache,
                    candidate,
                    stage="MANUAL_BASELINE",
                    physical_run_id=(
                        f"M7F-H1-MANUAL-{token}-"
                        f"C{int(candidate['candidate_set_ordinal']):03d}-E{ordinal:02d}"
                    ),
                    path_set_role="MANUAL",
                    seed=seed,
                    seed_ordinal=ordinal,
                    endpoint=TRAJECTORY_UPDATES,
                    checkpoints=MANUAL_CHECKPOINT_SCHEDULE,
                    representations=MANUAL_REPRESENTATION_SCHEDULE,
                    scientific=True,
                    selector=False,
                    budget_canary=False,
                )
            )
    return _seal_plan(
        rows,
        stage="MANUAL_BASELINE",
        schema="metachorus.v53.m7f-h1.manual-plan-seal.v1",
    )


def build_budget_canary_plan(caches: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(caches) != set(MANUAL_DATASETS):
        raise ValueError("budget canary plan requires exactly DBLP and AMiner caches")
    rows: list[dict[str, Any]] = []
    for dataset in MANUAL_DATASETS:
        cache = caches[dataset]
        token = dataset.upper()
        for role, candidate in select_budget_candidates(cache).items():
            path_role = "MANUAL" if role == "MANUAL" else f"BUDGET_{role}"
            for ordinal, seed in enumerate(BUDGET_MODEL_SEEDS):
                rows.append(
                    _base_row(
                        cache,
                        candidate,
                        stage="BUDGET_CANARY",
                        physical_run_id=(
                            f"M7F-H1-BUDGET-{token}-{role}-"
                            f"C{int(candidate['candidate_set_ordinal']):03d}-Q{ordinal:02d}"
                        ),
                        path_set_role=path_role,
                        seed=seed,
                        seed_ordinal=ordinal,
                        endpoint=BUDGET_CANARY_UPDATES,
                        checkpoints=BUDGET_CANARY_CHECKPOINT_SCHEDULE,
                        representations=BUDGET_CANARY_REPRESENTATION_SCHEDULE,
                        scientific=False,
                        selector=False,
                        budget_canary=True,
                    )
                )
    return _seal_plan(
        rows,
        stage="BUDGET_CANARY",
        schema="metachorus.v53.m7f-h1.budget-canary-plan-seal.v1",
    )


def validate_plan(
    rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any], *, stage: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        for key in (
            "candidate_set_ordinal",
            "model_seed",
            "model_seed_ordinal",
            "continuation_from_update",
            "endpoint_update",
            "downstream_evaluations_during_training",
            "attempt_index",
        ):
            row[key] = int(row[key])
        for key in ("path_ids", "checkpoint_schedule", "representation_schedule"):
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
            row[key] = list(row[key])
        for key in (
            "training_from_zero",
            "labels_mounted",
            "eligible_for_scientific_result",
            "eligible_for_selector_development",
            "budget_canary_not_scientific_selection",
        ):
            if isinstance(row.get(key), str):
                row[key] = row[key].casefold() == "true"
            else:
                row[key] = bool(row[key])
        normalized.append(row)
    basis = seal.get("basis", {})
    unsealed = [
        {key: value for key, value in row.items() if key != "plan_seal_identity"}
        for row in normalized
    ]
    if (
        seal.get("status") != "SEALED_BEFORE_GPU_TRAINING"
        or seal.get("identity") != canonical_json_hash(basis)
        or basis.get("stage") != stage
        or int(basis.get("row_count", -1)) != len(normalized)
        or basis.get("rows_identity") != canonical_json_hash(unsealed)
        or any(row.get("stage") != stage for row in normalized)
        or any(row.get("plan_seal_identity") != seal.get("identity") for row in normalized)
        or any(row.get("labels_mounted") is not False for row in normalized)
        or any(int(row.get("downstream_evaluations_during_training", -1)) != 0 for row in normalized)
    ):
        raise ValueError("M7F-H1 plan/seal identity mismatch")
    return normalized


def inventory_existing_sources(
    *,
    remote_result_root: str | Path,
    retention_manifest: str | Path,
    expected_physical_run_ids: Sequence[str],
) -> tuple[str, list[dict[str, Any]]]:
    """Inventory only paths declared by the parent retention manifest."""

    def parent_run_identity(
        run_id: str,
    ) -> tuple[str, Mapping[str, Any], int]:
        matched_id = re.fullmatch(
            r"M7F-H1-IMDB-TRAJECTORY-C(\d{3})-E(0[0-2])", str(run_id)
        )
        if matched_id is None:
            raise ValueError("existing IMDb evidence identity is not canonical")
        candidate_ordinal = int(matched_id.group(1))
        candidates = {
            int(IMDB_MANUAL_SET["candidate_set_ordinal"]): IMDB_MANUAL_SET,
            int(IMDB_FINAL_SET["candidate_set_ordinal"]): IMDB_FINAL_SET,
        }
        if candidate_ordinal not in candidates:
            raise ValueError("existing IMDb evidence candidate identity is not frozen")
        seed_ordinal = int(matched_id.group(2))
        return (
            f"M7F-FAST-IMDB-C{matched_id.group(1)}-E{matched_id.group(2)}",
            candidates[candidate_ordinal],
            int(EVALUATION_MODEL_SEEDS[seed_ordinal]),
        )

    root = Path(remote_result_root)
    if not root.is_dir() or root.is_symlink():
        return "IMDB_REMOTE_RESULT_ROOT_NOT_FOUND", [
            {
                "physical_run_id": run_id,
                "source_physical_run_id": parent_run_identity(str(run_id))[0],
                "remote_relative_path": "",
                "sha256": "",
                "size_bytes": 0,
                "row_count": 0,
                "columns": [],
                "covered_updates": [],
                "required_updates_present": False,
                "evidence_type": "MISSING_REMOTE_ROOT",
                "usable": False,
                "status": "IMDB_REMOTE_RESULT_ROOT_NOT_FOUND",
            }
            for run_id in expected_physical_run_ids
        ]
    rows = read_csv(retention_manifest)
    inventory: list[dict[str, Any]] = []
    for run_id in expected_physical_run_ids:
        source_run_id, expected_candidate, expected_seed = parent_run_identity(
            str(run_id)
        )
        matched = [
            row for row in rows
            if source_run_id
            in PurePosixPath(row.get("remote_relative_path", "")).parts
            and row.get("remote_relative_path", "").endswith((".csv", ".npy", ".json"))
        ]
        if not matched:
            inventory.append(
                {
                    "physical_run_id": run_id,
                    "source_physical_run_id": source_run_id,
                    "remote_relative_path": "",
                    "sha256": "",
                    "size_bytes": 0,
                    "row_count": 0,
                    "columns": [],
                    "covered_updates": [],
                    "required_updates_present": False,
                    "evidence_type": "MISSING",
                    "usable": False,
                    "status": "EXISTING_TRAJECTORY_INCOMPLETE",
                }
            )
            continue
        for declared in matched:
            relative = safe_relative(declared["remote_relative_path"])
            source = root / Path(*relative.parts)
            cursor = root
            symlink_chain = False
            for part in relative.parts:
                cursor /= part
                if cursor.is_symlink():
                    symlink_chain = True
                    break
            expected_hash = declared.get("sha256", "")
            usable = (
                source.is_file()
                and not symlink_chain
                and sha256_file(source) == expected_hash
            )
            columns: list[str] = []
            row_count = 0
            covered_updates: set[int] = set()
            if usable and source.suffix.casefold() == ".csv":
                observed = read_csv(source)
                row_count = len(observed)
                columns = list(observed[0]) if observed else []
                required_columns = {
                    "dataset", "candidate_set_id", "candidate_set_hash",
                    "model_seed", "Micro_F1", "Macro_F1", "NMI", "ARI",
                    "F_cls", "F_clu", "F_joint",
                }
                update_columns = [
                    name for name in ("update", "endpoint_update") if name in columns
                ]
                if (
                    not observed
                    or not required_columns.issubset(columns)
                    or not update_columns
                ):
                    usable = False
                else:
                    try:
                        seen_updates: set[int] = set()
                        for metric_row in observed:
                            updates = {
                                int(metric_row[name])
                                for name in update_columns
                            }
                            if len(updates) != 1:
                                raise ValueError("metric update columns disagree")
                            if metric_row.get("dataset") != "imdb":
                                raise ValueError("metric row is not IMDb")
                            update = next(iter(updates))
                            if update <= 0 or update > TRAJECTORY_UPDATES or update in seen_updates:
                                raise ValueError("metric update identity is invalid")
                            seen_updates.add(update)
                            if (
                                metric_row.get("candidate_set_id")
                                != expected_candidate["candidate_set_id"]
                                or metric_row.get("candidate_set_hash")
                                != expected_candidate["candidate_set_hash"]
                                or int(metric_row["model_seed"]) != expected_seed
                            ):
                                raise ValueError("metric candidate/seed identity changed")
                            metric_values = {
                                name: float(metric_row[name])
                                for name in (
                                    "Micro_F1", "Macro_F1", "NMI", "ARI",
                                    "F_cls", "F_clu", "F_joint",
                                )
                            }
                            if not all(math.isfinite(value) for value in metric_values.values()):
                                raise ValueError("metric row contains a non-finite value")
                            if (
                                not math.isclose(
                                    metric_values["F_cls"],
                                    (metric_values["Micro_F1"] + metric_values["Macro_F1"]) / 2.0,
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                )
                                or not math.isclose(
                                    metric_values["F_clu"],
                                    (metric_values["NMI"] + metric_values["ARI"]) / 2.0,
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                )
                                or not math.isclose(
                                    metric_values["F_joint"],
                                    (metric_values["F_cls"] + metric_values["F_clu"]) / 2.0,
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                )
                            ):
                                raise ValueError("metric formulas changed")
                            covered_updates.update(updates)
                    except (KeyError, TypeError, ValueError):
                        usable = False
                        covered_updates.clear()
            elif usable and source.suffix.casefold() == ".npy":
                update_match = re.search(r"update_(\d{6})\.npy$", relative.as_posix())
                if update_match is not None:
                    covered_updates.add(int(update_match.group(1)))
            required_present = set((300, 1000, 2000, 3000)).issubset(covered_updates)
            inventory.append(
                {
                    "physical_run_id": run_id,
                    "source_physical_run_id": source_run_id,
                    "remote_relative_path": relative.as_posix(),
                    "sha256": expected_hash,
                    "size_bytes": source.stat().st_size if source.is_file() else 0,
                    "row_count": row_count,
                    "columns": columns,
                    "covered_updates": sorted(covered_updates),
                    "required_updates_present": required_present,
                    "evidence_type": (
                        "EXISTING_METRIC" if source.suffix.casefold() == ".csv"
                        else "EVALUABLE_REPRESENTATION"
                    ),
                    "usable": usable,
                    "status": "PASS" if usable else "EXISTING_TRAJECTORY_INCOMPLETE",
                }
            )
    coverage = {
        str(run_id): {
            update
            for row in inventory
            if row["physical_run_id"] == run_id and row["usable"]
            for update in row["covered_updates"]
        }
        for run_id in expected_physical_run_ids
    }
    complete = all(
        set((300, 1000, 2000, 3000)).issubset(coverage[str(run_id)])
        for run_id in expected_physical_run_ids
    )
    status = (
        "IMDB_EXISTING_TRAJECTORY_COMPLETE"
        if complete
        else "EXISTING_TRAJECTORY_INCOMPLETE"
    )
    return status, inventory


__all__ = [
    "PLAN_FIELDS",
    "audit_parent_package",
    "total_hop_count",
    "select_budget_candidates",
    "build_imdb_trajectory_plan",
    "build_manual_plan",
    "build_budget_canary_plan",
    "validate_plan",
    "inventory_existing_sources",
    "load_cache_resolution",
]
