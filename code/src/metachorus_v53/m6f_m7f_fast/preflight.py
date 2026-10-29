"""M6F exact resolver and pre-GPU closure writer."""

from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile

from metachorus_v53.audit.common import canonical_json_hash, sha256_file

from .adapter import FastSanitizedAdapter, adapter_compliance
from .common import envelope, write_csv_once, write_json_once, write_text_once
from .constants import (
    DATASET_IDENTITIES,
    DATASET_NAMES,
    H8_METHOD_FREEZE_IDENTITY,
    H8_PATH_REGISTRY_IDENTITY,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    PARENT_TERMINAL_STATUS,
    RESOURCE_LIMITS,
    SEED_SEAL_IDENTITY,
    SEED_ROLE_PUBLIC_ARTIFACTS,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
)
from .dataset_specs import DATASET_SPECS
from .planning import build_canary_plan, build_search_plan, load_cache_resolution


def _clean_source(repo_root: Path) -> dict[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise RuntimeError("M6F/M7F source worktree must be clean and committed")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    return {"source_commit": commit, "source_tree": tree}


def _parent_audit(parent_package: Path) -> dict[str, Any]:
    if (
        parent_package.name != PARENT_PACKAGE_FILENAME
        or parent_package.stat().st_size != PARENT_PACKAGE_SIZE_BYTES
        or sha256_file(parent_package) != PARENT_PACKAGE_SHA256
    ):
        raise ValueError("H8 parent package file identity mismatch")
    with ZipFile(parent_package, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("H8 parent ZIP CRC failure")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("H8 parent ZIP member paths are unsafe")
        manifest = json.loads(archive.read("H8_MINIMAL_PACKAGE_MEMBER_MANIFEST.json"))
        checksums = {}
        for line in archive.read("H8_MINIMAL_PACKAGE_CHECKSUMS.sha256").decode("utf-8").splitlines():
            digest, separator, relative = line.partition("  ")
            if not separator or relative in checksums:
                raise ValueError("H8 parent checksum index is invalid")
            checksums[relative] = digest
        manifest_basis = manifest.get("basis", manifest)
        manifest_rows = {str(row["relative_path"]): row for row in manifest_basis["members"]}
        if set(manifest_rows) != set(checksums):
            raise ValueError("H8 parent manifest/checksum member sets differ")
        expected_archive_members = set(manifest_rows) | {
            "H8_MINIMAL_PACKAGE_MEMBER_MANIFEST.json",
            "H8_MINIMAL_PACKAGE_CHECKSUMS.sha256",
        }
        if set(names) != expected_archive_members:
            raise ValueError("H8 parent ZIP has undeclared or missing members")
        for relative, row in manifest_rows.items():
            payload = archive.read(relative)
            import hashlib

            if hashlib.sha256(payload).hexdigest() != checksums[relative] or len(payload) != int(row["size_bytes"]):
                raise ValueError(f"H8 parent member changed: {relative}")
        complete = json.loads(archive.read("M4A_H8_COMPLETE.json"))
        freeze = json.loads(archive.read("TC_LPMSO_H8_FINAL_METHOD_FREEZE.json"))
        registry = json.loads(archive.read("TC_LPMSO_H8_FINAL_PATH_SET_REGISTRY.json"))
        git_identity = json.loads(archive.read("H8_GIT_AND_CONFIG_IDENTITY.json"))
    if (
        complete.get("status") != PARENT_TERMINAL_STATUS
        or freeze.get("identity") != H8_METHOD_FREEZE_IDENTITY
        or registry.get("identity") != H8_PATH_REGISTRY_IDENTITY
        or git_identity.get("basis", {}).get("source_commit") != PARENT_SOURCE_COMMIT
        or git_identity.get("basis", {}).get("source_tree") != PARENT_SOURCE_TREE
        or freeze.get("basis", {}).get("further_retuning") is not False
    ):
        raise ValueError("H8 frozen method boundary mismatch")
    return envelope(
        "metachorus.v53.m6f-m7f.parent-package-audit.v1",
        "PASS",
        {
            "package_name": parent_package.name,
            "package_sha256": PARENT_PACKAGE_SHA256,
            "package_size_bytes": PARENT_PACKAGE_SIZE_BYTES,
            "member_count": len(names),
            "member_manifest_identity": manifest.get(
                "identity",
                manifest.get("manifest_identity", canonical_json_hash(manifest_basis)),
            ),
            "terminal_status": PARENT_TERMINAL_STATUS,
            "h8_method_freeze_identity": H8_METHOD_FREEZE_IDENTITY,
            "h8_path_registry_identity": H8_PATH_REGISTRY_IDENTITY,
            "parent_source_commit": PARENT_SOURCE_COMMIT,
            "parent_source_tree": PARENT_SOURCE_TREE,
            "method_frozen": True,
            "further_retuning": False,
        },
    )


def _resolution_rows(cache_rows: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    datasets: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        spec = DATASET_SPECS[dataset]
        cache = cache_rows[dataset]
        path_library = cache["path_library"]
        all_sets = cache["all_sets"]
        manual = cache["manual"]
        datasets.append(
            {
                "dataset": dataset,
                "resolution_status": "RESOLVED_FOR_FAST_SEARCH",
                "dataset_identity": DATASET_IDENTITIES[dataset],
                "structural_data_identity": spec.structural_data_identity,
                "loader": spec.loader,
                "source_relative_root": spec.source_relative_root,
                "processed_relative_path": spec.processed_relative_path,
                "processed_sha256": spec.processed_sha256,
                "target_node_type": spec.target_node_type,
                "target_node_count": spec.label_count,
                "feature_shape": json.dumps(list(spec.feature_shape), separators=(",", ":")),
                "class_count": spec.class_count,
                "cache_identity": cache["cache_identity"],
                "path_library_identity": cache["path_library_identity"],
                "all_sets_identity": cache["all_sets_identity"],
                "schema_path_count": path_library["schema_path_count"],
                "effective_unique_path_count": path_library["effective_path_count"],
                "fixed_cardinality": spec.view_budget,
                "logical_set_count": all_sets["set_count"],
                "unique_set_hash_count": len({row["candidate_set_hash"] for row in all_sets["sets"]}),
                "manual_status": "EXACT_IDENTITY_RESOLVED",
                "current_automatic_status": "NOT_AVAILABLE_EXACT_IDENTITY",
                "search_branch": "C_GT_40",
                "training_process_mounts_labels": False,
            }
        )
        for row in path_library["paths"]:
            paths.append(
                {
                    "dataset": dataset,
                    "path_ordinal": row["ordinal"],
                    "path_id": row["path_id"],
                    "semantic_projection_hash": row["semantic_projection_hash"],
                    "edge_count": row["edge_count"],
                    "gpu_edge_index_bytes": row["gpu_edge_index_bytes"],
                    "is_manual_representative": row["is_manual_representative"],
                    "equivalent_path_ids_json": json.dumps(row["equivalent_path_ids"], separators=(",", ":")),
                    "status": "LEGAL_FROZEN_PATH",
                }
            )
        baselines.extend(
            [
                {
                    "dataset": dataset,
                    "role": "MANUAL",
                    "availability": "EXACT_IDENTITY_RESOLVED",
                    "candidate_set_id": manual["candidate_set_id"],
                    "candidate_set_hash": manual["candidate_set_hash"],
                    "candidate_set_ordinal": manual["candidate_set_ordinal"],
                    "path_ids_json": json.dumps(manual["path_ids"], separators=(",", ":")),
                    "path_library_identity": cache["path_library_identity"],
                    "source_config_identity": cache["cache_identity"],
                },
                {
                    "dataset": dataset,
                    "role": "CURRENT_AUTOMATIC",
                    "availability": "NOT_AVAILABLE_EXACT_IDENTITY",
                    "candidate_set_id": "",
                    "candidate_set_hash": "",
                    "candidate_set_ordinal": "",
                    "path_ids_json": "[]",
                    "path_library_identity": cache["path_library_identity"],
                    "source_config_identity": "",
                },
            ]
        )
    return datasets, paths, baselines


def _validated_seed_role_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for name in SEED_ROLE_PUBLIC_ARTIFACTS:
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        value = json.loads(source.read_text(encoding="utf-8"))
        basis = value.get("basis")
        if not isinstance(basis, dict) or value.get("identity") != canonical_json_hash(basis):
            raise ValueError(f"public seed/label-role artifact identity mismatch: {name}")
        values[name] = value

    seed_manifest = values["FAST_SEED_NAMESPACE_MANIFEST.json"]
    collision = values["FAST_SEED_COLLISION_AUDIT.json"]
    role_manifest = values["FAST_LABEL_ROLE_MANIFEST.json"]
    role_seal = values["FAST_LABEL_ROLE_SEAL.json"]
    private_audit = values["FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json"]
    if (
        seed_manifest.get("status") != "SEALED_BEFORE_ANY_GPU_TRAINING"
        or seed_manifest.get("identity") != SEED_SEAL_IDENTITY
        or collision.get("status") != "PASS"
        or collision["basis"].get("derived_historical_intersection") != []
        or collision["basis"].get("search_eval_model_intersection") != []
        or role_manifest.get("status") != "SEALED_INPUT"
        or role_manifest["basis"].get("seed_seal_identity") != seed_manifest["identity"]
        or role_manifest["basis"].get("training_process_mounts_labels") is not False
        or role_manifest["basis"].get("raw_labels_embedded") is not False
        or role_manifest["basis"].get("prelim_locked_report_labels_embedded") is not False
        or private_audit.get("status") != "PASS"
        or private_audit["basis"].get("contains_full_labels") is not False
        or private_audit["basis"].get("contains_prelim_locked_report") is not False
        or role_seal.get("status") != "SEALED_BEFORE_SEARCH_TRAINING"
        or role_seal["basis"].get("role_manifest_identity") != role_manifest["identity"]
        or role_seal["basis"].get("private_mount_audit_identity") != private_audit["identity"]
        or role_seal["basis"].get("training_process_mounts_labels") is not False
    ):
        raise ValueError("seed/label role boundary is not sealed")
    return values


def build_preflight(
    *,
    repo_root: str | Path,
    taskbook: str | Path,
    parent_package: str | Path,
    cache_roots: Mapping[str, str | Path],
    seed_and_role_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    taskbook_path = Path(taskbook).resolve()
    parent = Path(parent_package).resolve()
    seeds = Path(seed_and_role_root).resolve()
    result = Path(result_root).resolve()
    if result.exists() and any(result.iterdir()):
        raise RuntimeError("M6F/M7F preflight result root must be fresh")
    result.mkdir(parents=True, exist_ok=True)
    if taskbook_path.name != TASKBOOK_FILENAME or sha256_file(taskbook_path) != TASKBOOK_SHA256:
        raise ValueError("M6F/M7F taskbook identity mismatch")
    source = _clean_source(repo)
    parent_audit = _parent_audit(parent)
    if set(cache_roots) != set(DATASET_NAMES):
        raise ValueError("preflight requires exactly three cache roots")
    cache_rows: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_NAMES:
        row = load_cache_resolution(cache_roots[dataset])
        FastSanitizedAdapter(cache_roots[dataset], expected_dataset=dataset)
        cache_rows[dataset] = row
    if adapter_compliance()["complete"] is not True:
        raise RuntimeError("fast training adapter compliance failed")
    seed_role_artifacts = _validated_seed_role_artifacts(seeds)
    seed_manifest = seed_role_artifacts["FAST_SEED_NAMESPACE_MANIFEST.json"]
    role_seal = seed_role_artifacts["FAST_LABEL_ROLE_SEAL.json"]
    datasets, paths, baselines = _resolution_rows(cache_rows)
    search_rows, search_seal = build_search_plan(cache_rows)
    canary_rows, canary_seal = build_canary_plan(
        cache_rows,
        source_commit=source["source_commit"],
        source_tree=source["source_tree"],
    )
    write_json_once(result / "M6F_M7F_PARENT_H8_AUDIT.json", parent_audit)
    for name, value in seed_role_artifacts.items():
        write_json_once(result / name, value)
    write_csv_once(result / "FAST_DATASET_RESOLUTION.csv", datasets, fieldnames=list(datasets[0]))
    write_json_once(result / "FAST_DATASET_RESOLUTION.json", envelope(
        "metachorus.v53.m6f-m7f.dataset-resolution.v1",
        "PASS",
        {"datasets": datasets, "resolved_dataset_count": len(datasets), "blocked_datasets": []},
    ))
    write_csv_once(result / "FAST_PATH_LIBRARY.csv", paths, fieldnames=list(paths[0]))
    write_csv_once(result / "FAST_BASELINE_SET_REGISTRY.csv", baselines, fieldnames=list(baselines[0]))
    warnings = [
        {
            "dataset": dataset,
            "warning_code": "CURRENT_AUTOMATIC_NOT_AVAILABLE_EXACT_IDENTITY",
            "blocking": False,
            "action": "do not create a synthetic Current Automatic run",
        }
        for dataset in DATASET_NAMES
    ]
    write_csv_once(result / "FAST_RESOLUTION_WARNINGS.csv", warnings, fieldnames=list(warnings[0]))
    write_csv_once(result / "FAST_SEARCH_EXPERIMENT_MATRIX.csv", search_rows, fieldnames=list(search_rows[0]))
    write_json_once(result / "FAST_SEARCH_EXPERIMENT_MATRIX_SEAL.json", search_seal)
    write_csv_once(result / "FAST_CANARY_RUN_PLAN.csv", canary_rows, fieldnames=list(canary_rows[0]))
    write_json_once(result / "FAST_CANARY_PLAN_SEAL.json", canary_seal)
    budget = envelope(
        "metachorus.v53.m6f-m7f.pre-canary-budget.v1",
        "PENDING_ENGINEERING_CANARY",
        {
            "dataset_run_counts": search_seal["basis"]["per_dataset_rows"],
            "total_r1_run_count": len(search_rows),
            "search_endpoint_update": 300,
            "gpu_count": 2,
            "resource_limits": RESOURCE_LIMITS,
            "estimated_seconds_per_step": None,
            "estimated_gpu_hours": None,
            "estimated_critical_path_hours": None,
            "estimated_remote_storage_bytes": None,
            "formal_search_authorized": False,
            "required_next_gate": "THREE_DATASET_WORST_CASE_20_UPDATE_ENGINEERING_CANARY",
        },
    )
    write_json_once(result / "FAST_PRE_GPU_BUDGET.json", budget)
    identity = envelope(
        "metachorus.v53.m6f-m7f.git-and-config-identity.v1",
        "PASS",
        {
            **source,
            "taskbook_sha256": TASKBOOK_SHA256,
            "parent_h8_package_sha256": PARENT_PACKAGE_SHA256,
            "parent_h8_method_freeze_identity": H8_METHOD_FREEZE_IDENTITY,
            "parent_h8_path_registry_identity": H8_PATH_REGISTRY_IDENTITY,
            "seed_seal_identity": SEED_SEAL_IDENTITY,
            "cache_identities": {name: cache_rows[name]["cache_identity"] for name in DATASET_NAMES},
            "path_library_identities": {name: cache_rows[name]["path_library_identity"] for name in DATASET_NAMES},
            "all_sets_identities": {name: cache_rows[name]["all_sets_identity"] for name in DATASET_NAMES},
            "algorithm_retuning": False,
            "training_process_mounts_labels": False,
        },
    )
    write_json_once(result / "FAST_GIT_AND_CONFIG_IDENTITY.json", identity)
    complete = envelope(
        "metachorus.v53.m6f-m7f.preflight-complete.v1",
        "READY_FOR_THREE_ENGINEERING_CANARIES",
        {
            "parent_audit_identity": parent_audit["identity"],
            "dataset_resolution_identity": canonical_json_hash(datasets),
            "path_library_rows_identity": canonical_json_hash(paths),
            "baseline_registry_identity": canonical_json_hash(baselines),
            "seed_seal_identity": SEED_SEAL_IDENTITY,
            "label_role_seal_identity": role_seal["identity"],
            "search_plan_identity": search_seal["identity"],
            "canary_plan_identity": canary_seal["identity"],
            "git_and_config_identity": identity["identity"],
            "resolved_dataset_count": 3,
            "blocked_dataset_count": 0,
            "formal_search_started": False,
            "preliminary_evaluation_started": False,
            "labels_mounted_during_training": False,
            "next_action": "run one worst-case 20-update engineering canary per dataset",
        },
    )
    write_json_once(result / "M6F_M7F_PREFLIGHT_COMPLETE.json", complete)
    write_text_once(
        result / "plan.md",
        "# M6F/M7F fast execution state\n\n"
        "- H8 method/path freeze: PASS.\n"
        "- DBLP/IMDb/AMiner exact resolver: PASS.\n"
        "- Current Automatic: NOT_AVAILABLE_EXACT_IDENTITY for all three datasets; no synthetic runs.\n"
        "- Seed and label roles: SEALED; training label mount forbidden.\n"
        "- R1 exact search matrix: SEALED.\n"
        "- Next: three worst-case 20-update engineering canaries. Formal search remains unauthorized.\n",
    )
    return complete


__all__ = ["build_preflight"]
