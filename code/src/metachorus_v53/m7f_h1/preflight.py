"""Fresh, label-safe preflight for the M7F-H1 remote campaign."""

from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
from typing import Any
import zipfile

from .common import (
    canonical_json_hash,
    envelope,
    read_json,
    sha256_file,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    PARENT_PACKAGE_FILENAME,
    REQUIRED_TEST_FILES,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
)
from .planning import (
    PLAN_FIELDS,
    audit_parent_package,
    build_budget_canary_plan,
    build_imdb_trajectory_plan,
    build_manual_plan,
    inventory_existing_sources,
    load_cache_resolution,
)


PUBLIC_ROLE_ARTIFACTS = (
    "FAST_SEED_NAMESPACE_MANIFEST.json",
    "FAST_SEED_COLLISION_AUDIT.json",
    "FAST_LABEL_ROLE_MANIFEST.json",
    "FAST_LABEL_ROLE_SEAL.json",
    "FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json",
)


def _git_identity(repo_root: Path) -> tuple[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    if status:
        raise RuntimeError("M7F-H1 formal preflight requires a clean source worktree")
    commit = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    if len(commit) != 40 or len(tree) != 40:
        raise RuntimeError("M7F-H1 source commit/tree is not canonical")
    return commit, tree


def _copy_public_roles(source_root: Path, result_root: Path) -> dict[str, str]:
    identities: dict[str, str] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name in PUBLIC_ROLE_ARTIFACTS:
        source = source_root / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        value = read_json(source)
        if value.get("identity") != canonical_json_hash(value.get("basis", {})):
            raise ValueError(f"public seed/role artifact identity changed: {name}")
        documents[name] = value
        identities[name] = str(value["identity"])
    manifest = documents["FAST_LABEL_ROLE_MANIFEST.json"]
    seal = documents["FAST_LABEL_ROLE_SEAL.json"]
    mount = documents["FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json"]
    collision = documents["FAST_SEED_COLLISION_AUDIT.json"]
    if (
        manifest.get("status") != "SEALED_INPUT"
        or manifest.get("basis", {}).get("raw_labels_embedded") is not False
        or manifest.get("basis", {}).get("training_process_mounts_labels") is not False
        or seal.get("status") != "SEALED_BEFORE_SEARCH_TRAINING"
        or seal.get("basis", {}).get("role_manifest_identity") != manifest["identity"]
        or seal.get("basis", {}).get("private_mount_audit_identity") != mount["identity"]
        or mount.get("status") != "PASS"
        or mount.get("basis", {}).get("contains_full_labels") is not False
        or mount.get("basis", {}).get("contains_prelim_locked_report") is not False
        or collision.get("status") != "PASS"
    ):
        raise ValueError("M7F-H1 public label/seed boundary is invalid")
    for name, value in documents.items():
        write_json_once(result_root / name, value)
    return identities


def _parent_retention_rows(parent_package: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(parent_package) as archive:
        try:
            payload = archive.read("REMOTE_FAST_ARTIFACT_RETENTION_MANIFEST.csv")
        except KeyError as exc:
            raise ValueError("parent package lacks the remote retention manifest") from exc
    rows = list(csv.DictReader(StringIO(payload.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("parent remote retention manifest is empty")
    return rows


def build_preflight(
    *,
    repo_root: str | Path,
    taskbook: str | Path,
    parent_package: str | Path,
    cache_parent: str | Path,
    seed_role_root: str | Path,
    result_root: str | Path,
    existing_remote_result_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    taskbook_path = Path(taskbook)
    parent_path = Path(parent_package)
    cache_parent_path = Path(cache_parent).resolve(strict=True)
    role_root = Path(seed_role_root).resolve(strict=True)
    result = Path(result_root).resolve()
    if result.exists() and any(result.iterdir()):
        raise FileExistsError("M7F-H1 preflight result root must be fresh")
    result.mkdir(parents=True, exist_ok=True)
    if (
        taskbook_path.name != TASKBOOK_FILENAME
        or taskbook_path.is_symlink()
        or not taskbook_path.is_file()
        or sha256_file(taskbook_path) != TASKBOOK_SHA256
    ):
        raise ValueError("M7F-H1 taskbook identity changed")
    if parent_path.name != PARENT_PACKAGE_FILENAME:
        raise ValueError("M7F-H1 parent filename changed")
    source_commit, source_tree = _git_identity(repo)
    parent_audit = audit_parent_package(parent_path)
    write_json_once(result / "M7F_H1_PARENT_PACKAGE_AUDIT.json", parent_audit)
    role_identities = _copy_public_roles(role_root, result)
    caches = {
        name: load_cache_resolution(cache_parent_path / name)
        for name in ("dblp", "imdb", "aminer")
    }
    trajectory_rows, trajectory_seal = build_imdb_trajectory_plan(caches["imdb"])
    manual_rows, manual_seal = build_manual_plan(
        {name: caches[name] for name in ("dblp", "aminer")}
    )
    canary_rows, canary_seal = build_budget_canary_plan(
        {name: caches[name] for name in ("dblp", "aminer")}
    )
    for filename, rows, seal_filename, seal in (
        (
            "M7F_H1_IMDB_TRAJECTORY_RUN_PLAN.csv",
            trajectory_rows,
            "M7F_H1_IMDB_TRAJECTORY_RUN_PLAN_SEAL.json",
            trajectory_seal,
        ),
        (
            "M7F_H1_MANUAL_RUN_PLAN.csv",
            manual_rows,
            "M7F_H1_MANUAL_RUN_PLAN_SEAL.json",
            manual_seal,
        ),
        (
            "M7F_H1_BUDGET_CANARY_RUN_PLAN.csv",
            canary_rows,
            "M7F_H1_BUDGET_CANARY_RUN_PLAN_SEAL.json",
            canary_seal,
        ),
    ):
        write_csv_once(result / filename, rows, fieldnames=PLAN_FIELDS)
        write_json_once(result / seal_filename, seal)
    retention_rows = _parent_retention_rows(parent_path)
    retention_path = result / "PARENT_REMOTE_FAST_ARTIFACT_RETENTION_MANIFEST.csv"
    write_csv_once(
        retention_path, retention_rows, fieldnames=list(retention_rows[0])
    )
    existing_root = (
        Path(existing_remote_result_root)
        if existing_remote_result_root is not None
        else result / "__MISSING_EXISTING_REMOTE_RESULT_ROOT__"
    )
    existing_status, inventory = inventory_existing_sources(
        remote_result_root=existing_root,
        retention_manifest=retention_path,
        expected_physical_run_ids=[row["physical_run_id"] for row in trajectory_rows],
    )
    write_csv_once(
        result / "IMDB_EXISTING_METRIC_SOURCE_INVENTORY.csv",
        inventory,
        fieldnames=list(inventory[0]),
    )
    existing_gate = envelope(
        "metachorus.v53.m7f-h1.existing-trajectory-gate.v1",
        existing_status,
        {
            "inventory_rows_identity": canonical_json_hash(inventory),
            "expected_physical_run_ids": [
                row["physical_run_id"] for row in trajectory_rows
            ],
            "remote_result_root_found": existing_status
            != "IMDB_REMOTE_RESULT_ROOT_NOT_FOUND",
            "fallback_six_trajectory_runs_required": existing_status
            != "IMDB_EXISTING_TRAJECTORY_COMPLETE",
            "training_started": False,
            "labels_mounted": False,
        },
    )
    write_json_once(result / "IMDB_EXISTING_TRAJECTORY_GATE.json", existing_gate)
    git_basis = {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_worktree_clean": True,
        "taskbook_filename": TASKBOOK_FILENAME,
        "taskbook_sha256": TASKBOOK_SHA256,
        "parent_package_identity": parent_audit["identity"],
        "cache_identities": {
            name: caches[name]["cache_identity"] for name in caches
        },
        "path_library_identities": {
            name: caches[name]["path_library_identity"] for name in caches
        },
        "all_sets_identities": {
            name: caches[name]["all_sets_identity"] for name in caches
        },
        "public_seed_role_identities": role_identities,
        "trajectory_plan_seal_identity": trajectory_seal["identity"],
        "manual_plan_seal_identity": manual_seal["identity"],
        "budget_canary_plan_seal_identity": canary_seal["identity"],
        "conditional_full_search_protocol": {
            "authorization_artifact": "DBLP_AMINER_SEARCH_AUTHORIZATION.json",
            "per_dataset_gate": True,
            "stage_endpoints": [300, 1000, 2000, 3000],
            "fresh_preliminary_runs_per_authorized_dataset": 6,
            "manual_safety_gate_required": True,
            "unique_selection_seal_required": True,
            "candidate_library_reduced": False,
            "algorithm_changed": False,
        },
        "required_test_files": list(REQUIRED_TEST_FILES),
        "labels_mounted_during_training": False,
        "algorithm_retuning": False,
    }
    git_identity = envelope(
        "metachorus.v53.m7f-h1.git-config-identity.v1", "PASS", git_basis
    )
    write_json_once(result / "M7F_H1_GIT_AND_CONFIG_IDENTITY.json", git_identity)
    preflight = envelope(
        "metachorus.v53.m7f-h1.preflight-complete.v1",
        "READY_FOR_M7F_H1_REMOTE_EXECUTION",
        {
            "git_and_config_identity": git_identity["identity"],
            "parent_package_identity": parent_audit["identity"],
            "existing_trajectory_gate_identity": existing_gate["identity"],
            "next_action": (
                "EVALUATE_EXISTING_IMDB_EVIDENCE_WITHOUT_TRAINING"
                if existing_status == "IMDB_EXISTING_TRAJECTORY_COMPLETE"
                else "RUN_SIX_IMDB_TRAJECTORIES_PLUS_MANUAL_AND_BUDGET_CANARIES"
            ),
            "imdb_fallback_run_count": 0
            if existing_status == "IMDB_EXISTING_TRAJECTORY_COMPLETE"
            else 6,
            "manual_run_count": 6,
            "budget_canary_run_count": 12,
            "maximum_gpu_count": 2,
            "run_level_parallel_only": True,
            "ddp_forbidden": True,
            "labels_mounted_during_training": False,
            "prelim_locked_report_access_started": False,
        },
    )
    write_json_once(result / "M7F_H1_PREFLIGHT_COMPLETE.json", preflight)
    write_text_once(
        result / "plan.md",
        "# M7F-H1 sealed execution\n\n"
        "1. Inventory/evaluate existing IMDb evidence without training.\n"
        "2. If incomplete, run exactly six IMDb trajectories from zero.\n"
        "3. Run six DBLP/AMiner Manual baselines and twelve engineering canaries.\n"
        "4. Freeze checkpoint rules using GLOBAL_SEARCH_POOL only.\n"
        "5. Open PRELIM_LOCKED_REPORT only after the selection/training seals.\n"
        "6. For each budget-authorized dataset, execute the complete frozen "
        "R1/R2/R3/R4 search, Manual gate, unique seal, and fresh three-seed "
        "preliminary comparison.\n",
    )
    return preflight


__all__ = ["PUBLIC_ROLE_ARTIFACTS", "build_preflight"]
