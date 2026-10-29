"""Fresh, label-safe local preflight for the AMiner H2 campaign."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

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
    ALL_SETS_IDENTITY,
    CACHE_IDENTITY,
    EVALUATION_MODEL_SEEDS,
    FROZEN_SEED_NAMESPACE_IDENTITY,
    LABEL_ROLE_MANIFEST_IDENTITY,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    PARENT_TERMINAL_STATUS,
    PATH_LIBRARY_IDENTITY,
    REQUIRED_TEST_FILES,
    SEARCH_MODEL_SEEDS,
    SEED_ROLE_PUBLIC_ARTIFACTS,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
    TRAINING_CORE_FILES,
)
from .planning import (
    build_canary_plan,
    build_r1_plan,
    load_aminer_cache,
    resolve_all_sets,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_identity(repo: Path) -> tuple[str, str]:
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("M7F-H2 formal preflight requires a clean source worktree")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if len(commit) != 40 or len(tree) != 40:
        raise RuntimeError("M7F-H2 source commit/tree is not canonical")
    return commit, tree


def audit_parent_package(parent_package: str | Path) -> dict[str, Any]:
    path = Path(parent_package).resolve(strict=True)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.name != PARENT_PACKAGE_FILENAME
        or path.stat().st_size != PARENT_PACKAGE_SIZE_BYTES
        or sha256_file(path) != PARENT_PACKAGE_SHA256
    ):
        raise ValueError("M7F-H2 parent ZIP identity mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("M7F-H2 parent ZIP integrity failed")
        complete = json.loads(archive.read("M7F_H1_COMPLETE.json").decode("utf-8"))
        git_config = json.loads(archive.read("M7F_H1_GIT_AND_CONFIG_IDENTITY.json").decode("utf-8"))
    complete_basis = complete.get("basis", {})
    git_basis = git_config.get("basis", {})
    if (
        complete.get("status") != PARENT_TERMINAL_STATUS
        or complete.get("identity") != canonical_json_hash(complete_basis)
        or git_config.get("status") != "PASS"
        or git_config.get("identity") != canonical_json_hash(git_basis)
        or git_basis.get("source_commit") != PARENT_SOURCE_COMMIT
        or git_basis.get("source_tree") != PARENT_SOURCE_TREE
        or git_basis.get("cache_identities", {}).get("aminer") != CACHE_IDENTITY
        or git_basis.get("path_library_identities", {}).get("aminer") != PATH_LIBRARY_IDENTITY
        or git_basis.get("all_sets_identities", {}).get("aminer") != ALL_SETS_IDENTITY
    ):
        raise ValueError("M7F-H2 parent scientific identity is not exact")
    basis = {
        "filename": path.name,
        "sha256": PARENT_PACKAGE_SHA256,
        "size_bytes": PARENT_PACKAGE_SIZE_BYTES,
        "source_commit": PARENT_SOURCE_COMMIT,
        "source_tree": PARENT_SOURCE_TREE,
        "terminal_status": PARENT_TERMINAL_STATUS,
        "parent_complete_identity": complete["identity"],
        "parent_git_identity": git_config["identity"],
        "aminer_cache_identity": CACHE_IDENTITY,
        "aminer_path_library_identity": PATH_LIBRARY_IDENTITY,
        "aminer_all_sets_identity": ALL_SETS_IDENTITY,
    }
    return envelope("metachorus.v53.m7f-h2.parent-identity.v1", "PASS", basis)


def _training_core_freeze(repo: Path) -> dict[str, Any]:
    rows = []
    for relative in TRAINING_CORE_FILES:
        current = repo / Path(*relative.split("/"))
        if current.is_symlink() or not current.is_file():
            raise FileNotFoundError(current)
        parent_payload = subprocess.run(
            ["git", "-C", str(repo), "show", f"{PARENT_SOURCE_COMMIT}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        parent_hash = hashlib.sha256(parent_payload).hexdigest()
        current_hash = sha256_file(current)
        if parent_hash != current_hash:
            raise ValueError(f"frozen training core changed from H1 parent: {relative}")
        rows.append({"relative_path": relative, "parent_sha256": parent_hash, "current_sha256": current_hash})
    basis = {
        "method_name": "RS-TC-LPMSO",
        "trainer": "O0-CodeFaithful",
        "view_pipeline": "S0",
        "maximum_update": 3000,
        "training_core_files": rows,
        "training_core_unchanged_from_parent": True,
        "labels_visible_during_training": False,
        "algorithm_retuning": False,
        "search_rules_reimplemented": False,
        "frozen_protocol_module": "metachorus_v53.m6f_m7f_fast.protocol",
    }
    return envelope("metachorus.v53.m7f-h2.method-freeze.v1", "PASS", basis)


def _copy_public_roles(source_root: Path, result: Path) -> dict[str, str]:
    documents: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}
    for name in SEED_ROLE_PUBLIC_ARTIFACTS:
        source = source_root / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        value = read_json(source)
        if value.get("identity") != canonical_json_hash(value.get("basis", {})):
            raise ValueError(f"H2 public role identity changed: {name}")
        documents[name] = value
        identities[name] = str(value["identity"])
    manifest = documents["FAST_LABEL_ROLE_MANIFEST.json"]
    seal = documents["FAST_LABEL_ROLE_SEAL.json"]
    mount = documents["FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json"]
    seed = documents["FAST_SEED_NAMESPACE_MANIFEST.json"]
    collision = documents["FAST_SEED_COLLISION_AUDIT.json"]
    if (
        manifest.get("status") != "SEALED_INPUT"
        or manifest.get("identity") != LABEL_ROLE_MANIFEST_IDENTITY
        or manifest.get("basis", {}).get("raw_labels_embedded") is not False
        or manifest.get("basis", {}).get("training_process_mounts_labels") is not False
        or seal.get("status") != "SEALED_BEFORE_SEARCH_TRAINING"
        or seal.get("basis", {}).get("role_manifest_identity") != manifest["identity"]
        or seal.get("basis", {}).get("private_mount_audit_identity") != mount["identity"]
        or mount.get("status") != "PASS"
        or mount.get("basis", {}).get("contains_full_labels") is not False
        or mount.get("basis", {}).get("contains_prelim_locked_report") is not False
        or seed.get("identity") != FROZEN_SEED_NAMESPACE_IDENTITY
        or tuple(seed.get("basis", {}).get("search_model_seeds", [])) != SEARCH_MODEL_SEEDS
        or tuple(seed.get("basis", {}).get("evaluation_model_seeds", [])) != EVALUATION_MODEL_SEEDS
        or collision.get("status") != "PASS"
    ):
        raise ValueError("M7F-H2 label/seed boundary is invalid")
    for name, value in documents.items():
        write_json_once(result / name, value)
    return identities


def build_preflight(
    *,
    repo_root: str | Path,
    taskbook: str | Path,
    parent_package: str | Path,
    cache_root: str | Path,
    seed_role_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    taskbook_path = Path(taskbook).resolve(strict=True)
    parent_path = Path(parent_package).resolve(strict=True)
    cache_path = Path(cache_root).resolve(strict=True)
    role_root = Path(seed_role_root).resolve(strict=True)
    result = Path(result_root).resolve()
    if result.exists() and any(result.iterdir()):
        raise FileExistsError("M7F-H2 preflight result root must be fresh")
    result.mkdir(parents=True, exist_ok=True)
    if (
        taskbook_path.is_symlink()
        or taskbook_path.name != TASKBOOK_FILENAME
        or sha256_file(taskbook_path) != TASKBOOK_SHA256
    ):
        raise ValueError("M7F-H2 taskbook identity changed")
    source_commit, source_tree = git_identity(repo)
    parent = audit_parent_package(parent_path)
    method = _training_core_freeze(repo)
    roles = _copy_public_roles(role_root, result)
    cache = load_aminer_cache(cache_path)
    resolution = resolve_all_sets(cache)
    canary_rows, canary_seal = build_canary_plan(cache)
    r1_rows, r1_seal = build_r1_plan(cache)
    write_json_once(result / "AMINER_H2_PARENT_IDENTITY.json", parent)
    write_json_once(result / "AMINER_H2_METHOD_FREEZE.json", method)
    write_csv_once(result / "AMINER_H2_ALL_SET_RESOLUTION.csv", resolution)
    write_csv_once(result / "AMINER_H2_CANARY_PLAN.csv", canary_rows)
    write_json_once(result / "AMINER_H2_CANARY_PLAN_SEAL.json", canary_seal)
    write_csv_once(result / "AMINER_H2_R1_RUN_PLAN.csv", r1_rows)
    write_json_once(result / "AMINER_H2_R1_RUN_PLAN_SEAL.json", r1_seal)
    git_basis = {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_worktree_clean": True,
        "taskbook_filename": TASKBOOK_FILENAME,
        "taskbook_sha256": TASKBOOK_SHA256,
        "parent_identity": parent["identity"],
        "method_freeze_identity": method["identity"],
        "dataset": "aminer",
        "cache_identity": CACHE_IDENTITY,
        "path_library_identity": PATH_LIBRARY_IDENTITY,
        "all_sets_identity": ALL_SETS_IDENTITY,
        "public_seed_role_identities": roles,
        "canary_plan_seal_identity": canary_seal["identity"],
        "r1_plan_seal_identity": r1_seal["identity"],
        "search_model_seeds": list(SEARCH_MODEL_SEEDS),
        "evaluation_model_seeds": list(EVALUATION_MODEL_SEEDS),
        "required_test_files": list(REQUIRED_TEST_FILES),
        "maximum_gpu_count": 4,
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
        "labels_mounted_during_training": False,
        "algorithm_retuning": False,
    }
    git_config = envelope("metachorus.v53.m7f-h2.git-and-config-identity.v1", "PASS", git_basis)
    write_json_once(result / "AMINER_H2_GIT_AND_CONFIG_IDENTITY.json", git_config)
    complete = envelope(
        "metachorus.v53.m7f-h2.preflight-complete.v1",
        "READY_FOR_AMINER_H2_4GPU_DEPLOYMENT",
        {
            "parent_identity": parent["identity"],
            "method_freeze_identity": method["identity"],
            "git_and_config_identity": git_config["identity"],
            "all_set_resolution_identity": canonical_json_hash(resolution),
            "all_sets_resolved": len(resolution) == 91,
            "canary_plan_seal_identity": canary_seal["identity"],
            "r1_plan_seal_identity": r1_seal["identity"],
            "scientific_training_started": False,
            "labels_mounted": False,
        },
    )
    write_json_once(result / "AMINER_H2_PREFLIGHT_COMPLETE.json", complete)
    write_text_once(
        result / "plan.md",
        "# AMiner H2 4GPU Plan\n\n"
        "1. Verify four RTX 4090 devices and one-device worker isolation.\n"
        "2. Run four 20-step engineering canaries; discard canary model state.\n"
        "3. Run all 91 sets x three search seeds to update 300.\n"
        "4. Freeze the original >40 high-recall pool; continue to 1000 and 2000.\n"
        "5. Freeze the robust shortlist; continue to 3000 and seal one set.\n"
        "6. Audit exact Manual reuse, run three fresh evaluation seeds if needed, then evaluate once on PRELIM_LOCKED_REPORT.\n"
        "7. Publish only the minimal ZIP and SHA-256 sidecar.\n",
    )
    return complete


__all__ = ["git_identity", "audit_parent_package", "build_preflight"]
