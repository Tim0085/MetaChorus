"""Fresh, append-only local preflight for the M7F-H3 campaign."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
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
    DATASET_IDENTITY,
    DBLP_LABEL_SOURCE_SHA256,
    GLOBAL_SEARCH_POOL_COUNT,
    PRELIM_LOCKED_REPORT_COUNT,
    EVALUATION_MODEL_SEEDS,
    FROZEN_PROTOCOL_SHA256,
    FROZEN_SEED_NAMESPACE_IDENTITY,
    LABEL_ROLE_MANIFEST_IDENTITY,
    LEGAL_PATH_COUNT,
    LEGAL_SET_COUNT,
    PARENT_AMINER_PACKAGE_FILENAME,
    PARENT_AMINER_PACKAGE_SHA256,
    PARENT_AMINER_PACKAGE_SIZE_BYTES,
    PARENT_AMINER_SOURCE_COMMIT,
    PARENT_AMINER_SOURCE_TREE,
    PARENT_H1_PACKAGE_FILENAME,
    PARENT_H1_PACKAGE_SHA256,
    PARENT_H1_PACKAGE_SIZE_BYTES,
    PATH_LIBRARY_IDENTITY,
    REQUIRED_TEST_FILES,
    R1_MODEL_SEEDS,
    SEARCH_MODEL_SEEDS,
    SEED_ROLE_PUBLIC_ARTIFACTS,
    SET_CARDINALITY,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
    TRAINING_CORE_FILES,
)
from .correction import build_aminer_correction
from .planning import build_canary_plan, build_r1_plan, load_dblp_cache, resolve_all_sets
from .protocol import build_protocol_scope_audit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_identity(repo: Path) -> tuple[str, str]:
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("M7F-H3 formal preflight requires a clean source worktree")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if len(commit) != 40 or len(tree) != 40:
        raise RuntimeError("M7F-H3 source commit/tree is not canonical")
    return commit, tree


def _safe_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError("M7F-H3 parent ZIP contains a non-portable member")
    value = PurePosixPath(name)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError("M7F-H3 parent ZIP contains an unsafe member")
    return value


def _read_parent(
    path_value: str | Path,
    *,
    filename: str,
    digest: str,
    size_bytes: int,
    complete_name: str,
    git_name: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    raw = Path(path_value).absolute()
    if raw.is_symlink():
        raise ValueError("M7F-H3 parent package cannot be a symlink")
    path = raw.resolve(strict=True)
    if (
        not path.is_file()
        or path.name != filename
        or path.stat().st_size != size_bytes
        or sha256_file(path) != digest
    ):
        raise ValueError("M7F-H3 parent package identity mismatch")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (
            archive.testzip() is not None
            or len(names) != len(set(names))
            or len(names) != len({name.casefold() for name in names})
        ):
            raise ValueError("M7F-H3 parent ZIP integrity failed")
        for info in infos:
            _safe_member(info.filename)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("M7F-H3 parent ZIP contains a symlink")
        if complete_name not in names or git_name not in names:
            raise ValueError("M7F-H3 parent ZIP lacks governed identity members")
        complete = json.loads(archive.read(complete_name).decode("utf-8"))
        git_config = json.loads(archive.read(git_name).decode("utf-8"))
    for value in (complete, git_config):
        basis = value.get("basis", {})
        if not isinstance(basis, dict) or value.get("identity") != canonical_json_hash(basis):
            raise ValueError("M7F-H3 parent envelope identity is invalid")
    return path, complete, git_config


def audit_parent_packages(
    parent_aminer_package: str | Path,
    parent_h1_package: str | Path,
) -> dict[str, Any]:
    _aminer_path, aminer_complete, aminer_git = _read_parent(
        parent_aminer_package,
        filename=PARENT_AMINER_PACKAGE_FILENAME,
        digest=PARENT_AMINER_PACKAGE_SHA256,
        size_bytes=PARENT_AMINER_PACKAGE_SIZE_BYTES,
        complete_name="AMINER_H2_COMPLETE.json",
        git_name="AMINER_H2_GIT_AND_CONFIG_IDENTITY.json",
    )
    _h1_path, h1_complete, h1_git = _read_parent(
        parent_h1_package,
        filename=PARENT_H1_PACKAGE_FILENAME,
        digest=PARENT_H1_PACKAGE_SHA256,
        size_bytes=PARENT_H1_PACKAGE_SIZE_BYTES,
        complete_name="M7F_H1_COMPLETE.json",
        git_name="M7F_H1_GIT_AND_CONFIG_IDENTITY.json",
    )
    aminer_basis = aminer_git.get("basis", {})
    h1_basis = h1_git.get("basis", {})
    if (
        aminer_complete.get("status") != "AMINER_H2_COMPLETE"
        or aminer_git.get("status") != "PASS"
        or aminer_basis.get("source_commit") != PARENT_AMINER_SOURCE_COMMIT
        or aminer_basis.get("source_tree") != PARENT_AMINER_SOURCE_TREE
        or aminer_basis.get("labels_mounted_during_training") is not False
        or aminer_basis.get("algorithm_retuning") is not False
        or h1_complete.get("status") != "M7F_H1_PARTIAL"
        or h1_git.get("status") != "PASS"
        or h1_basis.get("cache_identities", {}).get("dblp") != CACHE_IDENTITY
        or h1_basis.get("path_library_identities", {}).get("dblp") != PATH_LIBRARY_IDENTITY
        or h1_basis.get("all_sets_identities", {}).get("dblp") != ALL_SETS_IDENTITY
        or h1_basis.get("labels_mounted_during_training") is not False
        or h1_basis.get("algorithm_retuning") is not False
    ):
        raise ValueError("M7F-H3 parent scientific identity is not exact")
    basis = {
        "aminer_parent": {
            "filename": PARENT_AMINER_PACKAGE_FILENAME,
            "sha256": PARENT_AMINER_PACKAGE_SHA256,
            "size_bytes": PARENT_AMINER_PACKAGE_SIZE_BYTES,
            "source_commit": PARENT_AMINER_SOURCE_COMMIT,
            "source_tree": PARENT_AMINER_SOURCE_TREE,
            "terminal_status": "AMINER_H2_COMPLETE",
            "complete_identity": aminer_complete["identity"],
            "git_identity": aminer_git["identity"],
        },
        "h1_parent": {
            "filename": PARENT_H1_PACKAGE_FILENAME,
            "sha256": PARENT_H1_PACKAGE_SHA256,
            "size_bytes": PARENT_H1_PACKAGE_SIZE_BYTES,
            "terminal_status": "M7F_H1_PARTIAL",
            "complete_identity": h1_complete["identity"],
            "git_identity": h1_git["identity"],
        },
        "dblp_dataset_identity": DATASET_IDENTITY,
        "dblp_cache_identity": CACHE_IDENTITY,
        "dblp_path_library_identity": PATH_LIBRARY_IDENTITY,
        "dblp_all_sets_identity": ALL_SETS_IDENTITY,
    }
    return envelope("metachorus.v53.m7f-h3.parent-identity.v1", "PASS", basis)


def _training_core_freeze(repo: Path) -> dict[str, Any]:
    rows = []
    for relative in TRAINING_CORE_FILES:
        current = repo / Path(*relative.split("/"))
        if current.is_symlink() or not current.is_file():
            raise FileNotFoundError(current)
        parent_payload = subprocess.run(
            ["git", "-C", str(repo), "show", f"{PARENT_AMINER_SOURCE_COMMIT}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        parent_hash = hashlib.sha256(parent_payload).hexdigest()
        current_hash = sha256_file(current)
        if parent_hash != current_hash:
            raise ValueError(f"frozen training core changed from the H2 parent: {relative}")
        rows.append(
            {
                "relative_path": relative,
                "parent_sha256": parent_hash,
                "current_sha256": current_hash,
            }
        )
    basis = {
        "method_name": "RS-TC-LPMSO",
        "trainer": "O0-CodeFaithful",
        "view_pipeline": "S0",
        "maximum_update": 3000,
        "training_core_files": rows,
        "training_core_unchanged_from_parent": True,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "labels_visible_during_training": False,
        "algorithm_retuning": False,
        "search_rules_reimplemented": False,
        "frozen_protocol_module": "metachorus_v53.m6f_m7f_fast.protocol",
    }
    return envelope("metachorus.v53.m7f-h3.method-freeze.v1", "PASS", basis)


def _copy_public_roles(source_root: Path, result: Path) -> dict[str, str]:
    documents: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}
    for name in SEED_ROLE_PUBLIC_ARTIFACTS:
        source = source_root / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        value = read_json(source)
        if value.get("identity") != canonical_json_hash(value.get("basis", {})):
            raise ValueError(f"M7F-H3 public role identity changed: {name}")
        documents[name] = value
        identities[name] = str(value["identity"])
    manifest = documents["FAST_LABEL_ROLE_MANIFEST.json"]
    seal = documents["FAST_LABEL_ROLE_SEAL.json"]
    mount = documents["FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json"]
    seed = documents["FAST_SEED_NAMESPACE_MANIFEST.json"]
    collision = documents["FAST_SEED_COLLISION_AUDIT.json"]
    dblp_roles = [
        row
        for row in manifest.get("basis", {}).get("datasets", [])
        if row.get("dataset") == "dblp"
    ]
    if (
        manifest.get("status") != "SEALED_INPUT"
        or manifest.get("identity") != LABEL_ROLE_MANIFEST_IDENTITY
        or manifest.get("basis", {}).get("raw_labels_embedded") is not False
        or manifest.get("basis", {}).get("training_process_mounts_labels") is not False
        or len(dblp_roles) != 1
        or dblp_roles[0].get("dataset_identity") != DATASET_IDENTITY
        or int(dblp_roles[0].get("GLOBAL_SEARCH_POOL", {}).get("count", -1)) != GLOBAL_SEARCH_POOL_COUNT
        or int(dblp_roles[0].get("PRELIM_LOCKED_REPORT", {}).get("count", -1)) != PRELIM_LOCKED_REPORT_COUNT
        or dblp_roles[0].get("label_source_sha256") != DBLP_LABEL_SOURCE_SHA256
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
        raise ValueError("M7F-H3 label/seed boundary is invalid")
    for name, value in documents.items():
        write_json_once(result / name, value)
    return identities


def build_preflight(
    *,
    repo_root: str | Path,
    taskbook: str | Path,
    parent_aminer_package: str | Path,
    parent_h1_package: str | Path,
    cache_root: str | Path,
    seed_role_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    taskbook_path = Path(taskbook).resolve(strict=True)
    aminer_parent = Path(parent_aminer_package).resolve(strict=True)
    h1_parent = Path(parent_h1_package).resolve(strict=True)
    cache_path = Path(cache_root).resolve(strict=True)
    role_root = Path(seed_role_root).resolve(strict=True)
    result = Path(result_root).resolve()
    if result.exists() and any(result.iterdir()):
        raise FileExistsError("M7F-H3 preflight result root must be fresh")
    result.mkdir(parents=True, exist_ok=True)
    if (
        taskbook_path.is_symlink()
        or taskbook_path.name != TASKBOOK_FILENAME
        or sha256_file(taskbook_path) != TASKBOOK_SHA256
    ):
        raise ValueError("M7F-H3 taskbook identity changed")
    source_commit, source_tree = git_identity(repo)
    parent = audit_parent_packages(aminer_parent, h1_parent)
    method = _training_core_freeze(repo)
    correction = build_aminer_correction(
        parent_aminer_package=aminer_parent,
        parent_h1_package=h1_parent,
        result_root=result,
    )
    if correction.get("status") != "AMINER_FINAL_FALLBACK_CORRECTED":
        raise RuntimeError("M7F-H3 AMiner zero-GPU correction did not close")
    protocol = build_protocol_scope_audit(repo, result)
    roles = _copy_public_roles(role_root, result)
    cache = load_dblp_cache(cache_path)
    resolution = resolve_all_sets(cache)
    if len(resolution) != LEGAL_SET_COUNT:
        raise RuntimeError("M7F-H3 DBLP all-set resolution is not exactly 220")
    canary_rows, canary_seal = build_canary_plan(cache)
    r1_rows, r1_seal = build_r1_plan(cache)
    write_json_once(result / "M7F_H3_PARENT_IDENTITY.json", parent)
    write_json_once(result / "M7F_H3_METHOD_FREEZE.json", method)
    write_csv_once(result / "M7F_H3_DBLP_PATH_SET_RESOLUTION.csv", resolution)
    write_csv_once(result / "DBLP_H3_CANARY_PLAN.csv", canary_rows)
    write_json_once(result / "DBLP_H3_CANARY_PLAN_SEAL.json", canary_seal)
    write_csv_once(result / "DBLP_H3_R1_RUN_PLAN.csv", r1_rows)
    write_json_once(result / "DBLP_H3_R1_RUN_PLAN_SEAL.json", r1_seal)
    git_basis = {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_worktree_clean": True,
        "taskbook_filename": TASKBOOK_FILENAME,
        "taskbook_sha256": TASKBOOK_SHA256,
        "parent_identity": parent["identity"],
        "aminer_correction_identity": correction["identity"],
        "method_freeze_identity": method["identity"],
        "protocol_scope_audit_identity": protocol["identity"],
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "dataset": "dblp",
        "dataset_identity": DATASET_IDENTITY,
        "cache_identity": CACHE_IDENTITY,
        "path_library_identity": PATH_LIBRARY_IDENTITY,
        "all_sets_identity": ALL_SETS_IDENTITY,
        "legal_path_count": LEGAL_PATH_COUNT,
        "fixed_set_cardinality": SET_CARDINALITY,
        "legal_set_count": LEGAL_SET_COUNT,
        "public_seed_role_identities": roles,
        "canary_plan_seal_identity": canary_seal["identity"],
        "r1_plan_seal_identity": r1_seal["identity"],
        "r1_model_seeds": list(R1_MODEL_SEEDS),
        "high_fidelity_model_seeds": list(SEARCH_MODEL_SEEDS),
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
    git_config = envelope("metachorus.v53.m7f-h3.git-and-config-identity.v1", "PASS", git_basis)
    write_json_once(result / "M7F_H3_GIT_AND_CONFIG_IDENTITY.json", git_config)
    complete = envelope(
        "metachorus.v53.m7f-h3.preflight-complete.v1",
        "READY_FOR_DBLP_H3_4GPU_DEPLOYMENT",
        {
            "parent_identity": parent["identity"],
            "aminer_correction_identity": correction["identity"],
            "method_freeze_identity": method["identity"],
            "protocol_scope_audit_identity": protocol["identity"],
            "git_and_config_identity": git_config["identity"],
            "all_set_resolution_identity": canonical_json_hash(resolution),
            "all_sets_resolved": len(resolution) == LEGAL_SET_COUNT,
            "canary_plan_seal_identity": canary_seal["identity"],
            "r1_plan_seal_identity": r1_seal["identity"],
            "r1_run_count": len(r1_rows),
            "r1_model_seed_count": len(R1_MODEL_SEEDS),
            "scientific_training_started": False,
            "ten_seed_expansion_started": False,
            "labels_mounted": False,
        },
    )
    write_json_once(result / "M7F_H3_PREFLIGHT_COMPLETE.json", complete)
    write_text_once(
        result / "plan.md",
        "# M7F-H3 AMiner Correction and DBLP Four-GPU Plan\n\n"
        "1. Preserve the H2 AMiner unsafe-selection history and append the zero-GPU Manual fallback correction.\n"
        "2. Verify the frozen protocol source and exact DBLP 12-path, cardinality-three, 220-set identity.\n"
        "3. Verify four RTX 4090 devices and run four 20-step engineering canaries; discard canary state.\n"
        "4. Run all 220 sets with one frozen search seed to update 300 and seal an exact 24-set pool.\n"
        "5. Run the sealed 24 sets with three search seeds to 2000 and seal an exact eight-set shortlist.\n"
        "6. Continue the eight sets to 3000, seal one set, then run fresh paired three-seed preliminary evaluation.\n"
        "7. Do not start ten-seed confirmation; publish only the minimal ZIP and SHA-256 sidecar.\n",
    )
    return complete


__all__ = ["git_identity", "audit_parent_packages", "build_preflight"]
