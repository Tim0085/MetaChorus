"""Fresh, append-only formal preflight for M8-F3R-RERUN-2G."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.m4a_h8.constants import ROLE_MANIFEST_SHA256 as FREEBASE_ROLE_MANIFEST_SHA256

from .common import (
    canonical_json_hash,
    envelope,
    read_csv,
    read_json,
    sha256_file,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    ABLATION_DATASETS,
    CACHE_IDENTITIES,
    CANDIDATE_BANK_IDENTITIES,
    DATASET_IDENTITIES,
    DBLP_PUBLIC_ROLE_DATASET_IDENTITY,
    EVALUATION_CONTRACTS,
    PATH_LIBRARY_IDENTITIES,
    SELECTION_FIELDS,
    SOURCE_TASKBOOK_FILENAME,
    SOURCE_TASKBOOK_SHA256,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
    TASKBOOK_STAGE,
)
from .evidence import (
    audit_parent_packages,
    existing_safety_rows,
    load_dblp_selection_inputs,
    load_freebase_selection_inputs,
)
from .planning import (
    build_canary_plan,
    build_evaluation_matrix,
    build_two_gpu_group_schedule,
    resource_forecast,
    seal_matrix,
)
from .science import validate_existing_safety_evidence
from .selection import select_dataset_variants, validate_selection_rows
from .lifecycle import publish_json_state, publish_snapshot


PUBLIC_ROLE_NAMES = (
    "FAST_LABEL_ROLE_MANIFEST.json",
    "FAST_LABEL_ROLE_SEAL.json",
    "FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json",
    "FAST_SEED_NAMESPACE_MANIFEST.json",
    "FAST_SEED_COLLISION_AUDIT.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def git_identity(repo_root: str | Path) -> tuple[Path, str, str]:
    raw = Path(repo_root).absolute()
    if raw.is_symlink():
        raise ValueError("formal source root cannot be a symlink")
    repo = raw.resolve(strict=True)
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("M8-F3R-RERUN-2G formal preflight requires a clean worktree")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if len(commit) != 40 or len(tree) != 40:
        raise RuntimeError("source commit/tree is not canonical")
    return repo, commit, tree


def _fresh_result(path_value: str | Path) -> Path:
    raw = Path(path_value).absolute()
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise ValueError("result path contains a symlink component")
    if raw.exists():
        if not raw.is_dir() or any(raw.iterdir()):
            raise FileExistsError("formal result root must be fresh and empty")
    else:
        raw.mkdir(parents=True)
    return raw.resolve(strict=True)


def _taskbook(path_value: str | Path) -> Path:
    raw = Path(path_value).absolute()
    if raw.is_symlink():
        raise ValueError("taskbook cannot be a symlink")
    path = raw.resolve(strict=True)
    if path.name != TASKBOOK_FILENAME or sha256_file(path) != TASKBOOK_SHA256:
        raise ValueError("M8-F3R-RERUN-2G taskbook identity mismatch")
    return path


def _cache_metadata(cache_value: str | Path, dataset: str) -> dict[str, Any]:
    raw = Path(cache_value).absolute()
    if raw.is_symlink():
        raise ValueError(f"cache root cannot be a symlink: {dataset}")
    root = raw.resolve(strict=True)
    complete = read_json(root / "SANITIZED_CACHE_COMPLETE.json")
    manifest = read_json(root / "SANITIZED_CACHE_MANIFEST.json")
    data = read_json(root / "metadata" / "data_identity.json")
    projection = read_json(root / "metadata" / "projection_identity.json")
    all_sets = read_json(root / "candidates" / "all_sets.json")
    cache_identity = str(complete.get("cache_identity", ""))
    source_identity = str(data.get("source_identity_sha256", data.get("structural_data_identity", "")))
    all_sets_identity = str(all_sets.get("all_sets_sha256", all_sets.get("all_sets_identity", "")))
    path_library_identity = str(all_sets.get("path_library_sha256", all_sets.get("path_library_identity", "")))
    projection_identity = str(projection.get("projection_identity", canonical_json_hash(projection)))
    if (
        complete.get("status") != "COMPLETE"
        or cache_identity != CACHE_IDENTITIES[dataset]
        or source_identity != DATASET_IDENTITIES[dataset]
        or all_sets_identity != CANDIDATE_BANK_IDENTITIES[dataset]
        or path_library_identity != PATH_LIBRARY_IDENTITIES[dataset]
        or manifest.get("cache_identity") != cache_identity
        or str(all_sets.get("dataset")) != dataset
    ):
        raise ValueError(f"sealed label-free cache identity mismatch: {dataset}")
    sets = []
    for row in all_sets.get("sets", []):
        sets.append(
            {
                "set_id": str(row.get("set_id", row.get("candidate_set_id", ""))),
                "set_hash": str(row.get("set_sha256", row.get("candidate_set_hash", ""))),
                "ordinal": int(row.get("ordinal", row.get("candidate_set_ordinal", -1))),
                "path_ids": [str(value) for value in row.get("path_ids", [])],
            }
        )
    if not sets or len({row["set_hash"] for row in sets}) != len(sets):
        raise ValueError(f"candidate bank is empty or duplicated: {dataset}")
    return {
        "dataset": dataset,
        "root": str(root),
        "cache_identity": cache_identity,
        "dataset_identity": source_identity,
        "projection_identity": projection_identity,
        "candidate_bank_identity": all_sets_identity,
        "path_library_identity": path_library_identity,
        "sets": sets,
        "cache_manifest_sha256": sha256_file(root / "SANITIZED_CACHE_MANIFEST.json"),
        "cache_complete_sha256": sha256_file(root / "SANITIZED_CACHE_COMPLETE.json"),
    }


def _validate_selected_sets(selection_rows: Sequence[Mapping[str, Any]], caches: Mapping[str, Mapping[str, Any]]) -> None:
    for row in selection_rows:
        dataset = str(row["dataset"])
        matches = [value for value in caches[dataset]["sets"] if value["set_hash"] == row["set_hash"]]
        paths = [str(value) for value in json.loads(str(row["path_ids"]))]
        if len(matches) != 1 or matches[0]["set_id"] != row["set_id"] or matches[0]["path_ids"] != paths:
            raise ValueError("selected candidate does not match the sealed cache bank")


def _copy_public_roles(
    *,
    freebase_manifest: str | Path,
    freebase_mount_audit: str | Path,
    dblp_role_root: str | Path,
    result: Path,
) -> dict[str, str]:
    freebase_raw = Path(freebase_manifest).absolute()
    if freebase_raw.is_symlink():
        raise ValueError("Freebase H5 role manifest cannot be a symlink")
    freebase = freebase_raw.resolve(strict=True)
    if freebase.name != "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json" or sha256_file(freebase) != FREEBASE_ROLE_MANIFEST_SHA256:
        raise ValueError("Freebase H5 role manifest identity mismatch")
    freebase_value = read_json(freebase)
    freebase_rows = [row for row in freebase_value.get("datasets", []) if row.get("dataset") == "freebase"]
    if len(freebase_rows) != 1:
        raise ValueError("Freebase H5 role manifest lacks the exact dataset row")
    write_text_once(result / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json", freebase.read_text(encoding="utf-8"))
    freebase_audit_raw = Path(freebase_mount_audit).absolute()
    if freebase_audit_raw.is_symlink():
        raise ValueError("Freebase H5 private-search audit cannot be a symlink")
    freebase_audit = freebase_audit_raw.resolve(strict=True)
    freebase_audit_value = read_json(freebase_audit)
    freebase_members = [
        row for row in freebase_audit_value.get("members", []) if row.get("dataset") == "freebase"
    ]
    if (
        freebase_audit.name != "TC_LPMSO_H5_PRIVATE_SEARCH_MOUNT_AUDIT.json"
        or freebase_audit_value.get("status") != "PASS"
        or freebase_audit_value.get("contains_full_labels") is not False
        or freebase_audit_value.get("contains_h5_locked_report") is not False
        or len(freebase_members) != 1
        or freebase_members[0].get("member_name") != "freebase_global_search_pool.npz"
    ):
        raise ValueError("Freebase H5 private-search audit identity is invalid")
    write_text_once(
        result / "TC_LPMSO_H5_PRIVATE_SEARCH_MOUNT_AUDIT.json",
        freebase_audit.read_text(encoding="utf-8"),
    )

    role_raw = Path(dblp_role_root).absolute()
    if role_raw.is_symlink():
        raise ValueError("DBLP role root cannot be a symlink")
    role_root = role_raw.resolve(strict=True)
    documents = {name: read_json(role_root / name) for name in PUBLIC_ROLE_NAMES}
    manifest = documents["FAST_LABEL_ROLE_MANIFEST.json"]
    seal = documents["FAST_LABEL_ROLE_SEAL.json"]
    audit = documents["FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json"]
    dblp_rows = [row for row in manifest.get("basis", {}).get("datasets", []) if row.get("dataset") == "dblp"]
    if (
        manifest.get("status") != "SEALED_INPUT"
        or manifest.get("identity") != canonical_json_hash(manifest.get("basis", {}))
        or manifest.get("basis", {}).get("raw_labels_embedded") is not False
        or manifest.get("basis", {}).get("training_process_mounts_labels") is not False
        or len(dblp_rows) != 1
        or dblp_rows[0].get("dataset_identity") != DBLP_PUBLIC_ROLE_DATASET_IDENTITY
        or seal.get("status") != "SEALED_BEFORE_SEARCH_TRAINING"
        or seal.get("identity") != canonical_json_hash(seal.get("basis", {}))
        or seal.get("basis", {}).get("role_manifest_identity") != manifest.get("identity")
        or audit.get("status") != "PASS"
        or audit.get("identity") != seal.get("basis", {}).get("private_mount_audit_identity")
        or audit.get("basis", {}).get("contains_full_labels") is not False
        or audit.get("basis", {}).get("contains_prelim_locked_report") is not False
    ):
        raise ValueError("DBLP public label/search role identity chain is invalid")
    identities = {
        "freebase_role_manifest_sha256": sha256_file(freebase),
        "freebase_private_mount_audit_sha256": sha256_file(freebase_audit),
    }
    for name in PUBLIC_ROLE_NAMES:
        source = role_root / name
        write_text_once(result / name, source.read_text(encoding="utf-8"))
        identities[name] = sha256_file(source)
    return identities


def _fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("cannot publish a headerless governed CSV")
    values: list[str] = []
    for row in rows:
        for key in row:
            if key not in values:
                values.append(str(key))
    return values


def build_preflight(
    *,
    repo_root: str | Path,
    taskbook: str | Path,
    parent_packages: Mapping[str, str | Path],
    cache_roots: Mapping[str, str | Path],
    freebase_role_manifest: str | Path,
    freebase_mount_audit: str | Path,
    dblp_role_root: str | Path,
    result_root: str | Path,
    existing_rows: Sequence[Mapping[str, Any]] = (),
    estimated_remote_storage_bytes: int = 0,
) -> dict[str, Any]:
    repo, source_commit, source_tree = git_identity(repo_root)
    taskbook_path = _taskbook(taskbook)
    result = _fresh_result(result_root)
    started_status = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.execution-status.v1",
        "STARTED",
        {
            "stage": TASKBOOK_STAGE,
            "taskbook_sha256": TASKBOOK_SHA256,
            "terminal_status": "STARTED",
            "gpu_count": 2,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "scientific_training_started": False,
            "selection_sealed": False,
            "evaluation_matrix_sealed": False,
            "labels_mounted_during_training": False,
        },
    )
    publish_json_state(result / "ABLATION_EXECUTION_STATUS.json", started_status)
    publish_json_state(
        result / "M8_F3R_RERUN_2G_GIT_AND_CONFIG_IDENTITY.json",
        envelope(
            "metachorus.v53.m8-f3r-rerun-2g.git-and-config-identity.v1",
            "STARTED",
            {
                "stage": TASKBOOK_STAGE,
                "source_commit": source_commit,
                "source_tree": source_tree,
                "worktree_clean": True,
                "taskbook_filename": TASKBOOK_FILENAME,
                "taskbook_sha256": TASKBOOK_SHA256,
                "scientific_training_started": False,
                "labels_mounted_during_training": False,
            },
        ),
    )
    publish_json_state(
        result / "M8_F3R_RERUN_2G_LAUNCH_RECEIPT.json",
        envelope(
            "metachorus.v53.m8-f3r-rerun-2g.launch-receipt.v1",
            "STARTED",
            {
                "stage": TASKBOOK_STAGE,
                "taskbook_sha256": TASKBOOK_SHA256,
                "worker_count": 2,
                "scientific_training_started": False,
                "labels_mounted_during_training": False,
            },
        ),
    )
    if set(cache_roots) != set(ABLATION_DATASETS):
        raise ValueError("cache roots must cover exactly Freebase and DBLP")
    parent_rows = audit_parent_packages(parent_packages)
    parent_identity = canonical_json_hash(parent_rows)
    caches = {dataset: _cache_metadata(cache_roots[dataset], dataset) for dataset in ABLATION_DATASETS}
    role_identities = _copy_public_roles(
        freebase_manifest=freebase_role_manifest,
        freebase_mount_audit=freebase_mount_audit,
        dblp_role_root=dblp_role_root,
        result=result,
    )

    freebase_inputs = load_freebase_selection_inputs(
        h5_package=parent_packages["FREEBASE_H5"],
        h7_package=parent_packages["FREEBASE_H7_ACM_SAFETY"],
    )
    dblp_inputs = load_dblp_selection_inputs(h3_package=parent_packages["DBLP_H3"])
    selections = []
    components = []
    for dataset, values in (("freebase", freebase_inputs), ("dblp", dblp_inputs)):
        selected, detail = select_dataset_variants(
            dataset,
            values["candidates"],
            values["safety"],
            exploration_only_hashes=values["exploration_only_hashes"],
        )
        selections.extend(selected)
        components.extend(detail)
    validate_selection_rows(selections)
    _validate_selected_sets(selections, caches)

    selection_time = utc_now()
    selection_basis = {
        "stage": TASKBOOK_STAGE,
        "taskbook_sha256": TASKBOOK_SHA256,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "parent_package_audit_identity": parent_identity,
        "cache_identities": {dataset: caches[dataset]["cache_identity"] for dataset in ABLATION_DATASETS},
        "dataset_identities": {dataset: caches[dataset]["dataset_identity"] for dataset in ABLATION_DATASETS},
        "selection_input_identities": {"freebase": freebase_inputs["identity"], "dblp": dblp_inputs["identity"]},
        "row_count": len(selections),
        "rows_identity": canonical_json_hash(selections),
        "selection_components_identity": canonical_json_hash(components),
        "locked_report_access_count": 0,
        "search_metric_source_only": True,
        "algorithm_retuning": False,
        "sealed_at_utc": selection_time,
    }
    selection_seal = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.selection-seal.v1",
        "SELECTION_SEALED_BEFORE_LOCKED_EVALUATION",
        selection_basis,
    )
    matrix, reuse = build_evaluation_matrix(
        selections,
        dataset_identities={dataset: caches[dataset]["dataset_identity"] for dataset in ABLATION_DATASETS},
        evaluation_contracts=EVALUATION_CONTRACTS,
        existing_rows=existing_rows,
    )
    matrix_time = utc_now()
    if matrix_time <= selection_time:
        matrix_time = (datetime.fromisoformat(selection_time) + timedelta(microseconds=1)).isoformat()
    matrix_seal = seal_matrix(matrix, selection_seal_identity=selection_seal["identity"], sealed_at_utc=matrix_time)
    schedule = build_two_gpu_group_schedule(matrix)
    forecast = resource_forecast(
        matrix,
        schedule,
        search_continuation_runs=0,
        estimated_remote_storage_bytes=int(estimated_remote_storage_bytes),
    )
    if forecast.get("status") != "PASS":
        raise RuntimeError("formal evaluation forecast exceeds a frozen hard resource gate")
    canary = build_canary_plan()
    _validate_selected_sets(canary, caches)
    canary_seal = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.canary-plan-seal.v1",
        "SEALED_BEFORE_CANARY_GPU_TRAINING",
        {
            "row_count": len(canary),
            "rows_identity": canonical_json_hash(canary),
            "source_commit": source_commit,
            "source_tree": source_tree,
            "taskbook_sha256": TASKBOOK_SHA256,
            "labels_mounted_during_training": False,
            "scientific_result": False,
            "sealed_at_utc": utc_now(),
        },
    )
    safety_rows = existing_safety_rows(parent_rows)
    validate_existing_safety_evidence(safety_rows)

    protected_files = (
        "model.py",
        "utils.py",
        "src/metachorus_v53/training/o0_codefaithful_runner.py",
        "src/metachorus_v53/m8_f3r_rerun_2g/selection.py",
        "src/metachorus_v53/m8_f3r_rerun_2g/science.py",
    )
    protected_hashes = {}
    for relative in protected_files:
        path = repo / Path(*relative.split("/"))
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        protected_hashes[relative] = sha256_file(path)
    source_taskbook = taskbook_path.parent / SOURCE_TASKBOOK_FILENAME
    source_taskbook_status = "AVAILABLE_EXACT_SHA256" if source_taskbook.is_file() and sha256_file(source_taskbook) == SOURCE_TASKBOOK_SHA256 else "DECLARED_SOURCE_TASKBOOK_NOT_AVAILABLE"
    git_config = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.git-and-config-identity.v1",
        "PASS",
        {
            "stage": TASKBOOK_STAGE,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "worktree_clean": True,
            "taskbook_filename": TASKBOOK_FILENAME,
            "taskbook_sha256": TASKBOOK_SHA256,
            "source_taskbook_filename": SOURCE_TASKBOOK_FILENAME,
            "source_taskbook_declared_sha256": SOURCE_TASKBOOK_SHA256,
            "source_taskbook_status": source_taskbook_status,
            "parent_package_audit_identity": parent_identity,
            "cache_identities": {dataset: caches[dataset]["cache_identity"] for dataset in ABLATION_DATASETS},
            "dataset_identities": {dataset: caches[dataset]["dataset_identity"] for dataset in ABLATION_DATASETS},
            "projection_identities": {dataset: caches[dataset]["projection_identity"] for dataset in ABLATION_DATASETS},
            "candidate_bank_identities": {dataset: caches[dataset]["candidate_bank_identity"] for dataset in ABLATION_DATASETS},
            "path_library_identities": {dataset: caches[dataset]["path_library_identity"] for dataset in ABLATION_DATASETS},
            "public_role_identities": role_identities,
            "protected_file_sha256": protected_hashes,
            "selection_seal_identity": selection_seal["identity"],
            "evaluation_matrix_seal_identity": matrix_seal["identity"],
            "canary_plan_seal_identity": canary_seal["identity"],
            "labels_mounted_during_training": False,
            "algorithm_retuning": False,
        },
    )

    write_csv_once(result / "ABLATION_PARENT_PACKAGE_AUDIT.csv", parent_rows, _fields(parent_rows))
    write_csv_once(result / "ABLATION_SELECTIONS_FREEBASE.csv", [row for row in selections if row["dataset"] == "freebase"], SELECTION_FIELDS)
    write_csv_once(result / "ABLATION_SELECTIONS_DBLP.csv", [row for row in selections if row["dataset"] == "dblp"], SELECTION_FIELDS)
    write_csv_once(result / "ABLATION_SELECTIONS_ALL.csv", selections, SELECTION_FIELDS)
    write_csv_once(result / "ABLATION_SELECTION_COMPONENTS.csv", components, ("dataset", "ablation_variant", "component"))
    write_json_once(result / "ABLATION_SELECTION_SEAL.json", selection_seal)
    publish_json_state(
        result / "ABLATION_EXECUTION_STATUS.json",
        envelope(
            "metachorus.v53.m8-f3r-rerun-2g.execution-status.v1",
            "SELECTION_SEALED",
            {
                "stage": TASKBOOK_STAGE,
                "taskbook_sha256": TASKBOOK_SHA256,
                "terminal_status": "SELECTION_SEALED",
                "gpu_count": 2,
                "source_commit": source_commit,
                "source_tree": source_tree,
                "scientific_training_started": False,
                "selection_sealed": True,
                "selection_seal_identity": selection_seal["identity"],
                "evaluation_matrix_sealed": False,
                "labels_mounted_during_training": False,
            },
        ),
    )
    selection_snapshot = publish_snapshot(
        result,
        prefix="SNAPSHOT_0_SELECTION",
        member_names=(
            "ABLATION_PARENT_PACKAGE_AUDIT.csv",
            "ABLATION_SELECTIONS_FREEBASE.csv",
            "ABLATION_SELECTIONS_DBLP.csv",
            "ABLATION_SELECTIONS_ALL.csv",
            "ABLATION_SELECTION_COMPONENTS.csv",
            "ABLATION_SELECTION_SEAL.json",
            "ABLATION_EXECUTION_STATUS.json",
            "M8_F3R_RERUN_2G_GIT_AND_CONFIG_IDENTITY.json",
            "M8_F3R_RERUN_2G_LAUNCH_RECEIPT.json",
        ),
    )
    write_csv_once(result / "ABLATION_EXACT_REUSE_REGISTRY.csv", reuse, _fields(reuse))
    write_csv_once(result / "ABLATION_EVALUATION_MATRIX.csv", matrix, _fields(matrix))
    write_json_once(result / "ABLATION_EVALUATION_MATRIX_SEAL.json", matrix_seal)
    write_json_once(result / "ABLATION_RESOURCE_FORECAST.json", forecast)
    write_csv_once(result / "ABLATION_TWO_GPU_GROUP_SCHEDULE.csv", schedule, _fields(schedule))
    write_csv_once(result / "M8_F3R_RERUN_2G_CANARY_PLAN.csv", canary, _fields(canary))
    write_json_once(result / "M8_F3R_RERUN_2G_CANARY_PLAN_SEAL.json", canary_seal)
    write_csv_once(result / "SAFETY_GATE_ABLATION_EXISTING_EVIDENCE.csv", safety_rows, _fields(safety_rows))
    publish_json_state(result / "M8_F3R_RERUN_2G_GIT_AND_CONFIG_IDENTITY.json", git_config)
    complete = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.preflight-complete.v1",
        "READY_FOR_M8_F3R_RERUN_2G_REMOTE_DEPLOYMENT",
        {
            "git_and_config_identity": git_config["identity"],
            "parent_package_audit_identity": parent_identity,
            "selection_seal_identity": selection_seal["identity"],
            "selection_snapshot_identity": selection_snapshot["snapshot_identity"],
            "selection_snapshot_sha256": selection_snapshot["snapshot_sha256"],
            "selection_snapshot_name": selection_snapshot["snapshot_name"],
            "evaluation_matrix_seal_identity": matrix_seal["identity"],
            "canary_plan_seal_identity": canary_seal["identity"],
            "new_run_count": sum(row["status"] == "NEW_RUN_REQUIRED" for row in matrix),
            "exact_reuse_count": sum(row["status"] == "EXACT_REUSE" for row in matrix),
            "labels_mounted_during_training": False,
            "scientific_training_started": False,
        },
    )
    write_json_once(result / "M8_F3R_RERUN_2G_PREFLIGHT_COMPLETE.json", complete)
    publish_json_state(
        result / "ABLATION_EXECUTION_STATUS.json",
        envelope(
            "metachorus.v53.m8-f3r-rerun-2g.execution-status.v1",
            "EVALUATION_MATRIX_SEALED",
            {
                "stage": TASKBOOK_STAGE,
                "taskbook_sha256": TASKBOOK_SHA256,
                "terminal_status": "EVALUATION_MATRIX_SEALED",
                "gpu_count": 2,
                "source_commit": source_commit,
                "source_tree": source_tree,
                "scientific_training_started": False,
                "selection_sealed": True,
                "selection_seal_identity": selection_seal["identity"],
                "evaluation_matrix_sealed": True,
                "evaluation_matrix_seal_identity": matrix_seal["identity"],
                "labels_mounted_during_training": False,
            },
        ),
    )
    write_text_once(
        result / "plan.md",
        "# M8-F3R-RERUN-2G formal plan\n\n"
        "1. Verify the exact two-RTX4090 runtime and close four engineering canaries.\n"
        "2. Run only NEW_RUN_REQUIRED rows from fresh update 0 to 3000 with no labels.\n"
        "3. Seal the physical run registry before opening Freebase H5 or DBLP preliminary locked roles.\n"
        "4. Recompute all 60 variant rows plus six Manual controls, finalize, and publish only the MIN ZIP and sidecar.\n",
    )
    return complete


__all__ = ["build_preflight", "git_identity"]
