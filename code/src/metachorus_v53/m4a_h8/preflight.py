"""H8 parent audit and pre-training freeze without locked-label access."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from typing import Any, Mapping
from zipfile import ZipFile

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    DEVELOPMENT_MODEL_SEEDS,
    EVALUATOR_CONTRACT_SHA256,
    EVALUATOR_SEAL_SHA256,
    EXPECTED_CANARY_RUNS,
    EXPECTED_SCIENCE_RUNS,
    FROZEN_CACHE_IDENTITIES,
    FROZEN_PATH_SETS,
    FROZEN_PROJECTION_IDENTITIES,
    FREEBASE_CANDIDATE_BANK_SHA256,
    FREEBASE_H7_SELECTED_PROJECTION_HASHES,
    FREEBASE_PATH_LIBRARY_SHA256,
    H5_BANK_SEEDS,
    H5_CLASSIFICATION_SPLIT_SEEDS,
    H5_CLUSTER_SEARCH_SEEDS,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_REGISTRY_IDENTITY,
    PARENT_SELECTION_IDENTITY,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    PARENT_TERMINAL_STATUS,
    ROLE_MANIFEST_SHA256,
    ROLE_SEAL_SHA256,
    S_EVAL_MODEL_SEEDS,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
)
from .planning import (
    PLAN_FIELDS,
    TEMPLATE_FIELDS,
    build_canary_plan,
    build_science_draft,
)


PARENT_ZIP_REQUIRED = {
    "evidence/results/M4A_H7_COMPLETE.json",
    "evidence/results/SOURCE_DEPENDENCY_MANIFEST_M4A_H7.json",
    "evidence/results/TC_LPMSO_H7_SELECTED_PATH_SET_REGISTRY.json",
    "evidence/results/TC_LPMSO_H7_S_EVAL_CONFIRMATION_MATRIX_PENDING.csv",
    "PACKAGE_MEMBER_MANIFEST.json",
    "PACKAGE_CHECKSUMS.sha256",
}

H5_ROLE_ARTIFACTS = {
    "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json": ROLE_MANIFEST_SHA256,
    "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_SEAL.json": ROLE_SEAL_SHA256,
    "TC_LPMSO_H5_SEARCH_EVALUATOR_CONTRACT.yaml": EVALUATOR_CONTRACT_SHA256,
    "TC_LPMSO_H5_SEARCH_EVALUATOR_SEAL.json": EVALUATOR_SEAL_SHA256,
}


def _git(repo_root: Path, expression: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", expression],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _clean_source(repo_root: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if status:
        raise RuntimeError("H8 source worktree must be clean and committed")
    return {
        "source_commit": _git(repo_root, "HEAD"),
        "source_tree": _git(repo_root, "HEAD^{tree}"),
        "parent_h7_commit": PARENT_SOURCE_COMMIT,
        "parent_h7_tree": PARENT_SOURCE_TREE,
        "worktree_clean": True,
        "protocol_only_successor": True,
    }


def _safe_zip_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError("H8 parent ZIP contains a non-portable member")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("H8 parent ZIP contains an unsafe member")
    return path


def audit_parent_package(parent_package: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    package = parent_package.resolve(strict=True)
    if (
        package.name != PARENT_PACKAGE_FILENAME
        or package.stat().st_size != PARENT_PACKAGE_SIZE_BYTES
        or sha256_file(package) != PARENT_PACKAGE_SHA256
    ):
        raise ValueError("H8 parent package identity mismatch")
    payloads: dict[str, bytes] = {}
    with ZipFile(package) as archive:
        infos = archive.infolist()
        names = [info.filename.rstrip("/") for info in infos if not info.is_dir()]
        if len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names):
            raise ValueError("H8 parent ZIP contains duplicate members")
        for info in infos:
            _safe_zip_member(info.filename.rstrip("/") if info.is_dir() else info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("H8 parent ZIP symlink forbidden")
        if archive.testzip() is not None:
            raise ValueError("H8 parent ZIP CRC failure")
        missing = PARENT_ZIP_REQUIRED - set(names)
        if missing:
            raise ValueError(f"H8 parent ZIP missing frozen evidence: {sorted(missing)}")
        for name in PARENT_ZIP_REQUIRED:
            payloads[name] = archive.read(name)
        manifest = json.loads(payloads["PACKAGE_MEMBER_MANIFEST.json"].decode("utf-8"))
        manifest_rows = manifest.get("members")
        if (
            manifest.get("schema_version")
            != "metachorus.v53.m4a-h7.package-member-manifest.v1"
            or manifest.get("status") != "PASS"
            or not isinstance(manifest_rows, list)
            or int(manifest.get("member_count", -1)) != len(manifest_rows)
            or manifest.get("manifest_identity") != canonical_json_hash(manifest_rows)
        ):
            raise ValueError("H8 parent package member manifest is invalid")
        by_path: dict[str, Mapping[str, Any]] = {}
        for row in manifest_rows:
            if not isinstance(row, Mapping):
                raise ValueError("H8 parent package manifest row is invalid")
            relative = _safe_zip_member(str(row.get("relative_path", ""))).as_posix()
            if relative in by_path:
                raise ValueError("H8 parent package manifest contains duplicate members")
            by_path[relative] = row
        expected_names = set(by_path) | {
            "PACKAGE_MEMBER_MANIFEST.json",
            "PACKAGE_CHECKSUMS.sha256",
        }
        if {"PACKAGE_MEMBER_MANIFEST.json", "PACKAGE_CHECKSUMS.sha256"} & set(by_path):
            raise ValueError("H8 parent package indexes cannot be business manifest members")
        if set(names) != expected_names:
            raise ValueError("H8 parent package member set differs from its manifest")
        checksum_rows: dict[str, str] = {}
        for line in payloads["PACKAGE_CHECKSUMS.sha256"].decode("utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            relative = _safe_zip_member(relative).as_posix()
            if relative in checksum_rows:
                raise ValueError("H8 parent package checksum index contains duplicates")
            checksum_rows[relative] = digest
        if checksum_rows != {
            relative: str(row.get("sha256")) for relative, row in by_path.items()
        }:
            raise ValueError("H8 parent package checksum index mismatch")
        for relative, row in by_path.items():
            content = archive.read(relative)
            if (
                len(content) != int(row.get("size_bytes", -1))
                or hashlib.sha256(content).hexdigest() != str(row.get("sha256"))
            ):
                raise ValueError(f"H8 parent package member mismatch: {relative}")
    complete = json.loads(payloads["evidence/results/M4A_H7_COMPLETE.json"].decode("utf-8"))
    source = json.loads(
        payloads["evidence/results/SOURCE_DEPENDENCY_MANIFEST_M4A_H7.json"].decode("utf-8")
    )
    registry = json.loads(
        payloads["evidence/results/TC_LPMSO_H7_SELECTED_PATH_SET_REGISTRY.json"].decode("utf-8")
    )
    if (
        complete.get("status") != PARENT_TERMINAL_STATUS
        or registry.get("status") != PARENT_TERMINAL_STATUS
        or registry.get("identity") != PARENT_REGISTRY_IDENTITY
        or registry.get("basis", {}).get("selection_identity") != PARENT_SELECTION_IDENTITY
        or source.get("basis", {}).get("source_commit") != PARENT_SOURCE_COMMIT
        or source.get("basis", {}).get("source_tree") != PARENT_SOURCE_TREE
    ):
        raise ValueError("H8 parent ZIP does not preserve the frozen H7 decision")
    basis = {
        "package_name": package.name,
        "package_sha256": PARENT_PACKAGE_SHA256,
        "package_size_bytes": PARENT_PACKAGE_SIZE_BYTES,
        "zip_member_count": len(names),
        "manifest_member_count": len(by_path),
        "manifest_identity": manifest["manifest_identity"],
        "checksum_index_verified": True,
        "crc_pass": True,
        "unsafe_member_count": 0,
        "h7_terminal_status": PARENT_TERMINAL_STATUS,
        "h7_selected_registry_identity": PARENT_REGISTRY_IDENTITY,
        "h7_selection_identity": PARENT_SELECTION_IDENTITY,
        "h7_source_commit": PARENT_SOURCE_COMMIT,
        "h7_source_tree": PARENT_SOURCE_TREE,
        "h5_locked_report_read": False,
        "s_eval_training_started": False,
    }
    return envelope(
        "metachorus.v53.m4a-h8.parent-package-audit.v1", "PASS", basis
    ), payloads


def _csv_bytes(payload: bytes) -> list[dict[str, str]]:
    return [dict(row) for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))]


def _validate_parent_matrix(rows: list[Mapping[str, Any]]) -> None:
    expected = {
        (
            str(item["dataset"]),
            str(item["candidate_set_hash"]),
            int(seed),
        )
        for item in FROZEN_PATH_SETS
        for seed in S_EVAL_MODEL_SEEDS
    }
    observed = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"]))
        for row in rows
    }
    if (
        len(rows) != EXPECTED_SCIENCE_RUNS
        or len(observed) != EXPECTED_SCIENCE_RUNS
        or observed != expected
        or any(
            int(row["continuation_from_update"]) != 0
            or int(row["endpoint_update"]) != 3000
            or str(row["trainer"]) != "O0-S0"
            or str(row["training_labels_visible"]).casefold() != "false"
            or str(row["training_started"]).casefold() != "false"
            or str(row["status"]) != "PENDING_USER"
            for row in rows
        )
    ):
        raise ValueError("H8 parent pending matrix is not the frozen exact-50 design")


def _validate_h5_roles(h5_result_root: Path) -> dict[str, Any]:
    for name, expected in H5_ROLE_ARTIFACTS.items():
        path = h5_result_root / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"H8 H5 role/evaluator artifact mismatch: {name}")
    manifest = read_json(h5_result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json")
    seal = read_json(h5_result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_SEAL.json")
    datasets = {str(row.get("dataset")): row for row in manifest.get("datasets", [])}
    if (
        manifest.get("status") != "SEALED_INPUT"
        or set(datasets) != {"acm", "freebase"}
        or any(
            int(row.get("GLOBAL_SEARCH_POOL", {}).get("count", -1)) <= 0
            or int(row.get("H5_LOCKED_REPORT", {}).get("count", -1)) <= 0
            or row.get("node_disjoint") is not True
            or row.get("covers_all_target_nodes") is not True
            for row in datasets.values()
        )
        or seal.get("identity") is None
        or seal.get("basis", {}).get("training_process_mounts_labels") is not False
        or seal.get("basis", {}).get("one_h5_locked_report_per_dataset") is not True
    ):
        raise ValueError("H8 frozen H5 label roles are invalid")
    return {
        "role_manifest_sha256": ROLE_MANIFEST_SHA256,
        "role_seal_sha256": ROLE_SEAL_SHA256,
        "role_seal_identity": str(seal["identity"]),
        "evaluator_contract_sha256": EVALUATOR_CONTRACT_SHA256,
        "evaluator_seal_sha256": EVALUATOR_SEAL_SHA256,
        "classification_fit_role": "GLOBAL_SEARCH_POOL",
        "classification_test_role": "H5_LOCKED_REPORT",
        "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
        "clustering_metric_role": "H5_LOCKED_REPORT",
        "training_process_mounts_labels": False,
        "h5_locked_report_read": False,
    }


def _find_h7_templates(h7_result_root: Path, result_root: Path) -> list[dict[str, Any]]:
    run_rows = read_csv(h7_result_root / "TC_LPMSO_H7_R4_RUN_REGISTRY.csv")
    frozen_by_key = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): row
        for row in FROZEN_PATH_SETS
    }
    templates_dir = result_root / "templates"
    templates_dir.mkdir()
    output: list[dict[str, Any]] = []
    for key, frozen in sorted(frozen_by_key.items()):
        matches = sorted(
            (
                row
                for row in run_rows
                if (str(row["dataset"]), str(row["candidate_set_hash"])) == key
                and str(row["status"]) == "PASS"
                and str(row["training_reexecuted"]).casefold() == "false"
            ),
            key=lambda row: str(row["physical_run_id"]),
        )
        if len(matches) != len(DEVELOPMENT_MODEL_SEEDS):
            raise ValueError("H8 could not resolve three immutable H7 template runs")
        source_row = matches[0]
        resolved_configs: list[dict[str, Any]] = []
        resolved_markers: list[dict[str, Any]] = []
        for match in matches:
            match_attempt = (
                h7_result_root
                / "runs"
                / str(match["physical_run_id"])
                / f"attempt_{int(match['successful_attempt_index']):02d}"
            )
            match_config = match_attempt / "RESOLVED_CONFIG.json"
            match_marker = match_attempt / "H7_STAGE_003000_COMPLETE.json"
            if (
                match_attempt.is_symlink()
                or match_config.is_symlink()
                or not match_config.is_file()
                or match_marker.is_symlink()
                or not match_marker.is_file()
            ):
                raise ValueError("H8 template source artifact missing")
            resolved_configs.append(read_json(match_config))
            resolved_markers.append(read_json(match_marker))
        attempt_index = int(source_row["successful_attempt_index"])
        attempt = (
            h7_result_root
            / "runs"
            / str(source_row["physical_run_id"])
            / f"attempt_{attempt_index:02d}"
        )
        config = attempt / "RESOLVED_CONFIG.json"
        marker = attempt / "H7_STAGE_003000_COMPLETE.json"
        if (
            not config.is_file()
            or config.is_symlink()
            or not marker.is_file()
            or marker.is_symlink()
        ):
            raise ValueError("H8 template source artifact missing")
        config_value = resolved_configs[0]
        marker_value = resolved_markers[0]
        immutable_config_fields = (
            "dataset",
            "path_set_id",
            "candidate_set_hash",
            "candidate_set_ordinal",
            "path_ids",
            "cache_identity",
            "projection_identity",
            "candidate_bank_hash",
            "source_data_identity",
        )
        if any(
            any(value.get(field) != config_value.get(field) for field in immutable_config_fields)
            for value in resolved_configs[1:]
        ):
            raise ValueError("H8 three-seed template identities disagree")
        if (
            config_value.get("candidate_set_hash") != str(frozen["candidate_set_hash"])
            or config_value.get("path_set_id") != str(frozen["candidate_set_id"])
            or int(config_value.get("candidate_set_ordinal", -1))
            != int(frozen["candidate_set_ordinal"])
            or tuple(config_value.get("path_ids", [])) != tuple(frozen["path_ids"])
            or config_value.get("cache_identity")
            != FROZEN_CACHE_IDENTITIES[str(frozen["dataset"])]
            or config_value.get("projection_identity")
            != FROZEN_PROJECTION_IDENTITIES[str(frozen["dataset"])]
            or any(
                value.get("candidate_set_hash") != str(frozen["candidate_set_hash"])
                or value.get("status") != "PASS"
                for value in resolved_markers
            )
        ):
            raise ValueError("H8 template source identity drifted")
        relative = f"templates/{frozen['candidate_set_hash']}.json"
        destination = result_root / relative
        shutil.copy2(config, destination)
        if sha256_file(destination) != sha256_file(config):
            raise RuntimeError("H8 template copy verification failed")
        output.append(
            {
                "dataset": str(frozen["dataset"]),
                "role": str(frozen["role"]),
                "candidate_set_id": str(frozen["candidate_set_id"]),
                "candidate_set_hash": str(frozen["candidate_set_hash"]),
                "candidate_set_ordinal": int(frozen["candidate_set_ordinal"]),
                "path_ids_json": json.dumps(list(frozen["path_ids"]), ensure_ascii=False),
                "projection_identity": str(config_value["projection_identity"]),
                "projection_hashes_json": json.dumps(
                    list(FREEBASE_H7_SELECTED_PROJECTION_HASHES)
                    if frozen["dataset"] == "freebase" and frozen["role"] == "H7_SELECTED"
                    else [],
                    ensure_ascii=False,
                ),
                "template_config_relative_path": relative,
                "template_config_sha256": sha256_file(destination),
                "source_h7_physical_run_id": str(source_row["physical_run_id"]),
                "source_h7_marker_identity": str(marker_value["identity"]),
                "status": "PASS",
            }
        )
    return output


def build_preflight(
    *,
    repo_root: Path,
    taskbook: Path,
    parent_package: Path,
    h7_result_root: Path,
    h5_result_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    taskbook = taskbook.resolve(strict=True)
    parent_package = parent_package.resolve(strict=True)
    h7_result_root = h7_result_root.resolve(strict=True)
    h5_result_root = h5_result_root.resolve(strict=True)
    result_root = result_root.resolve() if result_root.exists() else result_root.absolute()
    if result_root.is_symlink() or (result_root.exists() and any(result_root.iterdir())):
        raise RuntimeError("H8_RESULT_ROOT_NOT_FRESH")
    if taskbook.name != TASKBOOK_FILENAME or sha256_file(taskbook) != TASKBOOK_SHA256:
        raise ValueError("H8 taskbook identity mismatch")
    source_identity = _clean_source(repo_root)
    parent_audit, parent_payloads = audit_parent_package(parent_package)
    parent_matrix = _csv_bytes(
        parent_payloads[
            "evidence/results/TC_LPMSO_H7_S_EVAL_CONFIRMATION_MATRIX_PENDING.csv"
        ]
    )
    _validate_parent_matrix(parent_matrix)
    h5_roles = _validate_h5_roles(h5_result_root)
    local_registry = read_json(
        h7_result_root / "TC_LPMSO_H7_SELECTED_PATH_SET_REGISTRY.json"
    )
    if (
        local_registry.get("identity") != PARENT_REGISTRY_IDENTITY
        or local_registry.get("basis", {}).get("selection_identity")
        != PARENT_SELECTION_IDENTITY
    ):
        raise ValueError("H8 local H7 result does not match the parent ZIP")

    result_root.mkdir(parents=True, exist_ok=True)
    try:
        write_json_once(result_root / "M4A_H8_PARENT_PACKAGE_AUDIT.json", parent_audit)
        parent_source_basis = {
            "parent_package_sha256": PARENT_PACKAGE_SHA256,
            "parent_source_commit": PARENT_SOURCE_COMMIT,
            "parent_source_tree": PARENT_SOURCE_TREE,
            "parent_registry_identity": PARENT_REGISTRY_IDENTITY,
            "parent_selection_identity": PARENT_SELECTION_IDENTITY,
            "h5_role_manifest_sha256": ROLE_MANIFEST_SHA256,
            "h5_role_seal_sha256": ROLE_SEAL_SHA256,
            "h5_locked_report_embedded": False,
            "private_labels_embedded": False,
            "s_eval_previously_started": False,
            "status": "PASS",
        }
        write_json_once(
            result_root / "M4A_H8_PARENT_SOURCE_IDENTITY.json",
            envelope(
                "metachorus.v53.m4a-h8.parent-source-identity.v1",
                "PASS",
                parent_source_basis,
            ),
        )
        freeze_basis = {
            "h7_parent_package_sha256": PARENT_PACKAGE_SHA256,
            "h7_registry_identity": PARENT_REGISTRY_IDENTITY,
            "h7_selection_identity": PARENT_SELECTION_IDENTITY,
            "h7_terminal_status": PARENT_TERMINAL_STATUS,
            "frozen_set_count": len(FROZEN_PATH_SETS),
            "candidate_sets_replaced": False,
            "second_choice_added": False,
            "algorithm_changed": False,
            "gate_changed": False,
            "future_model_seed_count": len(S_EVAL_MODEL_SEEDS),
            "future_seeds_previously_used": False,
            "training_started": False,
            "h5_locked_report_read": False,
        }
        write_json_once(
            result_root / "M4A_H8_H7_SELECTION_FREEZE.json",
            envelope("metachorus.v53.m4a-h8.h7-freeze-audit.v1", "PASS", freeze_basis),
        )
        template_rows = _find_h7_templates(h7_result_root, result_root)
        write_csv_once(
            result_root / "M4A_H8_PATH_SET_RESOLUTION_AUDIT.csv",
            template_rows,
            fieldnames=TEMPLATE_FIELDS,
        )
        mapping_rows = [
            {
                "dataset": str(row["dataset"]),
                "role": str(row["role"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "path_ids_json": str(row["path_ids_json"]),
                "projection_identity": str(row["projection_identity"]),
                "projection_hashes_json": str(row["projection_hashes_json"]),
                "readable_relation_sequence": "PATH_READABLE_MAPPING_NOT_AVAILABLE",
                "status": "PASS_WITH_STABLE_PATH_IDS",
            }
            for row in template_rows
        ]
        write_csv_once(
            result_root / "TC_LPMSO_H8_FINAL_PATH_MAPPING.csv",
            mapping_rows,
            fieldnames=list(mapping_rows[0]),
        )
        path_basis = {
            "record_count": len(template_rows),
            "records_identity": canonical_json_hash(template_rows),
            "acm_set_count": 2,
            "freebase_set_count": 3,
            "cache_identities": FROZEN_CACHE_IDENTITIES,
            "freebase_candidate_bank_sha256": FREEBASE_CANDIDATE_BANK_SHA256,
            "freebase_path_library_sha256": FREEBASE_PATH_LIBRARY_SHA256,
            "optimizer_constructed": False,
            "training_steps": 0,
            "labels_mounted": False,
        }
        write_json_once(
            result_root / "TC_LPMSO_H8_PATH_SET_RESOLUTION_SEAL.json",
            envelope("metachorus.v53.m4a-h8.path-resolution-seal.v1", "PASS", path_basis),
        )
        historical_seeds = (
            set(DEVELOPMENT_MODEL_SEEDS)
            | set(H5_CLUSTER_SEARCH_SEEDS)
            | set(H5_CLASSIFICATION_SPLIT_SEEDS)
            | set(H5_BANK_SEEDS)
        )
        if (
            len(S_EVAL_MODEL_SEEDS) != 10
            or len(set(S_EVAL_MODEL_SEEDS)) != len(S_EVAL_MODEL_SEEDS)
            or set(S_EVAL_MODEL_SEEDS) & historical_seeds
        ):
            raise ValueError("H8 future model seed namespace is not exact, unique, and unused")
        seed_rows = [
            {
                "seed_index": index,
                "model_seed": seed,
                "namespace": "TC_LPMSO_H8_S_EVAL",
                "previously_used": False,
                "collision_with_development": seed in DEVELOPMENT_MODEL_SEEDS,
                "collision_with_historical_seed_namespaces": seed in historical_seeds,
                "status": "PASS",
            }
            for index, seed in enumerate(S_EVAL_MODEL_SEEDS)
        ]
        write_csv_once(
            result_root / "TC_LPMSO_H8_S_EVAL_SEED_USAGE_AUDIT.csv",
            seed_rows,
            fieldnames=list(seed_rows[0]),
        )
        seed_basis = {
            "model_seeds": list(S_EVAL_MODEL_SEEDS),
            "seed_count": len(S_EVAL_MODEL_SEEDS),
            "unique": len(set(S_EVAL_MODEL_SEEDS)) == len(S_EVAL_MODEL_SEEDS),
            "disjoint_from_development": not bool(
                set(S_EVAL_MODEL_SEEDS) & set(DEVELOPMENT_MODEL_SEEDS)
            ),
            "disjoint_from_classification_kmeans_bank_namespaces": not bool(
                set(S_EVAL_MODEL_SEEDS) & historical_seeds
            ),
            "previously_used": False,
            "seed_rows_identity": canonical_json_hash(seed_rows),
        }
        write_json_once(
            result_root / "TC_LPMSO_H8_S_EVAL_SEED_SEAL.json",
            envelope("metachorus.v53.m4a-h8.seed-seal.v1", "SEALED_BEFORE_CANARY", seed_basis),
        )
        role_seal = envelope(
            "metachorus.v53.m4a-h8.locked-role-seal.v1",
            "SEALED_BEFORE_ANY_H8_TRAINING",
            {
                **h5_roles,
                "locked_access_authorized_only_after_registry_seal": True,
                "locked_access_count_before_training": 0,
                "training_labels_visible": False,
            },
        )
        write_json_once(result_root / "TC_LPMSO_H8_LOCKED_ROLE_SEAL.json", role_seal)
        canary_rows, canary_seal = build_canary_plan(template_rows)
        write_csv_once(
            result_root / "M4A_H8_CANARY_RUN_PLAN.csv",
            canary_rows,
            fieldnames=PLAN_FIELDS,
        )
        write_json_once(result_root / "M4A_H8_CANARY_PLAN_SEAL.json", canary_seal)
        science_draft = build_science_draft(template_rows)
        write_csv_once(
            result_root / "TC_LPMSO_H8_RUN_PLAN_DRAFT.csv",
            science_draft,
            fieldnames=PLAN_FIELDS,
        )
        draft_basis = {
            "row_count": len(science_draft),
            "rows_identity": canonical_json_hash(science_draft),
            "acm_run_count": 20,
            "freebase_run_count": 30,
            "frozen_set_count": 5,
            "model_seed_count": 10,
            "continuation_from_update": 0,
            "endpoint_update": 3000,
            "labels_mounted": False,
            "formal_fanout_authorized": False,
            "canary_pass_required": True,
        }
        write_json_once(
            result_root / "TC_LPMSO_H8_RUN_PLAN_DRAFT_SEAL.json",
            envelope("metachorus.v53.m4a-h8.science-draft-seal.v1", "PENDING_CANARY", draft_basis),
        )
        git_identity = envelope(
            "metachorus.v53.m4a-h8.git-config-identity.v1",
            "PASS",
            {
                **source_identity,
                "taskbook_filename": TASKBOOK_FILENAME,
                "taskbook_sha256": TASKBOOK_SHA256,
                "parent_package_sha256": PARENT_PACKAGE_SHA256,
                "template_inventory_identity": canonical_json_hash(template_rows),
                "trainer": "O0-CodeFaithful",
                "view_pipeline": "S0",
                "method_frozen": True,
                "further_retuning": False,
            },
        )
        write_json_once(result_root / "H8_GIT_AND_CONFIG_IDENTITY.json", git_identity)
        write_text_once(result_root / "TC_LPMSO_H8_OPERATIONAL_DEVIATION_LOG.jsonl", "")
        append_jsonl_once(
            result_root / "M4A_H8_EXECUTION_LOG.jsonl",
            {
                "event_id": "H8_PREFLIGHT_COMPLETE",
                "status": "READY_FOR_TWO_H8_ENGINEERING_CANARIES",
                "canary_run_count": EXPECTED_CANARY_RUNS,
                "science_run_count_if_authorized": EXPECTED_SCIENCE_RUNS,
                "labels_mounted": False,
                "h5_locked_report_read": False,
                "formal_training_started": False,
            },
        )
        preflight_basis = {
            "parent_package_audit_identity": parent_audit["identity"],
            "h7_freeze_identity": canonical_json_hash(freeze_basis),
            "path_resolution_identity": path_basis["records_identity"],
            "seed_seal_identity": canonical_json_hash(seed_basis),
            "locked_role_seal_identity": role_seal["identity"],
            "canary_plan_identity": canary_seal["identity"],
            "science_draft_identity": canonical_json_hash(science_draft),
            "canary_run_count": EXPECTED_CANARY_RUNS,
            "science_run_count_if_authorized": EXPECTED_SCIENCE_RUNS,
            "source_commit": source_identity["source_commit"],
            "source_tree": source_identity["source_tree"],
            "labels_mounted": False,
            "h5_locked_report_read": False,
            "formal_training_started": False,
            "fanout_authorized": False,
        }
        preflight = envelope(
            "metachorus.v53.m4a-h8.preflight-complete.v1",
            "READY_FOR_TWO_H8_ENGINEERING_CANARIES",
            preflight_basis,
        )
        write_json_once(result_root / "M4A_H8_PREFLIGHT_COMPLETE.json", preflight)
        write_text_once(
            result_root / "plan.md",
            "# M4A-H8 execution state\n\n"
            "- Parent H7 package and five-set freeze: PASS.\n"
            "- Ten future model seeds and exact 50-run draft: SEALED.\n"
            "- Locked role contract: SEALED; access count is zero.\n"
            "- Next action: run exactly two engineering canaries.\n"
            "- Formal 50-run fan-out is forbidden before H8_CANARY_PASS.\n",
        )
        return preflight
    except Exception:
        if result_root.exists():
            shutil.rmtree(result_root, ignore_errors=True)
        raise
