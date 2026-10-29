"""H6 parent audit, incident closure, pool seal, path resolution, and canary plan."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, Mapping
import zipfile

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file, utc_now

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    safe_relative,
    seal_rows,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    EXPECTED_POOL_COUNTS,
    EXPECTED_POOL_TOTAL,
    FROZEN_CACHE_IDENTITIES,
    H5_R3_CHECKPOINT_REGISTRY_SHA256,
    H5_R3_RUN_REGISTRY_SHA256,
    H5_R3_SEARCH_METRICS_SHA256,
    PARENT_COMMON_ROOT_CAUSE_COUNT,
    PARENT_FAILURE_CODE,
    PARENT_HISTORICAL_FAILED_ATTEMPTS,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_RECOVERED_ATTEMPTS,
    PARENT_RESOURCE_STATUS,
    PARENT_SCIENTIFIC_STATUS,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    RESOURCE_LIMITS,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
)
from .protocol import build_canary_identities, derive_uncertainty_pool


POOL_FIELDS = (
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "roles_json",
    "is_manual",
    "is_current_automatic",
    "truth_fields_read",
    "h5_locked_report_access",
    "s_eval_access",
)
COMPONENT_FIELDS = (
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "component",
    "component_rank",
    "component_score",
    "model_seed",
    "source_field",
    "truth_fields_read",
    "h5_locked_report_access",
    "s_eval_access",
)
PATH_AUDIT_FIELDS = (
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "path_ids_json",
    "projection_hashes_json",
    "projection_files_json",
    "projection_file_sha256_json",
    "projection_identity",
    "candidate_bank_sha256",
    "path_library_sha256",
    "cache_manifest_sha256",
    "cache_identity",
    "optimizer_constructed",
    "training_steps",
    "labels_mounted",
    "status",
)
CANARY_PLAN_FIELDS = (
    "stage",
    "physical_run_id",
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "model_seed",
    "parent_source",
    "parent_physical_run_id",
    "parent_attempt_relative_path",
    "path_ids_json",
    "continuation_from_update",
    "endpoint_update",
    "continuation_mode",
    "execution_action",
    "run_role",
    "eligible_for_scientific_result",
    "eligible_for_selector_development",
    "labels_mounted",
    "downstream_evaluations",
    "reusable_for_science",
    "gpu_preference",
    "attempt_index",
    "status",
    "plan_seal_sha256",
)


@dataclass(frozen=True)
class H6PreflightInputs:
    repo_root: Path
    taskbook: Path
    parent_package: Path
    h5_result_root: Path
    cache_root: Path
    private_label_mount_root: Path
    result_root: Path


class H6PreflightResultRootNotFresh(FileExistsError):
    """The caller selected an existing result root; it must remain untouched."""


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def source_identity(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise RuntimeError("H6 preflight requires a clean source worktree")
    basis = {
        "source_commit": _git(repo_root, "rev-parse", "HEAD"),
        "source_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "worktree_clean": True,
        "parent_h5_source_commit": PARENT_SOURCE_COMMIT,
        "parent_h5_source_tree": PARENT_SOURCE_TREE,
        "protocol_only_successor": True,
    }
    return envelope("metachorus.v53.m4a-h6.source-identity.v1", "PASS", basis)


def _safe_zip_name(value: str) -> bool:
    if "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and bool(path.parts) and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def audit_parent_package(path: Path) -> dict[str, Any]:
    if path.name != PARENT_PACKAGE_FILENAME:
        raise ValueError("H6 parent package filename drift")
    if sha256_file(path) != PARENT_PACKAGE_SHA256 or path.stat().st_size != PARENT_PACKAGE_SIZE_BYTES:
        raise ValueError("H6 parent package hash/size mismatch")
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if len(names) != len(set(names)) or any(not _safe_zip_name(name) for name in names):
            raise ValueError("H6 parent package has unsafe or duplicate members")
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise ValueError(f"H6 parent package CRC failure: {bad_crc}")
        manifest = json.loads(archive.read("PACKAGE_MEMBER_MANIFEST.json").decode("utf-8"))
        members = manifest.get("members")
        if not isinstance(members, list):
            raise ValueError("parent package member manifest is malformed")
        expected = {str(row["relative_path"]): row for row in members}
        if len(expected) != len(members) or int(manifest["member_count"]) != len(members):
            raise ValueError("parent package member count mismatch")
        business_names = set(names) - {"PACKAGE_MEMBER_MANIFEST.json", "PACKAGE_CHECKSUMS.sha256"}
        if business_names != set(expected):
            raise ValueError("parent package manifest/member set mismatch")
        for name, row in expected.items():
            payload = archive.read(name)
            import hashlib

            if len(payload) != int(row["size_bytes"]) or hashlib.sha256(payload).hexdigest() != str(row["sha256"]):
                raise ValueError(f"parent package member mismatch: {name}")
        if canonical_json_hash(members) != str(manifest["manifest_identity"]):
            raise ValueError("parent package manifest identity mismatch")
        source = json.loads(
            archive.read("evidence/results/SOURCE_DEPENDENCY_MANIFEST_M4A_H5.json").decode("utf-8")
        )
        source_basis = source.get("basis", {})
        if (
            source.get("status") != "PASS"
            or source_basis.get("source_commit") != PARENT_SOURCE_COMMIT
            or source_basis.get("source_tree") != PARENT_SOURCE_TREE
            or source.get("identity") != canonical_json_hash(source_basis)
        ):
            raise ValueError("parent package source identity mismatch")
    basis = {
        "package_name": path.name,
        "package_sha256": PARENT_PACKAGE_SHA256,
        "package_size_bytes": PARENT_PACKAGE_SIZE_BYTES,
        "member_count": len(members),
        "manifest_identity": manifest["manifest_identity"],
        "crc_pass": True,
        "path_traversal_absent": True,
        "member_hashes_pass": True,
        "parent_source_commit": PARENT_SOURCE_COMMIT,
        "parent_source_tree": PARENT_SOURCE_TREE,
        "parent_scientific_status": PARENT_SCIENTIFIC_STATUS,
        "parent_resource_status": PARENT_RESOURCE_STATUS,
        "parent_result_immutable": True,
    }
    return envelope("metachorus.v53.m4a-h6.parent-package-audit.v1", "PASS", basis)


def parent_tree_snapshot(root: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"parent H5 result contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        members.append({"relative_path": relative, "size_bytes": size, "sha256": sha256_file(path)})
        total_bytes += size
    basis = {
        "parent_result_member_count": len(members),
        "parent_result_total_bytes": total_bytes,
        "parent_result_tree_identity": canonical_json_hash(members),
        "old_h5_result_modified": False,
    }
    return envelope("metachorus.v53.m4a-h6.parent-immutability-snapshot.v1", "PASS", basis)


def _verify_h5_parent_inputs(root: Path) -> dict[str, Any]:
    search = root / "TC_LPMSO_H5_R3_SEARCH_METRICS.csv"
    run_registry = root / "TC_LPMSO_H5_R3_RUN_REGISTRY.csv"
    checkpoint_registry = root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv"
    if sha256_file(search) != H5_R3_SEARCH_METRICS_SHA256:
        raise ValueError("H5 R3 search metrics hash mismatch")
    if sha256_file(run_registry) != H5_R3_RUN_REGISTRY_SHA256:
        raise ValueError("H5 R3 run registry hash mismatch")
    if sha256_file(checkpoint_registry) != H5_R3_CHECKPOINT_REGISTRY_SHA256:
        raise ValueError("H5 R3 checkpoint registry hash mismatch")
    failure = read_json(root / "M4A_H5_FAILURE.json")
    resource = read_json(root / "TC_LPMSO_H5_RESOURCE_AUDIT.json")
    complete = read_json(root / "M4A_H5_COMPLETE.json")
    failure_basis = failure.get("basis", {})
    resource_basis = resource.get("basis", {})
    complete_basis = complete.get("basis", {})
    counts = failure_basis.get("counts", {})
    if (
        failure.get("schema_version") != "metachorus.v53.m4a-h5.failure.v1"
        or resource.get("schema_version") != "metachorus.v53.m4a-h5.resource-audit.v1"
        or complete.get("schema_version") != "metachorus.v53.m4a-h5.complete.v1"
        or failure.get("identity") != canonical_json_hash(failure_basis)
        or resource.get("identity") != canonical_json_hash(resource_basis)
        or complete.get("identity") != canonical_json_hash(complete_basis)
        or failure.get("status") != PARENT_SCIENTIFIC_STATUS
        or failure_basis.get("failure_code") != PARENT_FAILURE_CODE
        or resource.get("status") != PARENT_RESOURCE_STATUS
        or complete.get("status") != PARENT_RESOURCE_STATUS
        or int(counts.get("r3_identity_count", -1)) != 264
        or counts.get("third_seed_started") is not False
        or failure_basis.get("downstream_training_started") is not False
        or failure_basis.get("s_eval_started") is not False
        or failure_basis.get("h5_locked_report_access") is not False
        or int(resource_basis.get("failed_attempt_count", -1))
        != PARENT_HISTORICAL_FAILED_ATTEMPTS
        or resource_basis.get("hard_limits_pass") is not False
        or resource_basis.get("stage_reachability", {}).get("THIRD_SEED") != "NOT_REACHED"
        or resource_basis.get("stage_reachability", {}).get("R4") != "NOT_REACHED"
        or complete_basis.get("execution_complete") is not False
        or complete_basis.get("execution_status") != "PARTIAL"
        or complete_basis.get("final_scientific_status") != PARENT_SCIENTIFIC_STATUS
        or complete_basis.get("terminal_status") != PARENT_RESOURCE_STATUS
        or complete_basis.get("third_seed_started") is not False
        or complete_basis.get("r4_started") is not False
        or complete_basis.get("s_eval_started") is not False
        or complete_basis.get("h5_locked_report_access") is not False
    ):
        raise ValueError("H5 terminal failure/resource boundary mismatch")
    return {
        "search_metrics_sha256": H5_R3_SEARCH_METRICS_SHA256,
        "run_registry_sha256": H5_R3_RUN_REGISTRY_SHA256,
        "checkpoint_registry_sha256": H5_R3_CHECKPOINT_REGISTRY_SHA256,
        "failure_identity": failure["identity"],
        "resource_audit_identity": resource["identity"],
        "complete_identity": complete["identity"],
    }


def incident_disposition(root: Path) -> dict[str, Any]:
    failures = read_json(root / "M4A_H5_PRE_HOTFIX_FAILURE_MANIFEST.json")
    outcomes = read_json(root / "M4A_H5_HOTFIX_RETRY_OUTCOME_MANIFEST.json")
    failure_records = failures.get("basis", {}).get("records", [])
    outcome_records = outcomes.get("basis", {}).get("records", [])
    if len(failure_records) != PARENT_HISTORICAL_FAILED_ATTEMPTS:
        raise ValueError("H5 preserved failure count is not 120")
    if len(outcome_records) != PARENT_RECOVERED_ATTEMPTS:
        raise ValueError("H5 recovered retry count is not 120")
    root_causes: set[str] = set()
    failed_ids: set[str] = set()
    for record in failure_records:
        receipt = read_json(root / str(record["failure_receipt_relative_path"]))
        basis = receipt.get("basis", {})
        root_causes.add(
            canonical_json_hash(
                {
                    "failure_kind": basis.get("failure_kind"),
                    "error_message": basis.get("error_message"),
                }
            )
        )
        failed_ids.add(str(record["physical_run_id"]))
        if (
            int(record["optimizer_steps_after_boundary"]) != 0
            or record["scientific_metrics_observed"] is not False
        ):
            raise ValueError("H5 pre-hotfix failure was not pre-metric/pre-step")
    recovered_ids = {str(record["physical_run_id"]) for record in outcome_records}
    if len(root_causes) != PARENT_COMMON_ROOT_CAUSE_COUNT or recovered_ids != failed_ids:
        raise ValueError("H5 common-root/recovery identity closure mismatch")
    basis = {
        "historical_h5_failed_attempt_count": len(failure_records),
        "historical_h5_recovered_attempt_count": len(outcome_records),
        "historical_h5_unrecovered_attempt_count": 0,
        "historical_h5_unique_common_root_cause_count": len(root_causes),
        "root_cause_identities": sorted(root_causes),
        "optimizer_steps_after_boundary": 0,
        "scientific_metrics_observed_before_recovery": False,
        "completed_runs_preserved": True,
        "h5_failure_receipts_preserved": True,
        "h5_counters_excluded_from_h6_new_failure_limit": True,
        "cumulative_failed_attempts_including_h5": len(failure_records),
        "h6_new_failed_attempts": 0,
        "h6_unique_root_causes": 0,
        "h6_unrecovered_scientific_identities": 0,
        "disposition": "CLOSED_AS_HISTORICAL_COMMON_ROOT_ENGINEERING_INCIDENT",
    }
    return envelope("metachorus.v53.m4a-h6.common-root-cause-incident-disposition.v1", "PASS", basis)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _manifest_file_hash(row: Mapping[str, Any]) -> str | None:
    observed: list[str] = []
    for key in ("sha256", "file_sha256", "raw_file_sha256"):
        if key not in row:
            continue
        value = row[key]
        if not _is_sha256(value):
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: invalid manifest file hash")
        observed.append(str(value).lower())
    if len(set(observed)) > 1:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: conflicting manifest file hashes")
    return observed[0] if observed else None


def _manifest_file_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        manifest.get("local_absolute_paths_present") is not False
        or manifest.get("uses_evaluation_information") is not False
    ):
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: unsafe cache manifest flags")
    files = manifest.get("files")
    rows: list[tuple[object, object]]
    if isinstance(files, list):
        rows = []
        for value in files:
            if not isinstance(value, Mapping):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: malformed cache manifest row")
            rows.append((value.get("relative_path"), value))
    elif isinstance(files, Mapping):
        rows = list(files.items())
    else:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: unsupported cache manifest files schema")

    output: dict[str, dict[str, Any]] = {}
    for raw_relative, metadata in rows:
        if not isinstance(raw_relative, str) or not isinstance(metadata, Mapping):
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: malformed cache manifest member")
        try:
            relative = safe_relative(raw_relative)
        except ValueError as error:
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: unsafe cache manifest path") from error
        canonical = relative.as_posix()
        if raw_relative != canonical:
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: non-canonical cache manifest path")
        declared_relative = metadata.get("relative_path")
        if declared_relative is not None and str(declared_relative) != canonical:
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache manifest path/key mismatch")
        if canonical in output:
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: duplicate cache manifest path")
        row = dict(metadata)
        row["relative_path"] = canonical
        expected_hash = _manifest_file_hash(row)
        size = row.get("size_bytes")
        if expected_hash is None or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: incomplete cache manifest member")
        output[canonical] = row
    if not output:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: empty cache manifest")
    return output


def _resolve_cache_member(
    root: Path,
    relative_value: str,
    file_index: Mapping[str, Mapping[str, Any]],
    *,
    require_manifest: bool,
) -> tuple[Path, str]:
    try:
        relative = safe_relative(relative_value)
    except ValueError as error:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: unsafe cache member path") from error
    if relative_value != relative.as_posix():
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: non-canonical cache member path")
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: missing dataset cache root") from error
    if root.is_symlink():
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: dataset cache root is a symlink")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache member symlink forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache member escapes or is missing") from error
    if not resolved.is_file():
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache member is not a file")
    observed_hash = sha256_file(resolved)
    manifest_row = file_index.get(relative.as_posix())
    if manifest_row is None:
        if require_manifest:
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache member missing from manifest")
        return resolved, observed_hash
    expected_hash = _manifest_file_hash(manifest_row)
    if expected_hash is None:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache member hash missing")
    if int(manifest_row["size_bytes"]) != resolved.stat().st_size or expected_hash != observed_hash:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache member size/hash mismatch")
    return resolved, observed_hash


def resolve_path_sets(pool_rows: list[dict[str, Any]], cache_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        root = cache_root / dataset
        bank_relative = (
            "candidates/fixed_cardinality_sets.json" if dataset == "acm" else "candidates/all_sets.json"
        )
        library_relative = "candidates/path_library.json"
        manifest_path = root / "SANITIZED_CACHE_MANIFEST.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache manifest missing or symlinked")
        manifest = read_json(manifest_path)
        expected_schema = (
            "metachorus.sanitized-cache.v1"
            if dataset == "acm"
            else "metachorus.freebase-sanitized-cache.v1"
        )
        cache_identity = manifest.get("cache_identity")
        identity_basis_key = "cache_identity_basis" if dataset == "acm" else "identity_basis"
        cache_identity_basis = manifest.get(identity_basis_key)
        if (
            manifest.get("schema_version") != expected_schema
            or not _is_sha256(cache_identity)
            or cache_identity != FROZEN_CACHE_IDENTITIES[dataset]
            or not isinstance(cache_identity_basis, Mapping)
            or canonical_json_hash(cache_identity_basis) != cache_identity
        ):
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: cache manifest identity mismatch")
        file_index = _manifest_file_index(manifest)
        if dataset == "acm":
            file_count = manifest.get("file_count")
            basis_file_index = _manifest_file_index(
                {
                    "local_absolute_paths_present": False,
                    "uses_evaluation_information": False,
                    "files": cache_identity_basis.get("files"),
                }
            )
            if (
                basis_file_index != file_index
                or isinstance(file_count, bool)
                or not isinstance(file_count, int)
                or file_count != len(file_index)
                or manifest.get("deterministic_cache_id")
                != cache_identity_basis.get("deterministic_cache_id")
            ):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: ACM cache identity basis mismatch")
        else:
            arrays = cache_identity_basis.get("arrays")
            if not isinstance(arrays, Mapping) or set(arrays) != set(file_index) or any(
                not _is_sha256(semantic_hash)
                or file_index[relative].get("semantic_sha256") != semantic_hash
                for relative, semantic_hash in arrays.items()
            ):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: Freebase cache identity basis mismatch")
        bank_path, bank_file_hash = _resolve_cache_member(
            root,
            bank_relative,
            file_index,
            require_manifest=dataset == "acm",
        )
        library_path, library_file_hash = _resolve_cache_member(
            root,
            library_relative,
            file_index,
            require_manifest=dataset == "acm",
        )
        bank = read_json(bank_path)
        library = read_json(library_path)
        library_rows = library.get("paths")
        bank_rows = bank.get("sets")
        if not isinstance(library_rows, list) or not isinstance(bank_rows, list):
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: malformed candidate bank or path library")
        paths: dict[str, Mapping[str, Any]] = {}
        for row in library_rows:
            if not isinstance(row, Mapping) or not str(row.get("path_id", "")):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: malformed path library row")
            path_id = str(row["path_id"])
            if path_id in paths:
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: duplicate path identity")
            paths[path_id] = row
        projection_identity = library.get("projection_identity")
        if not _is_sha256(projection_identity):
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: invalid projection identity")
        if dataset == "freebase":
            identity_basis = manifest.get("identity_basis")
            if not isinstance(identity_basis, Mapping) or (
                identity_basis.get("all_sets_sha256") != bank.get("all_sets_sha256")
                or identity_basis.get("path_library_sha256") != bank.get("path_library_sha256")
                or identity_basis.get("path_library_sha256") != library.get("source_path_library_sha256")
                or identity_basis.get("projection_identity") != projection_identity
                or bank.get("projection_identity") != projection_identity
            ):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: Freebase bank/library manifest mismatch")
        elif (
            bank.get("candidate_space_identity") != library.get("candidate_space_identity")
            or bank.get("dataset_id") != "acm"
            or library.get("dataset_id") != "acm"
        ):
            raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: ACM bank/library identity mismatch")
        sets: dict[str, Mapping[str, Any]] = {}
        for row in bank_rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: malformed candidate set row")
            candidate_hash = str(row.get("candidate_set_hash", row.get("set_sha256")))
            if not _is_sha256(candidate_hash) or candidate_hash in sets:
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: invalid or duplicate candidate hash")
            sets[candidate_hash] = row
        for pool in [row for row in pool_rows if str(row["dataset"]) == dataset]:
            candidate_hash = str(pool["candidate_set_hash"])
            source = sets.get(candidate_hash)
            if source is None:
                raise RuntimeError(f"BLOCKED_PATH_SET_RESOLUTION: {dataset}/{candidate_hash}")
            source_id = str(source.get("set_id", source.get("path_set_id")))
            source_ordinal = int(source.get("set_ordinal", source.get("ordinal", -1)))
            if source_id != str(pool["candidate_set_id"]) or source_ordinal != int(pool["candidate_set_ordinal"]):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: candidate identity mismatch")
            raw_path_ids = source.get("path_ids")
            if not isinstance(raw_path_ids, list) or not raw_path_ids:
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: missing candidate path ids")
            path_ids = [str(value) for value in raw_path_ids]
            if len(path_ids) != len(set(path_ids)):
                raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: duplicate candidate path ids")
            projection_hashes: list[str] = []
            projection_files: list[str] = []
            projection_file_hashes: list[str] = []
            for path_id in path_ids:
                path_row = paths.get(path_id)
                if path_row is None:
                    raise RuntimeError(f"BLOCKED_PATH_SET_RESOLUTION: missing {path_id}")
                relative_value = path_row.get("array")
                if not isinstance(relative_value, str):
                    raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: projection path missing")
                try:
                    relative_path = safe_relative(relative_value)
                except ValueError as error:
                    raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: unsafe projection path") from error
                relative = relative_path.as_posix()
                if relative_value != relative:
                    raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: non-canonical projection path")
                _, file_hash = _resolve_cache_member(
                    root,
                    relative,
                    file_index,
                    require_manifest=True,
                )
                semantic_hash = path_row.get("semantic_projection_hash", path_row.get("adjacency_sha256"))
                if not _is_sha256(semantic_hash):
                    raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: projection semantic hash invalid")
                projection_hashes.append(str(semantic_hash).lower())
                projection_files.append(relative)
                projection_file_hashes.append(file_hash)
            output.append(
                {
                    "dataset": dataset,
                    "candidate_set_id": source_id,
                    "candidate_set_hash": candidate_hash,
                    "candidate_set_ordinal": source_ordinal,
                    "path_ids_json": json.dumps(path_ids, ensure_ascii=False),
                    "projection_hashes_json": json.dumps(projection_hashes),
                    "projection_files_json": json.dumps(projection_files, ensure_ascii=False),
                    "projection_file_sha256_json": json.dumps(projection_file_hashes),
                    "projection_identity": str(projection_identity).lower(),
                    "candidate_bank_sha256": bank_file_hash,
                    "path_library_sha256": library_file_hash,
                    "cache_manifest_sha256": sha256_file(manifest_path),
                    "cache_identity": str(cache_identity).lower(),
                    "optimizer_constructed": False,
                    "training_steps": 0,
                    "labels_mounted": False,
                    "status": "PASS",
                }
            )
    if len(output) != EXPECTED_POOL_TOTAL:
        raise RuntimeError("BLOCKED_PATH_SET_RESOLUTION: row count")
    return sorted(output, key=lambda row: (str(row["dataset"]), int(row["candidate_set_ordinal"])))


def record_preflight_failure(result_root: Path, error: BaseException) -> dict[str, Any]:
    raw_root = Path(result_root)
    if raw_root.is_symlink():
        raise H6PreflightResultRootNotFresh("H6 result root must not be a symlink")
    root = raw_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "failed_attempts" / "preflight_attempt_00" / "PREFLIGHT_ATTEMPT_FAILED.json"
    if receipt_path.exists():
        observed = read_json(receipt_path)
        basis = observed.get("basis", {})
        if (
            observed.get("schema_version")
            != "metachorus.v53.m4a-h6.preflight-failed-attempt.v1"
            or observed.get("status") != "FAILED_PRESERVED"
            or observed.get("identity") != canonical_json_hash(basis)
        ):
            raise RuntimeError("existing H6 preflight failure receipt is invalid")
        return observed

    existing: list[dict[str, Any]] = []
    for path in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise RuntimeError("cannot preserve H6 preflight failure with symlinked artifacts")
        relative = path.relative_to(root).as_posix()
        existing.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    message = str(error)
    failure_code = (
        "BLOCKED_PATH_SET_RESOLUTION"
        if "BLOCKED_PATH_SET_RESOLUTION" in message
        else "H6_PREFLIGHT_FAILED"
    )
    basis = {
        "attempt_index": 0,
        "recorded_at_utc": utc_now(),
        "failed_operation": "BUILD_H6_PREFLIGHT",
        "failure_code": failure_code,
        "failure_kind": type(error).__name__,
        "error_message_sha256": canonical_json_hash({"error_message": message}),
        "preserved_artifact_count": len(existing),
        "preserved_artifacts": existing,
        "scientific_training_started": False,
        "optimizer_constructed": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
        "result_root_deleted_or_overwritten": False,
    }
    payload = envelope(
        "metachorus.v53.m4a-h6.preflight-failed-attempt.v1",
        "FAILED_PRESERVED",
        basis,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(receipt_path, payload)
    return payload


def _verify_private_mount(h5_result_root: Path, private_root: Path) -> dict[str, Any]:
    audit_path = h5_result_root / "TC_LPMSO_H5_PRIVATE_SEARCH_MOUNT_AUDIT.json"
    audit = read_json(audit_path)
    members = audit.get("members", audit.get("basis", {}).get("members", []))
    if audit.get("status") != "PASS" or len(members) != 2:
        raise ValueError("H5 private search mount audit is not sealed")
    observed = []
    for row in members:
        member_name = str(row.get("member_name", row.get("relative_path")))
        path = private_root / member_name
        if not path.is_file() or sha256_file(path) != str(row["sha256"]):
            raise ValueError("H5 private search mount member mismatch")
        observed.append(member_name)
    if sorted(path.name for path in private_root.iterdir() if path.is_file()) != sorted(observed):
        raise ValueError("private search mount contains an undeclared member")
    return {
        "h5_private_mount_audit_sha256": sha256_file(audit_path),
        "private_member_count": 2,
        "contains_h5_locked_report": False,
        "contains_full_labels": False,
        "future_s_eval_seeds_used": False,
    }


def build_preflight(inputs: H6PreflightInputs) -> dict[str, Any]:
    repo_root = inputs.repo_root.resolve()
    raw_result_root = inputs.result_root
    if raw_result_root.is_symlink():
        raise H6PreflightResultRootNotFresh("H6 result root must not be a symlink")
    result_root = raw_result_root.resolve()
    h5_result_root = inputs.h5_result_root.resolve()
    if result_root.exists() and any(result_root.iterdir()):
        raise H6PreflightResultRootNotFresh("H6 result root must be fresh and empty")
    result_root.mkdir(parents=True, exist_ok=True)
    if inputs.taskbook.name != TASKBOOK_FILENAME or sha256_file(inputs.taskbook) != TASKBOOK_SHA256:
        raise ValueError("H6 taskbook identity mismatch")
    source = source_identity(repo_root)
    package_audit = audit_parent_package(inputs.parent_package)
    parent_inputs = _verify_h5_parent_inputs(h5_result_root)
    mount = _verify_private_mount(h5_result_root, inputs.private_label_mount_root.resolve())
    incident = incident_disposition(h5_result_root)

    search_rows = read_csv(h5_result_root / "TC_LPMSO_H5_R3_SEARCH_METRICS.csv")
    components, pool = derive_uncertainty_pool(search_rows)
    path_rows = resolve_path_sets(pool, inputs.cache_root.resolve())
    before = parent_tree_snapshot(h5_result_root)
    component_path = write_csv_once(
        result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL_COMPONENTS.csv",
        components,
        fieldnames=COMPONENT_FIELDS,
    )
    pool_path = write_csv_once(
        result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL.csv",
        pool,
        fieldnames=POOL_FIELDS,
    )
    pool_basis = {
        "source_artifact": "TC_LPMSO_H5_R3_SEARCH_METRICS.csv",
        "source_sha256": H5_R3_SEARCH_METRICS_SHA256,
        "components_sha256": sha256_file(component_path),
        "pool_sha256": sha256_file(pool_path),
        "pool_identity": canonical_json_hash(pool),
        "acm_count": EXPECTED_POOL_COUNTS["acm"],
        "freebase_count": EXPECTED_POOL_COUNTS["freebase"],
        "total_count": EXPECTED_POOL_TOTAL,
        "truth_artifacts_opened_before_seal": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
        "third_seed_started": False,
    }
    pool_seal = envelope(
        "metachorus.v53.m4a-h6.r3-uncertainty-pool-seal.v1",
        "SEALED_BEFORE_THIRD_SEED_TRAINING",
        pool_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL_SEAL.json", pool_seal)

    path_audit = write_csv_once(
        result_root / "M4A_H6_PATH_SET_RESOLUTION_AUDIT.csv",
        path_rows,
        fieldnames=PATH_AUDIT_FIELDS,
    )
    failure_manifest = read_json(h5_result_root / "M4A_H5_PRE_HOTFIX_FAILURE_MANIFEST.json")
    canaries = build_canary_identities(pool, failure_manifest["basis"]["records"])
    path_by_hash = {str(row["candidate_set_hash"]): row for row in path_rows}
    h5_registry = read_csv(h5_result_root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv")
    h5_template_by_hash = {
        str(row["candidate_set_hash"]): row
        for row in h5_registry
        if int(row["model_seed"]) == 1985123271
    }
    for row in canaries:
        candidate_hash = str(row["candidate_set_hash"])
        template = h5_template_by_hash[candidate_hash]
        checkpoint_relative = PurePosixPath(str(template["checkpoint_relative_path"]))
        row["parent_source"] = "H5_R3_TEMPLATE_CONFIG_ONLY"
        row["parent_physical_run_id"] = str(template["physical_run_id"])
        row["parent_attempt_relative_path"] = checkpoint_relative.parent.parent.as_posix()
        row["path_ids_json"] = str(path_by_hash[candidate_hash]["path_ids_json"])
    canary_rows, canary_seal = seal_rows(canaries, stage="CANARY")
    canary_path = write_csv_once(
        result_root / "M4A_H6_CANARY_RUN_PLAN.csv",
        canary_rows,
        fieldnames=CANARY_PLAN_FIELDS,
    )
    write_json_once(result_root / "M4A_H6_CANARY_PLAN_SEAL.json", canary_seal)

    write_json_once(result_root / "M4A_H6_PARENT_PACKAGE_AUDIT.json", package_audit)
    parent_source_basis = {
        "parent_package_audit_identity": package_audit["identity"],
        "parent_source_commit": PARENT_SOURCE_COMMIT,
        "parent_source_tree": PARENT_SOURCE_TREE,
        "parent_scientific_status": PARENT_SCIENTIFIC_STATUS,
        "parent_resource_status": PARENT_RESOURCE_STATUS,
        "old_h5_result_immutable": True,
        **parent_inputs,
        **mount,
    }
    write_json_once(
        result_root / "M4A_H6_PARENT_SOURCE_IDENTITY.json",
        envelope("metachorus.v53.m4a-h6.parent-source-identity.v1", "PASS", parent_source_basis),
    )
    write_json_once(result_root / "M4A_H6_PARENT_IMMUTABILITY_BEFORE.json", before)
    write_json_once(result_root / "M4A_H6_COMMON_ROOT_CAUSE_INCIDENT_DISPOSITION.json", incident)
    supersession_basis = {
        "old_h5_scientific_status": PARENT_SCIENTIFIC_STATUS,
        "old_h5_resource_status": PARENT_RESOURCE_STATUS,
        "old_h5_failure_preserved": True,
        "old_h5_result_modified": False,
        "old_h5_thresholds_reinterpreted": False,
        "old_h5_120_attempts_preserved": True,
        "h6_protocol_active": True,
        "protocol_only_supersession": True,
        "h6_source_identity": source["identity"],
        "h6_taskbook_sha256": TASKBOOK_SHA256,
    }
    supersession = envelope(
        "metachorus.v53.m4a-h6.h5-status-supersession.v1",
        "H5_FAIL_AND_RESOURCE_PAUSE_PRESERVED_H6_PROTOCOL_ACTIVE",
        supersession_basis,
    )
    write_json_once(result_root / "M4A_H5_STATUS_SUPERSESSION_H6.json", supersession)
    write_text_once(
        result_root / "M4A_H5_STATUS_SUPERSESSION_H6.md",
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n',
    )
    free_bytes = shutil.disk_usage(result_root).free
    resource_basis = {
        "historical_h5_failed_attempts": PARENT_HISTORICAL_FAILED_ATTEMPTS,
        "h6_new_failed_attempts": 0,
        "h6_unique_root_causes": 0,
        "h6_unrecovered_scientific_identities": 0,
        "h6_limits": dict(RESOURCE_LIMITS),
        "result_volume_free_bytes_at_preflight": free_bytes,
        "cpu_logical_count": os.cpu_count(),
        "maximum_gpu_count": 2,
        "canary_run_count": 2,
        "third_seed_run_count_if_authorized": EXPECTED_POOL_TOTAL,
        "fanout_authorized": False,
        "authorization_scope": "PREFLIGHT_AND_TWO_ENGINEERING_CANARIES_ONLY",
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    resource = envelope(
        "metachorus.v53.m4a-h6.new-stage-resource-authorization.v1",
        "AUTHORIZED_FOR_CANARY_ONLY",
        resource_basis,
    )
    write_json_once(result_root / "M4A_H6_NEW_STAGE_RESOURCE_AUTHORIZATION.json", resource)

    evidence_taskbook = result_root / "evidence" / "taskbook" / TASKBOOK_FILENAME
    evidence_taskbook.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs.taskbook, evidence_taskbook)
    if sha256_file(evidence_taskbook) != TASKBOOK_SHA256:
        raise RuntimeError("H6 taskbook snapshot copy mismatch")
    plan_text = (
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
    )
    write_text_once(result_root / "plan.md", plan_text)
    append_jsonl_once(
        result_root / "M4A_H6_EXECUTION_LOG.jsonl",
        {
            "event_id": "H6_PREFLIGHT_AND_POOL_SEAL",
            "created_at_utc": utc_now(),
            "status": "READY_FOR_TWO_ENGINEERING_CANARIES",
            "source_identity": source["identity"],
            "parent_package_audit_identity": package_audit["identity"],
            "pool_seal_identity": pool_seal["identity"],
            "path_resolution_sha256": sha256_file(path_audit),
            "canary_plan_sha256": sha256_file(canary_path),
            "labels_mounted": False,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    write_text_once(result_root / "TC_LPMSO_H6_OPERATIONAL_DEVIATION_LOG.jsonl", "")
    complete_basis = {
        "source_identity": source["identity"],
        "parent_package_audit_identity": package_audit["identity"],
        "parent_immutability_before_identity": before["identity"],
        "incident_disposition_identity": incident["identity"],
        "pool_seal_identity": pool_seal["identity"],
        "path_resolution_row_count": len(path_rows),
        "canary_plan_row_count": len(canary_rows),
        "truth_opened_before_pool_seal": False,
        "fanout_started": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    payload = envelope(
        "metachorus.v53.m4a-h6.preflight-complete.v1",
        "READY_FOR_TWO_ENGINEERING_CANARIES",
        complete_basis,
    )
    write_json_once(result_root / "M4A_H6_PREFLIGHT_COMPLETE.json", payload)
    return payload
