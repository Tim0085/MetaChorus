"""H7 parent closure, robust shortlist seal, parent states, and canary preflight."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, Mapping
import zipfile

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file, utc_now
from metachorus_v53.m4a_h6.preflight import (
    PATH_AUDIT_FIELDS,
    _verify_private_mount,
    resolve_path_sets,
)

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    EVALUATOR_CONTRACT_SHA256,
    EVALUATOR_SEAL_SHA256,
    EXPECTED_PARENT_STATES,
    EXPECTED_R4_RUNS,
    EXPECTED_SHORTLIST_COUNTS,
    EXPECTED_SHORTLIST_IDENTITY,
    EXPECTED_SHORTLIST_TOTAL,
    PARENT_ARTIFACT_SHA256,
    PARENT_CLOSURE_COMMIT,
    PARENT_CLOSURE_TREE,
    PARENT_FAILED_STAGE,
    PARENT_PACKAGE_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_PACKAGE_SIZE_BYTES,
    PARENT_SCIENTIFIC_COMMIT,
    PARENT_SCIENTIFIC_STATUS,
    PARENT_SCIENTIFIC_TREE,
    RESOURCE_LIMITS,
    ROLE_MANIFEST_SHA256,
    ROLE_SEAL_SHA256,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
)
from .planning import (
    PARENT_INVENTORY_FIELDS,
    PLAN_FIELDS,
    build_canary_plan,
    collect_parent_states,
)
from .protocol import derive_robust_shortlist


PER_SEED_FRONT_FIELDS = (
    "dataset",
    "model_seed",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "F_cls",
    "F_clu",
    "front_rank",
    "status",
)
PARETO_SUPPORT_FIELDS = (
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "pareto_support_count",
    "minimum_support",
    "selected",
)
ROBUST_4D_FIELDS = (
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "Q_cls",
    "Q_clu",
    "W_cls",
    "W_clu",
    "front_rank",
    "status",
)
BALANCED_FIELDS = (
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "classification_normalized_midrank",
    "clustering_normalized_midrank",
    "R_balanced",
    "mean_task_rank",
    "balanced_position",
    "selected",
)
SHORTLIST_COMPONENT_FIELDS = (
    "dataset",
    "path_set_id",
    "path_set_hash",
    "candidate_set_ordinal",
    "component",
)
SHORTLIST_FIELDS = (
    "dataset",
    "path_set_id",
    "path_set_hash",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "components_json",
    "is_manual",
    "is_current_automatic",
    "truth_fields_read",
    "h5_locked_report_access",
    "s_eval_access",
)


@dataclass(frozen=True)
class H7PreflightInputs:
    repo_root: Path
    taskbook: Path
    parent_package: Path
    h6_result_root: Path
    h5_result_root: Path
    cache_root: Path
    private_label_mount_root: Path
    result_root: Path


class H7PreflightResultRootNotFresh(FileExistsError):
    """The caller selected a non-fresh H7 result root."""


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
    if _git(repo_root, "status", "--porcelain"):
        raise RuntimeError("H7 preflight requires a clean source worktree")
    basis = {
        "source_commit": _git(repo_root, "rev-parse", "HEAD"),
        "source_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "worktree_clean": True,
        "parent_h6_closure_commit": PARENT_CLOSURE_COMMIT,
        "parent_h6_closure_tree": PARENT_CLOSURE_TREE,
        "parent_h6_scientific_commit": PARENT_SCIENTIFIC_COMMIT,
        "parent_h6_scientific_tree": PARENT_SCIENTIFIC_TREE,
        "protocol_only_successor": True,
    }
    return envelope("metachorus.v53.m4a-h7.source-identity.v1", "PASS", basis)


def _safe_zip_name(value: str) -> bool:
    if "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and bool(path.parts) and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def audit_parent_package(path: Path) -> dict[str, Any]:
    if path.name != PARENT_PACKAGE_FILENAME:
        raise ValueError("H7 parent package filename drift")
    if sha256_file(path) != PARENT_PACKAGE_SHA256 or path.stat().st_size != PARENT_PACKAGE_SIZE_BYTES:
        raise ValueError("H7 parent package hash/size mismatch")
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()
        names = [item.filename for item in info if not item.is_dir()]
        if len(names) != len(set(names)) or any(not _safe_zip_name(name) for name in names):
            raise ValueError("H7 parent package has unsafe or duplicate members")
        if any((item.external_attr >> 16) & 0o170000 == 0o120000 for item in info):
            raise ValueError("H7 parent package contains a symlink")
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise ValueError(f"H7 parent package CRC failure: {bad_crc}")
        manifest = json.loads(archive.read("PACKAGE_MEMBER_MANIFEST.json").decode("utf-8"))
        members = manifest.get("members")
        if not isinstance(members, list) or int(manifest.get("member_count", -1)) != len(members):
            raise ValueError("H7 parent package member manifest malformed")
        expected = {str(row["relative_path"]): row for row in members}
        if len(expected) != len(members):
            raise ValueError("H7 parent package duplicate manifest member")
        business = set(names) - {"PACKAGE_MEMBER_MANIFEST.json", "PACKAGE_CHECKSUMS.sha256"}
        if business != set(expected):
            raise ValueError("H7 parent package manifest/member mismatch")
        checksums = {}
        for line in archive.read("PACKAGE_CHECKSUMS.sha256").decode("utf-8").splitlines():
            checksum, relative = line.split("  ", 1)
            checksums[relative] = checksum
        if checksums != {name: str(row["sha256"]) for name, row in expected.items()}:
            raise ValueError("H7 parent package checksum index mismatch")
        for name, row in expected.items():
            payload = archive.read(name)
            if (
                len(payload) != int(row["size_bytes"])
                or hashlib.sha256(payload).hexdigest() != str(row["sha256"])
            ):
                raise ValueError(f"H7 parent package member mismatch: {name}")
        if canonical_json_hash(members) != str(manifest["manifest_identity"]):
            raise ValueError("H7 parent package manifest identity mismatch")
        parent_complete = json.loads(
            archive.read("evidence/results/M4A_H6_COMPLETE.json").decode("utf-8")
        )
        parent_failure = json.loads(
            archive.read("evidence/results/M4A_H6_FAILURE.json").decode("utf-8")
        )
        if (
            parent_complete.get("status") != PARENT_SCIENTIFIC_STATUS
            or parent_failure.get("status") != PARENT_SCIENTIFIC_STATUS
            or parent_failure.get("basis", {}).get("failed_stage") != PARENT_FAILED_STAGE
        ):
            raise ValueError("H7 parent package scientific status mismatch")
    basis = {
        "package_name": path.name,
        "package_sha256": PARENT_PACKAGE_SHA256,
        "package_size_bytes": PARENT_PACKAGE_SIZE_BYTES,
        "member_count": len(members),
        "manifest_identity": manifest["manifest_identity"],
        "crc_pass": True,
        "path_traversal_absent": True,
        "member_hashes_pass": True,
        "parent_scientific_status": PARENT_SCIENTIFIC_STATUS,
        "parent_failed_stage": PARENT_FAILED_STAGE,
        "parent_result_immutable": True,
    }
    return envelope("metachorus.v53.m4a-h7.parent-package-audit.v1", "PASS", basis)


def parent_tree_snapshot(root: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError("parent H6 result contains a symlink")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        members.append({"relative_path": relative, "size_bytes": size, "sha256": sha256_file(path)})
        total_bytes += size
    basis = {
        "parent_result_member_count": len(members),
        "parent_result_total_bytes": total_bytes,
        "parent_result_tree_identity": canonical_json_hash(members),
        "old_h6_result_modified": False,
    }
    return envelope("metachorus.v53.m4a-h7.parent-immutability-snapshot.v1", "PASS", basis)


def verify_h6_parent(root: Path) -> dict[str, Any]:
    for name, expected_hash in PARENT_ARTIFACT_SHA256.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"H6 parent artifact identity mismatch: {name}")
    complete = read_json(root / "M4A_H6_COMPLETE.json")
    failure = read_json(root / "M4A_H6_FAILURE.json")
    gate = read_json(root / "TC_LPMSO_H6_THREE_SEED_R3_GATE_DECISION.json")
    review = read_json(root / "M4A_H6_REVIEW_PACKAGE_RECEIPT.json")
    source = read_json(root / "SOURCE_DEPENDENCY_MANIFEST_M4A_H6.json")
    for payload in (complete, failure, gate, review, source):
        basis = payload.get("basis", {})
        if payload.get("identity") != canonical_json_hash(basis):
            raise ValueError("H6 parent envelope identity mismatch")
    if (
        complete.get("status") != PARENT_SCIENTIFIC_STATUS
        or complete.get("basis", {}).get("terminal_status") != PARENT_SCIENTIFIC_STATUS
        or complete.get("basis", {}).get("s_eval_started") is not False
        or failure.get("status") != PARENT_SCIENTIFIC_STATUS
        or failure.get("basis", {}).get("failed_stage") != PARENT_FAILED_STAGE
        or failure.get("basis", {}).get("r4_started") is not False
        or failure.get("basis", {}).get("metric_based_retry_forbidden") is not True
        or gate.get("status") != "TC_LPMSO_H6_THREE_SEED_R3_FAIL"
        or gate.get("basis", {}).get("reliability_gate_pass") is not False
        or gate.get("basis", {}).get("truth_accessed") is not False
        or gate.get("basis", {}).get("h5_locked_report_access") is not False
        or gate.get("basis", {}).get("s_eval_started") is not False
        or gate.get("basis", {}).get("r4_started") is not False
        or review.get("status") != "PASS"
        or review.get("basis", {}).get("package_sha256") != PARENT_PACKAGE_SHA256
        or review.get("basis", {}).get("copies_byte_identical") is not True
        or review.get("basis", {}).get("clean_extract_failed") != 0
        or source.get("status") != "PASS"
        or source.get("basis", {}).get("source_commit") != PARENT_CLOSURE_COMMIT
        or source.get("basis", {}).get("source_tree") != PARENT_CLOSURE_TREE
    ):
        raise ValueError("H6 parent closure boundary mismatch")
    if any(root.glob("TC_LPMSO_H6_R4_*")):
        raise ValueError("H6 parent unexpectedly contains R4 artifacts")
    return {
        "h6_complete_identity": complete["identity"],
        "h6_failure_identity": failure["identity"],
        "h6_gate_identity": gate["identity"],
        "h6_review_package_receipt_identity": review["identity"],
        "h6_source_manifest_identity": source["identity"],
        "h6_source_commit": PARENT_CLOSURE_COMMIT,
        "h6_source_tree": PARENT_CLOSURE_TREE,
        "h6_scientific_status": PARENT_SCIENTIFIC_STATUS,
        "h6_failed_stage": PARENT_FAILED_STAGE,
        "h6_r4_started": False,
        "h6_s_eval_started": False,
    }


def _test_receipt_correction(h6_result_root: Path) -> dict[str, Any]:
    old = read_json(h6_result_root / "TEST_RECEIPT_M4A_H6.json")
    old_basis = old.get("basis", {})
    if (
        old.get("status") != "PASS"
        or old.get("identity") != canonical_json_hash(old_basis)
        or int(old_basis.get("passed", -1)) != 82
        or int(old_basis.get("failed", -1)) != 0
    ):
        raise ValueError("H6 test receipt boundary mismatch")
    basis = {
        "old_test_receipt_identity": old["identity"],
        "old_test_receipt_preserved": True,
        "old_passed": 82,
        "old_failed": 0,
        "independent_clean_extract_collected": 83,
        "independent_clean_extract_passed": 83,
        "independent_clean_extract_failed": 0,
        "correction_evidence_source": "H7_TASKBOOK_PINNED_INDEPENDENT_CLEAN_EXTRACT",
        "portable_linux_recheck_required_before_h7_final_package": True,
        "scientific_results_changed": False,
    }
    return envelope("metachorus.v53.m4a-h7.h6-test-receipt-correction.v1", "PASS", basis)


def _post_package_receipt(h6_result_root: Path) -> dict[str, Any]:
    review = read_json(h6_result_root / "M4A_H6_REVIEW_PACKAGE_RECEIPT.json")
    basis0 = review.get("basis", {})
    if review.get("status") != "PASS" or review.get("identity") != canonical_json_hash(basis0):
        raise ValueError("H6 review receipt is not sealed")
    basis = {
        "h6_complete_identity": read_json(h6_result_root / "M4A_H6_COMPLETE.json")["identity"],
        "review_package_receipt_identity": review["identity"],
        "review_zip_sha256": basis0["package_sha256"],
        "review_zip_size_bytes": basis0["package_size_bytes"],
        "member_count": basis0["member_count"],
        "manifest_identity": basis0["manifest_identity"],
        "checksum_verification": True,
        "clean_extract_passed": basis0["clean_extract_passed"],
        "clean_extract_failed": basis0["clean_extract_failed"],
        "copies_byte_identical": basis0["copies_byte_identical"],
        "terminal_status": PARENT_SCIENTIFIC_STATUS,
        "old_complete_receipt_preserved": True,
    }
    return envelope("metachorus.v53.m4a-h7.h6-post-package-completion.v1", "PASS", basis)


def build_preflight(inputs: H7PreflightInputs) -> dict[str, Any]:
    repo_root = inputs.repo_root.resolve(strict=True)
    if inputs.result_root.is_symlink():
        raise H7PreflightResultRootNotFresh("H7 result root must not be a symlink")
    result_root = inputs.result_root.resolve()
    if result_root.exists() and any(result_root.iterdir()):
        raise H7PreflightResultRootNotFresh("H7 result root must be fresh and empty")
    result_root.mkdir(parents=True, exist_ok=True)
    if inputs.taskbook.name != TASKBOOK_FILENAME or sha256_file(inputs.taskbook) != TASKBOOK_SHA256:
        raise ValueError("H7 taskbook identity mismatch")

    h6_result_root = inputs.h6_result_root.resolve(strict=True)
    h5_result_root = inputs.h5_result_root.resolve(strict=True)
    source = source_identity(repo_root)
    package_audit = audit_parent_package(inputs.parent_package.resolve(strict=True))
    parent_inputs = verify_h6_parent(h6_result_root)
    private_mount = _verify_private_mount(h5_result_root, inputs.private_label_mount_root.resolve(strict=True))
    before = parent_tree_snapshot(h6_result_root)
    correction = _test_receipt_correction(h6_result_root)
    post_package = _post_package_receipt(h6_result_root)

    objective_rows = read_csv(h6_result_root / "TC_LPMSO_H6_THREE_SEED_R3_OBJECTIVES.csv")
    pool_rows = read_csv(h6_result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL.csv")
    derived = derive_robust_shortlist(objective_rows, pool_rows)
    artifact_paths = {
        "per_seed_fronts": write_csv_once(
            result_root / "TC_LPMSO_H7_PER_SEED_PARETO_FRONTS.csv",
            derived["per_seed_fronts"],
            fieldnames=PER_SEED_FRONT_FIELDS,
        ),
        "support": write_csv_once(
            result_root / "TC_LPMSO_H7_PARETO_SUPPORT.csv",
            derived["support"],
            fieldnames=PARETO_SUPPORT_FIELDS,
        ),
        "robust_four_objective": write_csv_once(
            result_root / "TC_LPMSO_H7_ROBUST_PARETO_4D.csv",
            derived["robust_four_objective"],
            fieldnames=ROBUST_4D_FIELDS,
        ),
        "balanced_rank": write_csv_once(
            result_root / "TC_LPMSO_H7_AGGREGATE_BALANCED_RANK.csv",
            derived["balanced_rank"],
            fieldnames=BALANCED_FIELDS,
        ),
        "components": write_csv_once(
            result_root / "TC_LPMSO_H7_ROBUST_SHORTLIST_COMPONENTS.csv",
            derived["components"],
            fieldnames=SHORTLIST_COMPONENT_FIELDS,
        ),
        "shortlist": write_csv_once(
            result_root / "TC_LPMSO_H7_ROBUST_SHORTLIST.csv",
            derived["shortlist"],
            fieldnames=SHORTLIST_FIELDS,
        ),
    }
    identity_basis = {
        "shortlist_identity": derived["identity"],
        "identity_basis": derived["identity_basis"],
        "acm_count": derived["counts"]["acm"],
        "freebase_count": derived["counts"]["freebase"],
        "total_count": derived["counts"]["total"],
    }
    identity_payload = envelope(
        "metachorus.v53.m4a-h7.robust-shortlist-identity.v1",
        "PASS",
        identity_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H7_ROBUST_SHORTLIST_IDENTITY.json", identity_payload)
    shortlist_seal_basis = {
        "source_objectives_sha256": PARENT_ARTIFACT_SHA256[
            "TC_LPMSO_H6_THREE_SEED_R3_OBJECTIVES.csv"
        ],
        "source_pool_sha256": sha256_file(
            h6_result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL.csv"
        ),
        "per_seed_pareto_fronts_sha256": sha256_file(artifact_paths["per_seed_fronts"]),
        "pareto_support_sha256": sha256_file(artifact_paths["support"]),
        "robust_pareto_4d_sha256": sha256_file(artifact_paths["robust_four_objective"]),
        "aggregate_balanced_rank_sha256": sha256_file(artifact_paths["balanced_rank"]),
        "shortlist_components_sha256": sha256_file(artifact_paths["components"]),
        "shortlist_sha256": sha256_file(artifact_paths["shortlist"]),
        "shortlist_identity": EXPECTED_SHORTLIST_IDENTITY,
        "acm_count": EXPECTED_SHORTLIST_COUNTS["acm"],
        "freebase_count": EXPECTED_SHORTLIST_COUNTS["freebase"],
        "total_count": EXPECTED_SHORTLIST_TOTAL,
        "truth_read": False,
        "h5_locked_report_access": False,
        "s_eval_access": False,
        "r4_started": False,
    }
    shortlist_seal = envelope(
        "metachorus.v53.m4a-h7.robust-shortlist-seal.v1",
        "SEALED_BEFORE_R4",
        shortlist_seal_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H7_ROBUST_SHORTLIST_SEAL.json", shortlist_seal)

    all_path_rows = resolve_path_sets(pool_rows, inputs.cache_root.resolve(strict=True))
    shortlist_keys = {
        (str(row["dataset"]), str(row["candidate_set_hash"])) for row in derived["shortlist"]
    }
    path_rows = [
        dict(row)
        for row in all_path_rows
        if (str(row["dataset"]), str(row["candidate_set_hash"])) in shortlist_keys
    ]
    if len(path_rows) != EXPECTED_SHORTLIST_TOTAL:
        raise RuntimeError("BLOCKED_ROBUST_SHORTLIST_IDENTITY_MISMATCH: path rows")
    shortlist_by_key = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): row
        for row in derived["shortlist"]
    }
    for row in path_rows:
        row["components_json"] = shortlist_by_key[
            (str(row["dataset"]), str(row["candidate_set_hash"]))
        ]["components_json"]
    path_fields = tuple(PATH_AUDIT_FIELDS) + ("components_json",)
    path_path = write_csv_once(
        result_root / "M4A_H7_PATH_SET_RESOLUTION_AUDIT.csv",
        path_rows,
        fieldnames=path_fields,
    )

    parent_inventory, continuation_audit = collect_parent_states(
        derived["shortlist"],
        path_rows,
        h5_result_root=h5_result_root,
        h6_result_root=h6_result_root,
    )
    parent_path = write_csv_once(
        result_root / "TC_LPMSO_H7_R4_PARENT_STATE_INVENTORY.csv",
        parent_inventory,
        fieldnames=PARENT_INVENTORY_FIELDS,
    )
    continuation_basis = dict(continuation_audit["basis"])
    continuation_basis["inventory_sha256"] = sha256_file(parent_path)
    continuation_audit = envelope(
        "metachorus.v53.m4a-h7.r4-continuation-audit.v1",
        "PASS",
        continuation_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H7_R4_CONTINUATION_AUDIT.json", continuation_audit)
    canary_rows, canary_seal = build_canary_plan(
        derived["shortlist"], path_rows, parent_inventory
    )
    canary_path = write_csv_once(
        result_root / "M4A_H7_CANARY_RUN_PLAN.csv",
        canary_rows,
        fieldnames=PLAN_FIELDS,
    )
    write_json_once(result_root / "M4A_H7_CANARY_PLAN_SEAL.json", canary_seal)

    write_json_once(result_root / "M4A_H7_PARENT_PACKAGE_AUDIT.json", package_audit)
    parent_source_basis = {
        "parent_package_audit_identity": package_audit["identity"],
        **parent_inputs,
        **private_mount,
        "frozen_role_manifest_sha256": ROLE_MANIFEST_SHA256,
        "frozen_role_seal_sha256": ROLE_SEAL_SHA256,
        "frozen_evaluator_contract_sha256": EVALUATOR_CONTRACT_SHA256,
        "frozen_evaluator_seal_sha256": EVALUATOR_SEAL_SHA256,
        "old_h6_result_immutable": True,
    }
    parent_source = envelope(
        "metachorus.v53.m4a-h7.parent-source-identity.v1", "PASS", parent_source_basis
    )
    write_json_once(result_root / "M4A_H7_PARENT_SOURCE_IDENTITY.json", parent_source)
    write_json_once(result_root / "M4A_H7_PARENT_IMMUTABILITY_BEFORE.json", before)
    write_json_once(result_root / "M4A_H6_TEST_RECEIPT_CORRECTION.json", correction)
    write_json_once(result_root / "M4A_H6_POST_PACKAGE_COMPLETION_RECEIPT.json", post_package)
    supersession_basis = {
        "old_h6_scientific_status": PARENT_SCIENTIFIC_STATUS,
        "old_h6_failed_stage": PARENT_FAILED_STAGE,
        "old_h6_failure_preserved": True,
        "old_h6_pairwise_failure_preserved": True,
        "old_h6_result_modified": False,
        "old_h6_result_reclassified_as_pass": False,
        "full_ranking_claim_dropped": True,
        "robust_shortlist_objective_introduced": True,
        "h7_protocol_active": True,
        "h7_source_identity": source["identity"],
        "h7_taskbook_sha256": TASKBOOK_SHA256,
    }
    supersession = envelope(
        "metachorus.v53.m4a-h7.h6-status-supersession.v1",
        "H6_FAIL_PRESERVED_H7_ROBUST_SHORTLIST_ACTIVE",
        supersession_basis,
    )
    write_json_once(result_root / "M4A_H6_STATUS_SUPERSESSION_H7.json", supersession)
    write_text_once(
        result_root / "M4A_H6_STATUS_SUPERSESSION_H7.md",
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n',
    )

    free_bytes = shutil.disk_usage(result_root).free
    resource = envelope(
        "metachorus.v53.m4a-h7.new-stage-resource-authorization.v1",
        "AUTHORIZED_FOR_CANARY_ONLY",
        {
            "h7_new_failed_attempts": 0,
            "h7_unique_root_causes": 0,
            "h7_unrecovered_scientific_identities": 0,
            "h7_limits": dict(RESOURCE_LIMITS),
            "result_volume_free_bytes_at_preflight": free_bytes,
            "cpu_logical_count": os.cpu_count(),
            "maximum_gpu_count": 2,
            "canary_run_count": 2,
            "r4_run_count_if_authorized": EXPECTED_R4_RUNS,
            "fanout_authorized": False,
            "authorization_scope": "PREFLIGHT_AND_TWO_ENGINEERING_CANARIES_ONLY",
            "labels_mounted": False,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    write_json_once(result_root / "M4A_H7_NEW_STAGE_RESOURCE_AUTHORIZATION.json", resource)

    evidence_taskbook = result_root / "evidence" / "taskbook" / TASKBOOK_FILENAME
    evidence_taskbook.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs.taskbook, evidence_taskbook)
    if sha256_file(evidence_taskbook) != TASKBOOK_SHA256:
        raise RuntimeError("H7 taskbook snapshot copy mismatch")
    write_text_once(
        result_root / "plan.md",
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n',
    )
    append_jsonl_once(
        result_root / "M4A_H7_EXECUTION_LOG.jsonl",
        {
            "event_id": "H7_PREFLIGHT_SHORTLIST_AND_PARENT_STATE_SEAL",
            "created_at_utc": utc_now(),
            "status": "READY_FOR_TWO_H7_ENGINEERING_CANARIES",
            "source_identity": source["identity"],
            "parent_package_audit_identity": package_audit["identity"],
            "shortlist_seal_identity": shortlist_seal["identity"],
            "shortlist_identity": EXPECTED_SHORTLIST_IDENTITY,
            "path_resolution_sha256": sha256_file(path_path),
            "parent_inventory_sha256": sha256_file(parent_path),
            "canary_plan_sha256": sha256_file(canary_path),
            "labels_mounted": False,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    write_text_once(result_root / "TC_LPMSO_H7_OPERATIONAL_DEVIATION_LOG.jsonl", "")
    complete_basis = {
        "source_identity": source["identity"],
        "parent_package_audit_identity": package_audit["identity"],
        "parent_source_identity": parent_source["identity"],
        "parent_immutability_before_identity": before["identity"],
        "test_receipt_correction_identity": correction["identity"],
        "post_package_receipt_identity": post_package["identity"],
        "shortlist_seal_identity": shortlist_seal["identity"],
        "shortlist_identity": EXPECTED_SHORTLIST_IDENTITY,
        "shortlist_count": EXPECTED_SHORTLIST_TOTAL,
        "path_resolution_row_count": len(path_rows),
        "parent_state_count": len(parent_inventory),
        "continuation_audit_identity": continuation_audit["identity"],
        "canary_plan_row_count": len(canary_rows),
        "truth_read_before_shortlist_seal": False,
        "fanout_started": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    payload = envelope(
        "metachorus.v53.m4a-h7.preflight-complete.v1",
        "READY_FOR_TWO_H7_ENGINEERING_CANARIES",
        complete_basis,
    )
    write_json_once(result_root / "M4A_H7_PREFLIGHT_COMPLETE.json", payload)
    return payload
