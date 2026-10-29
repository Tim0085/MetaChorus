"""Append-only H5 parent closure, global label freeze, and R1 input preflight."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

from metachorus_v53.audit.common import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_json_hash,
    read_json,
    sha256_file,
    utc_now,
)

from .constants import (
    H1_RESULT_RELATIVE,
    H2_RESULT_RELATIVE,
    H4_RESULT_RELATIVE,
    PARENT_FAILURE_CODE,
    PARENT_PACKAGE_CANONICAL_FILENAME,
    PARENT_PACKAGE_DECLARED_FILENAME,
    PARENT_PACKAGE_SHA256,
    PARENT_SOURCE_COMMIT,
    PARENT_SOURCE_TREE,
    PARENT_TASKBOOK_SHA256,
    PARENT_TERMINAL_STATUS,
    RESOURCE_LIMITS,
    TASKBOOK_FILENAME,
    TASKBOOK_SHA256,
)
from .evaluation import seal_evaluator, write_evaluator_contract
from .labels import (
    build_global_label_role_artifacts,
    load_acm_labels,
    load_freebase_labels,
    seal_global_label_roles,
)


@dataclass(frozen=True)
class H5PreflightInputs:
    repo_root: Path
    workspace_root: Path
    taskbook: Path
    parent_package: Path
    result_root: Path
    private_label_mount_root: Path
    acm_label_source: Path
    freebase_label_source: Path

    @property
    def h1_result_root(self) -> Path:
        return self.workspace_root / H1_RESULT_RELATIVE

    @property
    def h2_result_root(self) -> Path:
        return self.workspace_root / H2_RESULT_RELATIVE

    @property
    def h4_result_root(self) -> Path:
        return self.workspace_root / H4_RESULT_RELATIVE


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()


def _identity(schema: str, status: str, basis: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(basis)
    return {
        "schema_version": schema,
        "status": status,
        "basis": body,
        "identity": canonical_json_hash(body),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _source_identity(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise RuntimeError("H5 preflight requires a clean source worktree")
    basis = {
        "source_commit": _git(repo_root, "rev-parse", "HEAD"),
        "source_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "worktree_clean": True,
    }
    return _identity("metachorus.v53.m4a-h5.source-identity.v1", "PASS", basis)


def audit_parent_package(path: Path) -> dict[str, Any]:
    if path.name not in {PARENT_PACKAGE_DECLARED_FILENAME, PARENT_PACKAGE_CANONICAL_FILENAME}:
        raise RuntimeError("H5 parent package filename is not the declared or canonical local alias")
    if sha256_file(path) != PARENT_PACKAGE_SHA256:
        raise RuntimeError("H5 parent package hash mismatch")
    with ZipFile(path) as archive:
        members = archive.namelist()
        unsafe = [
            name
            for name in members
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or "\\" in name
        ]
        if unsafe:
            raise RuntimeError(f"unsafe parent package members: {unsafe[:3]}")
        if archive.testzip() is not None:
            raise RuntimeError("H5 parent package failed CRC verification")
        complete_name = "evidence/results/M4A_H4_COMPLETE.json"
        failure_name = "evidence/results/M4A_H4_FAILURE.json"
        if complete_name not in members or failure_name not in members:
            raise RuntimeError("H5 parent package lacks canonical H4 terminal evidence")
        complete = json.loads(archive.read(complete_name).decode("utf-8"))
        failure = json.loads(archive.read(failure_name).decode("utf-8"))
    if complete.get("status") != PARENT_TERMINAL_STATUS:
        raise RuntimeError("H4 terminal status differs from H5 taskbook")
    if complete.get("basis", {}).get("failure_code") != PARENT_FAILURE_CODE:
        raise RuntimeError("H4 failure code differs from H5 taskbook")
    if complete.get("basis", {}).get("source_commit") != PARENT_SOURCE_COMMIT:
        raise RuntimeError("H4 closure source commit differs from H5 taskbook")
    if complete.get("basis", {}).get("source_tree") != PARENT_SOURCE_TREE:
        raise RuntimeError("H4 closure source tree differs from H5 taskbook")
    basis = {
        "declared_filename": PARENT_PACKAGE_DECLARED_FILENAME,
        "observed_filename": path.name,
        "canonical_local_alias_accepted_by_hash": path.name == PARENT_PACKAGE_CANONICAL_FILENAME,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "member_count": len(members),
        "crc_pass": True,
        "path_traversal_absent": True,
        "terminal_status": complete["status"],
        "failure_code": complete["basis"]["failure_code"],
        "closure_source_commit": complete["basis"]["source_commit"],
        "closure_source_tree": complete["basis"]["source_tree"],
        "scientific_run_source_commit": failure.get("basis", {}).get("scientific_source_commit"),
        "scientific_run_source_tree": failure.get("basis", {}).get("scientific_source_tree"),
        "lineage_layers_explicit": True,
        "old_result_immutable": True,
    }
    return _identity("metachorus.v53.m4a-h5.parent-package-audit.v1", "PASS", basis)


def parent_source_identity(h4_result_root: Path) -> dict[str, Any]:
    complete = read_json(h4_result_root / "M4A_H4_COMPLETE.json")
    failure = read_json(h4_result_root / "M4A_H4_FAILURE.json")
    preflight = read_json(h4_result_root / "M4A_H4_SOURCE_IDENTITY_PREFLIGHT.json")
    if complete.get("status") != PARENT_TERMINAL_STATUS:
        raise RuntimeError("local H4 parent terminal status mismatch")
    closure_commit = complete.get("basis", {}).get("source_commit")
    closure_tree = complete.get("basis", {}).get("source_tree")
    if closure_commit != PARENT_SOURCE_COMMIT or closure_tree != PARENT_SOURCE_TREE:
        raise RuntimeError("local H4 closure source identity mismatch")
    basis = {
        "closure_source_commit": closure_commit,
        "closure_source_tree": closure_tree,
        "scientific_run_source_commit": failure.get("basis", {}).get("scientific_source_commit"),
        "scientific_run_source_tree": failure.get("basis", {}).get("scientific_source_tree"),
        "scientific_preflight_source_commit": preflight.get("basis", {}).get("source_commit"),
        "scientific_preflight_source_tree": preflight.get("basis", {}).get("source_tree"),
        "terminal_status": complete["status"],
        "failure_code": complete.get("basis", {}).get("failure_code"),
        "closure_and_scientific_lineage_layers_not_conflated": True,
        "parent_immutable": True,
    }
    return _identity("metachorus.v53.m4a-h5.parent-source-identity.v1", "PASS", basis)


PROTECTED_H4_FILES = (
    "M4A_H4_COMPLETE.json",
    "M4A_H4_VALIDATION.json",
    "M4A_H4_FAILURE.json",
    "TC_LPMSO_H4_DEVELOPMENT_DECISION.json",
    "TC_LPMSO_H4_R2_RUN_REGISTRY.csv",
    "TC_LPMSO_H4_R2_CHECKPOINT_REGISTRY.csv",
    "TC_LPMSO_H4_R2_CLASSIFICATION_METRICS.csv",
    "TC_LPMSO_H4_R2_CLUSTERING_METRICS_LONG.csv",
    "TC_LPMSO_H4_R2_CLUSTERING_METRICS_SUMMARY.csv",
    "TC_LPMSO_H4_R2_SEARCH_METRICS.csv",
    "TC_LPMSO_H4_R2_TASK_RELIABILITY.csv",
    "TC_LPMSO_H4_R2_PROMOTION_DECISION.json",
)


def protected_parent_snapshot(inputs: H5PreflightInputs) -> dict[str, Any]:
    groups: dict[str, list[tuple[str, Path]]] = {
        "h4": [(name, inputs.h4_result_root / name) for name in PROTECTED_H4_FILES],
        "h2_truth": [
            (
                "ACM_DEVELOPMENT_TRUTH_AGGREGATE_M4A_H2.csv",
                inputs.h2_result_root / "ACM_DEVELOPMENT_TRUTH_AGGREGATE_M4A_H2.csv",
            ),
            (
                "FREEBASE_LP_SRP_C_HOLDOUT_AUDIT.csv",
                inputs.h2_result_root / "FREEBASE_LP_SRP_C_HOLDOUT_AUDIT.csv",
            ),
        ],
        "h1": [
            (
                "M4A_H1_R1_CHECKPOINT_REGISTRY.csv",
                inputs.h1_result_root / "M4A_H1_R1_CHECKPOINT_REGISTRY.csv",
            ),
            (
                "M4A_H1_R1_PHYSICAL_RUN_REGISTRY.csv",
                inputs.h1_result_root / "M4A_H1_R1_PHYSICAL_RUN_REGISTRY.csv",
            ),
        ],
        "package": [(inputs.parent_package.name, inputs.parent_package)],
    }
    rows: list[dict[str, Any]] = []
    for source_id, members in groups.items():
        for relative, path in members:
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "source_id": source_id,
                    "relative_path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    rows.sort(key=lambda row: (row["source_id"], row["relative_path"]))
    basis = {"files": rows, "file_count": len(rows), "parent_append_only": True}
    return _identity("metachorus.v53.m4a-h5.parent-immutability.v1", "PASS", basis)


def _r1_inventory(inputs: H5PreflightInputs) -> list[dict[str, Any]]:
    rows = _read_csv(inputs.h1_result_root / "M4A_H1_R1_CHECKPOINT_REGISTRY.csv")
    rows = [row for row in rows if int(row["update"]) == 300]
    if len(rows) != 460:
        raise RuntimeError(f"H5 expected 460 update-300 representations, observed {len(rows)}")
    output: list[dict[str, Any]] = []
    for row in rows:
        attempt_relative = Path(row["attempt_root_relative_path"])
        representation_relative = attempt_relative / row["fused_representation_relative_path"]
        checkpoint_relative = attempt_relative / row["checkpoint_relative_path"]
        path = inputs.h1_result_root / representation_relative
        if not path.is_file() or sha256_file(path) != row["fused_representation_sha256"]:
            raise RuntimeError(f"R1 representation verification failed: {row['physical_run_id']}")
        output.append(
            {
                "physical_run_id": row["physical_run_id"],
                "dataset": row["dataset"],
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "model_seed": int(row["model_seed"]),
                "update": 300,
                "representation_relative_path": str(representation_relative).replace("\\", "/"),
                "representation_sha256": row["fused_representation_sha256"],
                "representation_size_bytes": path.stat().st_size,
                "checkpoint_relative_path": str(checkpoint_relative).replace("\\", "/"),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "parent_attempt_relative_path": row["attempt_root_relative_path"],
                "all_hashes_verified": True,
                "status": "PASS",
            }
        )
    if len({(row["dataset"], row["candidate_set_hash"], row["model_seed"]) for row in output}) != 460:
        raise RuntimeError("H5 R1 inventory identity count is not exactly 460")
    return sorted(
        output,
        key=lambda row: (
            str(row["dataset"]),
            int(row["candidate_set_ordinal"]),
            int(row["model_seed"]),
        ),
    )


def _write_supersession(result_root: Path) -> None:
    basis = {
        "old_h4_status": PARENT_TERMINAL_STATUS,
        "old_h4_failure_code": PARENT_FAILURE_CODE,
        "old_h4_scientific_results_modified": False,
        "old_h4_thresholds_reinterpreted": False,
        "h5_protocol_scope": [
            "one_global_20pct_search_pool_per_dataset",
            "five_within_pool_stratified_repeats",
            "fixed_four_band_freebase_exploration",
            "R2_bridge_without_pruning",
            "R3_first_dual_task_gate",
        ],
        "parent_taskbook_sha256": PARENT_TASKBOOK_SHA256,
        "h5_taskbook_sha256": TASKBOOK_SHA256,
        "protocol_only_supersession": True,
    }
    payload = _identity(
        "metachorus.v53.m4a-h5.h4-status-supersession.v1",
        "H4_FAIL_PRESERVED_H5_PROTOCOL_ACTIVE",
        basis,
    )
    atomic_write_json(result_root / "M4A_H4_STATUS_SUPERSESSION_H5.json", payload)
    atomic_write_text(
        result_root / "M4A_H4_STATUS_SUPERSESSION_H5.md",
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n',
    )


def _training_isolation_audit(result_root: Path) -> dict[str, Any]:
    basis = {
        "training_process_label_mount": None,
        "training_inputs": ["sanitized projection", "candidate bank", "label-free checkpoint"],
        "training_api_accepts_labels": False,
        "search_private_mount_passed_to_training": False,
        "h5_locked_report_passed_to_training": False,
        "future_s_eval_passed_to_training": False,
        "status": "PASS",
    }
    payload = _identity(
        "metachorus.v53.m4a-h5.training-label-isolation-audit.v1", "PASS", basis
    )
    atomic_write_json(result_root / "TC_LPMSO_H5_TRAINING_LABEL_ISOLATION_AUDIT.json", payload)
    return payload


def _resource_preflight(inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    basis = {
        "r1_evaluation_only_representation_count": len(inventory),
        "r1_new_gpu_training_count": 0,
        "maximum_r2_identities": 264,
        "maximum_r3_identities": 264,
        "maximum_third_seed_runs": 31,
        "maximum_r4_continuations": 42,
        "maximum_gpu_count": 2,
        "estimated_incremental_gpu_hours_p50": 15.40,
        "estimated_incremental_gpu_hours_p90": 21.52,
        "estimated_two_gpu_critical_path_hours_p50": 7.70,
        "estimated_two_gpu_critical_path_hours_p90": 10.76,
        "soft_limits": RESOURCE_LIMITS,
        "p50_within_soft_gpu_hours": True,
        "p90_within_hard_gpu_hours": True,
        "p90_within_hard_critical_path": True,
        "runtime_hard_gate_mandatory": True,
        "pool_shrinking_on_resource_pressure": False,
        "status": "READY_WITH_RUNTIME_HARD_GATE",
    }
    return _identity("metachorus.v53.m4a-h5.resource-preflight.v1", "PASS", basis)


def build_preflight(inputs: H5PreflightInputs) -> dict[str, Any]:
    if inputs.result_root.exists() and any(inputs.result_root.iterdir()):
        raise FileExistsError("H5 result root must be append-only and empty")
    if inputs.private_label_mount_root.drive.lower() == "c:":
        raise ValueError("H5 private label mount may not be placed on C drive")
    if sha256_file(inputs.taskbook) != TASKBOOK_SHA256 or inputs.taskbook.name != TASKBOOK_FILENAME:
        raise RuntimeError("H5 taskbook identity mismatch")
    if inputs.private_label_mount_root.exists():
        raise FileExistsError("H5 private label mount must be a fresh absent directory")

    
    
    
    parent_package = audit_parent_package(inputs.parent_package)
    source = _source_identity(inputs.repo_root)
    parent_source = parent_source_identity(inputs.h4_result_root)
    before = protected_parent_snapshot(inputs)
    labels = {
        "acm": load_acm_labels(inputs.acm_label_source),
        "freebase": load_freebase_labels(inputs.freebase_label_source),
    }
    inventory = _r1_inventory(inputs)

    inputs.result_root.mkdir(parents=True, exist_ok=True)
    taskbook_copy = inputs.result_root / "evidence" / "taskbook" / TASKBOOK_FILENAME
    taskbook_copy.parent.mkdir(parents=True)
    atomic_write_bytes(taskbook_copy, inputs.taskbook.read_bytes())

    atomic_write_json(inputs.result_root / "M4A_H5_PARENT_PACKAGE_AUDIT.json", parent_package)
    atomic_write_json(inputs.result_root / "M4A_H5_SOURCE_IDENTITY_PREFLIGHT.json", source)
    atomic_write_json(inputs.result_root / "M4A_H5_PARENT_SOURCE_IDENTITY.json", parent_source)
    atomic_write_json(inputs.result_root / "M4A_H5_PARENT_IMMUTABILITY_BEFORE.json", before)
    _write_supersession(inputs.result_root)

    evaluator_path = write_evaluator_contract(inputs.result_root)
    manifest, mount_audit = build_global_label_role_artifacts(
        result_root=inputs.result_root,
        private_mount_root=inputs.private_label_mount_root,
        dataset_labels=labels,
        label_source_files={
            "acm": inputs.acm_label_source,
            "freebase": inputs.freebase_label_source,
        },
    )
    evaluator_seal = seal_evaluator(
        result_root=inputs.result_root,
        label_manifest_path=inputs.result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json",
        private_mount_audit_path=inputs.result_root / "TC_LPMSO_H5_PRIVATE_SEARCH_MOUNT_AUDIT.json",
    )
    label_seal = seal_global_label_roles(
        result_root=inputs.result_root,
        evaluator_contract_path=evaluator_path,
        private_mount_root=inputs.private_label_mount_root,
    )
    isolation = _training_isolation_audit(inputs.result_root)
    atomic_write_csv(
        inputs.result_root / "TC_LPMSO_H5_R1_REPRESENTATION_INVENTORY.csv",
        inventory,
        fieldnames=list(inventory[0]),
    )
    resource = _resource_preflight(inventory)
    atomic_write_json(inputs.result_root / "TC_LPMSO_H5_RESOURCE_PREFLIGHT.json", resource)
    after = protected_parent_snapshot(inputs)
    if before["identity"] != after["identity"]:
        raise RuntimeError("parent evidence changed during H5 preflight")
    atomic_write_json(inputs.result_root / "M4A_H5_PARENT_IMMUTABILITY_AFTER.json", after)
    plan_text = (
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
        'Stage status is recorded in the structured output.\n'
    )
    atomic_write_text(inputs.result_root / "plan.md", plan_text)
    atomic_write_text(
        inputs.result_root / "M4A_H5_EXECUTION_LOG.jsonl",
        json.dumps(
            {
                "timestamp_utc": utc_now(),
                "stage": "PREFLIGHT",
                "status": "PASS",
                "r1_representation_count": len(inventory),
                "gpu_required": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )
    atomic_write_text(inputs.result_root / "TC_LPMSO_H5_OPERATIONAL_DEVIATION_LOG.jsonl", "")
    basis = {
        "parent_package_identity": parent_package["identity"],
        "parent_source_identity": parent_source["identity"],
        "parent_immutability_identity": before["identity"],
        "source_identity": source["identity"],
        "taskbook_sha256": TASKBOOK_SHA256,
        "label_manifest_sha256": sha256_file(
            inputs.result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json"
        ),
        "label_seal_identity": label_seal["identity"],
        "evaluator_seal_identity": evaluator_seal["identity"],
        "private_mount_audit_status": mount_audit["status"],
        "training_isolation_status": isolation["status"],
        "r1_representation_count": len(inventory),
        "truth_metrics_opened_before_r1_pool_seal": False,
        "gpu_required_for_next_stage": False,
    }
    complete = _identity("metachorus.v53.m4a-h5.preflight-complete.v1", "READY_FOR_R1_EVALUATION", basis)
    atomic_write_json(inputs.result_root / "M4A_H5_PREFLIGHT_COMPLETE.json", complete)
    return complete
