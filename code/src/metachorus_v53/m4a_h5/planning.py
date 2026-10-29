"""H5 exact-reuse and multi-fidelity execution-plan construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import (
    atomic_write_csv,
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
)

from .constants import MAX_R2_IDENTITIES, MAX_R3_IDENTITIES, MAX_R4_CONTINUATIONS, MAX_THIRD_SEED_RUNS, SEARCH_MODEL_SEEDS, THIRD_DEVELOPMENT_MODEL_SEED
from .continuation import checkpoint_state_audit, h5_physical_run_id


PLAN_FIELDS = (
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
    "parent_checkpoint_relative_path",
    "parent_checkpoint_sha256",
    "parent_representation_relative_path",
    "parent_representation_sha256",
    "continuation_from_update",
    "endpoint_update",
    "continuation_mode",
    "execution_action",
    "plan_seal_sha256",
    "roles_json",
    "labels_mounted",
    "gpu_preference",
    "run_role",
    "attempt_index",
    "status",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _endpoint_registry(
    path: Path, *, endpoint_update: int
) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = _read_csv(path)
    output: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        if int(row["endpoint_update"]) != int(endpoint_update):
            raise RuntimeError(f"unexpected endpoint in {path.name}: {row['endpoint_update']}")
        key = (
            str(row["dataset"]),
            str(row["candidate_set_hash"]),
            int(row["model_seed"]),
        )
        if key in output:
            raise RuntimeError(f"duplicate endpoint registry identity in {path.name}: {key}")
        output[key] = row
    return output


def _seal_rows(rows: Sequence[Mapping[str, Any]], stage: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_rows = []
    for row in rows:
        value = dict(row)
        value.pop("plan_seal_sha256", None)
        canonical_rows.append(value)
    basis = {
        "stage": stage,
        "row_count": len(rows),
        "rows_identity": canonical_json_hash(canonical_rows),
        "labels_mounted": False,
        "run_level_parallel_only": True,
        "maximum_gpu_count": 2,
    }
    identity = canonical_json_hash(basis)
    sealed = [{**dict(row), "plan_seal_sha256": identity} for row in rows]
    payload = {
        "schema_version": "metachorus.v53.m4a-h5.execution-plan-seal.v1",
        "status": "SEALED_BEFORE_GPU_TRAINING",
        "basis": basis,
        "identity": identity,
    }
    return sealed, payload


def _h1_update300(h1_result_root: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = _read_csv(h1_result_root / "M4A_H1_R1_CHECKPOINT_REGISTRY.csv")
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["candidate_set_hash"], int(row["model_seed"]))
        grouped.setdefault(key, []).append(row)
    output = {
        key: max(values, key=lambda row: int(row["update"]))
        for key, values in grouped.items()
    }
    if len(output) != 460:
        raise RuntimeError("H1 scientific identity registry is not exactly 460")
    return output


def _h1_update300_is_exact(h1_result_root: Path, row: Mapping[str, Any]) -> bool:
    if int(row.get("update", -1)) != 300:
        return False
    required_true = (
        "loadable_full_state",
        "optimizer_state_present",
        "rng_state_present",
        "exact_update_verified",
        "capture_rng_unchanged",
    )
    if any(str(row.get(field)) != "True" for field in required_true):
        return False
    attempt_root = h1_result_root / str(row["attempt_root_relative_path"])
    checkpoint = attempt_root / str(row["checkpoint_relative_path"])
    config = attempt_root / "RESOLVED_CONFIG.json"
    if not checkpoint.is_file() or not config.is_file():
        return False
    try:
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=300,
            expected_sha256=str(row["checkpoint_sha256"]),
        )
    except Exception:
        return False
    return bool(audit["strict_state_complete"])


def _h4_exact_r2(h4_result_root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    checkpoints = _read_csv(h4_result_root / "TC_LPMSO_H4_R2_CHECKPOINT_REGISTRY.csv")
    continuations = {
        (row["dataset"], row["candidate_set_hash"], int(row["model_seed"])): row
        for row in _read_csv(h4_result_root / "TC_LPMSO_H4_CONTINUATION_AUDIT.csv")
    }
    plans = {
        (row["dataset"], row["candidate_set_hash"], int(row["model_seed"])): row
        for row in _read_csv(h4_result_root / "TC_LPMSO_H4_R2_RUN_PLAN.csv")
    }
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for checkpoint in checkpoints:
        key = (checkpoint["dataset"], checkpoint["candidate_set_hash"], int(checkpoint["model_seed"]))
        continuation = continuations.get(key)
        plan = plans.get(key)
        if continuation is None or plan is None:
            continue
        checkpoint_path = h4_result_root / checkpoint["checkpoint_relative_path"]
        representation_path = h4_result_root / checkpoint["representation_relative_path"]
        attempt_root = checkpoint_path.parents[1]
        config_path = attempt_root / "RESOLVED_CONFIG.json"
        lineage_exact = False
        endpoint_state_audit: dict[str, Any] | None = None
        if checkpoint_path.is_file() and config_path.is_file():
            try:
                import torch

                endpoint_state_audit = checkpoint_state_audit(
                    checkpoint_path,
                    expected_update=1000,
                    expected_sha256=str(checkpoint["checkpoint_sha256"]),
                )
                config = read_json(config_path)
                payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                metadata = payload.get("metadata", {})
                lineage_exact = (
                    isinstance(metadata, Mapping)
                    and str(plan["physical_run_id"]) == str(checkpoint["physical_run_id"])
                    and str(plan["dataset"]) == str(checkpoint["dataset"])
                    and str(plan["candidate_set_hash"]) == str(checkpoint["candidate_set_hash"])
                    and int(plan["model_seed"]) == int(checkpoint["model_seed"])
                    and str(config.get("physical_run_id")) == str(checkpoint["physical_run_id"])
                    and str(config.get("dataset")) == str(checkpoint["dataset"])
                    and str(config.get("candidate_set_hash")) == str(checkpoint["candidate_set_hash"])
                    and int(config.get("candidate_set_ordinal", -1)) == int(plan["candidate_set_ordinal"])
                    and int(config.get("model_seed", -1)) == int(checkpoint["model_seed"])
                    and str(metadata.get("physical_run_id")) == str(checkpoint["physical_run_id"])
                    and str(metadata.get("path_set_hash")) == str(checkpoint["candidate_set_hash"])
                    and int(metadata.get("model_seed", -1)) == int(checkpoint["model_seed"])
                    and str(metadata.get("dataset")) == str(checkpoint["dataset"])
                    and int(metadata.get("candidate_set_ordinal", -1)) == int(plan["candidate_set_ordinal"])
                    and all(
                        metadata.get(metadata_key) == config.get(config_key)
                        for metadata_key, config_key in (
                            ("source_commit", "source_commit"),
                            ("source_tree", "source_tree"),
                            ("data_cache_hash", "cache_identity"),
                            ("augmentation_namespace", "augmentation_namespace"),
                            ("trainer_id", "trainer_id"),
                            ("source_data_identity", "source_data_identity"),
                            ("projection_identity", "projection_identity"),
                            ("score_execution_contract_hash", "score_execution_contract_hash"),
                            ("diagnostic_source_identity_hash", "diagnostic_source_identity_hash"),
                            ("diagnostic_config_hash", "diagnostic_config_hash"),
                            ("bank_contract_hash", "bank_contract_hash"),
                            ("candidate_bank_hash", "candidate_bank_hash"),
                            ("precision_runtime_contract_hash", "precision_runtime_contract_hash"),
                        )
                    )
                )
            except Exception:
                lineage_exact = False
        exact = (
            checkpoint["status"] == "PASS"
            and checkpoint["strict_state_complete"] == "True"
            and int(checkpoint["update"]) == 1000
            and continuation["status"] == "PASS"
            and continuation["strict_state_complete"] == "True"
            and int(continuation["expected_update"]) == 300
            and int(continuation["observed_update"]) == 300
            and checkpoint_path.is_file()
            and representation_path.is_file()
            and sha256_file(checkpoint_path) == checkpoint["checkpoint_sha256"]
            and sha256_file(representation_path) == checkpoint["representation_sha256"]
            and endpoint_state_audit is not None
            and bool(endpoint_state_audit["strict_state_complete"])
            and lineage_exact
        )
        if exact:
            output[key] = {
                "checkpoint": checkpoint,
                "continuation": continuation,
                "plan": plan,
                "checkpoint_path": checkpoint_path,
                "representation_path": representation_path,
                "endpoint_state_audit": endpoint_state_audit,
            }
    return output


def build_r2_plan(
    *,
    result_root: Path,
    h1_result_root: Path,
    h4_result_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pool = _read_csv(result_root / "TC_LPMSO_H5_R1_POOL_TABLE.csv")
    decision = read_json(result_root / "TC_LPMSO_H5_R1_DECISION.json")
    if decision.get("status") != "H5_R1_POOL_RECALL_PASS":
        raise RuntimeError("R2 plan is not eligible before the H5 R1 recall gate passes")
    h1 = _h1_update300(h1_result_root)
    h4 = _h4_exact_r2(h4_result_root)
    rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    for pool_row in pool:
        dataset = pool_row["dataset"]
        candidate_hash = pool_row["candidate_set_hash"]
        ordinal = int(pool_row["candidate_set_ordinal"])
        for model_seed in SEARCH_MODEL_SEEDS:
            key = (dataset, candidate_hash, model_seed)
            if key not in h1:
                raise RuntimeError(f"H5 R2 candidate is absent from H1: {key}")
            h1_row = h1[key]
            run_id = h5_physical_run_id(dataset, ordinal, model_seed)
            roles = pool_row["roles_json"]
            if key in h4:
                source = h4[key]
                checkpoint = source["checkpoint"]
                parent_attempt = str(Path(checkpoint["checkpoint_relative_path"]).parents[1]).replace("\\", "/")
                row = {
                    "stage": "R2",
                    "physical_run_id": run_id,
                    "dataset": dataset,
                    "candidate_set_id": pool_row["candidate_set_id"],
                    "candidate_set_hash": candidate_hash,
                    "candidate_set_ordinal": ordinal,
                    "model_seed": model_seed,
                    "parent_source": "H4_R2",
                    "parent_physical_run_id": checkpoint["physical_run_id"],
                    "parent_attempt_relative_path": parent_attempt,
                    "parent_checkpoint_relative_path": checkpoint["checkpoint_relative_path"],
                    "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "parent_representation_relative_path": checkpoint["representation_relative_path"],
                    "parent_representation_sha256": checkpoint["representation_sha256"],
                    "continuation_from_update": 1000,
                    "endpoint_update": 1000,
                    "continuation_mode": "EXACT_H4_R2_REUSE",
                    "execution_action": "EVALUATION_ONLY_REUSE",
                    "roles_json": roles,
                    "labels_mounted": False,
                    "gpu_preference": ordinal % 2,
                    "run_role": "SEARCH",
                    "attempt_index": 0,
                    "status": "REUSED_READY_FOR_H5_EVALUATION",
                }
                reuse_rows.append(
                    {
                        "dataset": dataset,
                        "candidate_set_id": pool_row["candidate_set_id"],
                        "candidate_set_hash": candidate_hash,
                        "candidate_set_ordinal": ordinal,
                        "model_seed": model_seed,
                        "h4_physical_run_id": checkpoint["physical_run_id"],
                        "h5_physical_run_id": run_id,
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                        "representation_sha256": checkpoint["representation_sha256"],
                        "source_data_path_identity_match": True,
                        "candidate_model_seed_match": True,
                        "parent_update300_state_match": True,
                        "optimizer_rng_augmentation_complete": True,
                        "endpoint_update1000_exact": True,
                        "o0_s0_contract_match": True,
                        "checkpoint_lineage_metadata_match": True,
                        "training_reexecuted": False,
                        "status": "EXACT_REUSE",
                    }
                )
            else:
                h1_exact = _h1_update300_is_exact(h1_result_root, h1_row)
                row = {
                    "stage": "R2",
                    "physical_run_id": run_id,
                    "dataset": dataset,
                    "candidate_set_id": pool_row["candidate_set_id"],
                    "candidate_set_hash": candidate_hash,
                    "candidate_set_ordinal": ordinal,
                    "model_seed": model_seed,
                    "parent_source": "H1_R1" if h1_exact else "H1_CONFIG_ONLY",
                    "parent_physical_run_id": h1_row["physical_run_id"],
                    "parent_attempt_relative_path": h1_row["attempt_root_relative_path"],
                    "parent_checkpoint_relative_path": (
                        str(Path(h1_row["attempt_root_relative_path"]) / h1_row["checkpoint_relative_path"]).replace("\\", "/")
                        if h1_exact
                        else ""
                    ),
                    "parent_checkpoint_sha256": h1_row["checkpoint_sha256"] if h1_exact else "",
                    "parent_representation_relative_path": (
                        str(Path(h1_row["attempt_root_relative_path"]) / h1_row["fused_representation_relative_path"]).replace("\\", "/")
                        if h1_exact
                        else ""
                    ),
                    "parent_representation_sha256": h1_row["fused_representation_sha256"] if h1_exact else "",
                    "continuation_from_update": 300 if h1_exact else 0,
                    "endpoint_update": 1000,
                    "continuation_mode": (
                        "STRICT_CONTINUATION" if h1_exact else "FROM_ZERO_MISSING_R2_STATE"
                    ),
                    "execution_action": (
                        "TRAIN_MISSING_R2_STATE"
                        if h1_exact
                        else "TRAIN_MISSING_R2_STATE_FROM_ZERO"
                    ),
                    "roles_json": roles,
                    "labels_mounted": False,
                    "gpu_preference": ordinal % 2,
                    "run_role": "SEARCH",
                    "attempt_index": 0,
                    "status": "PENDING_GPU",
                }
            rows.append(row)
    expected_identity_count = len(pool) * len(SEARCH_MODEL_SEEDS)
    if (
        len(rows) != expected_identity_count
        or len(rows) > MAX_R2_IDENTITIES
        or len({row["physical_run_id"] for row in rows}) != len(rows)
    ):
        raise RuntimeError(f"H5 R2 plan identity violation: {len(rows)}")
    rows, seal = _seal_rows(rows, "R2")
    atomic_write_csv(result_root / "TC_LPMSO_H5_R2_RUN_PLAN.csv", rows, fieldnames=list(PLAN_FIELDS))
    reuse_fields = list(reuse_rows[0]) if reuse_rows else [
        "dataset", "candidate_set_id", "candidate_set_hash", "candidate_set_ordinal", "model_seed",
        "h4_physical_run_id", "h5_physical_run_id", "checkpoint_sha256", "representation_sha256",
        "source_data_path_identity_match", "candidate_model_seed_match", "parent_update300_state_match",
        "optimizer_rng_augmentation_complete", "endpoint_update1000_exact", "o0_s0_contract_match",
        "checkpoint_lineage_metadata_match", "training_reexecuted", "status",
    ]
    atomic_write_csv(result_root / "TC_LPMSO_H5_R2_REUSE_REGISTRY.csv", reuse_rows, fieldnames=reuse_fields)
    atomic_write_json(result_root / "TC_LPMSO_H5_R2_PLAN_SEAL.json", seal)
    budget_basis = {
        "r2_identity_count": len(rows),
        "exact_h4_reuse_count": len(reuse_rows),
        "missing_state_training_count": len(rows) - len(reuse_rows),
        "restart_from_zero_count": sum(
            int(row["continuation_from_update"]) == 0 for row in rows
        ),
        "maximum_identity_count": MAX_R2_IDENTITIES,
        "r2_no_pruning": True,
        "gpu_count_maximum": 2,
        "run_level_parallel_only": True,
    }
    budget = {
        "schema_version": "metachorus.v53.m4a-h5.r2-budget.v1",
        "status": "READY_FOR_R2_MISSING_STATE_EXECUTION",
        "basis": budget_basis,
        "identity": canonical_json_hash(budget_basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_R2_BUDGET.json", budget)
    return rows, reuse_rows, budget


def build_r3_plan(
    *, result_root: Path, h4_result_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bridge = read_json(result_root / "TC_LPMSO_H5_R2_BRIDGE_DECISION.json")
    if bridge.get("status") != "H5_R2_BRIDGE_PASS":
        raise RuntimeError("H5 R3 plan is not eligible before the R2 bridge passes")
    r2_plan = _read_csv(result_root / "TC_LPMSO_H5_R2_RUN_PLAN.csv")
    r2_registry = _endpoint_registry(
        result_root / "TC_LPMSO_H5_R2_CHECKPOINT_REGISTRY.csv",
        endpoint_update=1000,
    )
    rows: list[dict[str, Any]] = []
    for source in r2_plan:
        key = (
            str(source["dataset"]),
            str(source["candidate_set_hash"]),
            int(source["model_seed"]),
        )
        registry = r2_registry.get(key)
        if registry is None:
            raise RuntimeError(f"missing completed H5 R2 registry identity: {key}")
        if registry["source_root"] == "H4_PARENT_IMMUTABLE":
            parent_source = "H4_R2"
            artifact_root = h4_result_root
        elif registry["source_root"] == "H5_RESULT_ROOT":
            parent_source = "H5_R2"
            artifact_root = result_root
        else:
            raise RuntimeError(f"unsupported R2 endpoint source: {registry['source_root']}")
        checkpoint_relative = registry["checkpoint_relative_path"]
        representation_relative = registry["representation_relative_path"]
        checkpoint = artifact_root / checkpoint_relative
        representation = artifact_root / representation_relative
        checkpoint_sha = registry["checkpoint_sha256"]
        representation_sha = registry["representation_sha256"]
        if (
            not checkpoint.is_file()
            or not representation.is_file()
            or sha256_file(checkpoint) != checkpoint_sha
            or sha256_file(representation) != representation_sha
        ):
            raise RuntimeError(f"changed completed H5 R2 state: {source['physical_run_id']}")
        parent_attempt = str(Path(checkpoint_relative).parents[1]).replace("\\", "/")
        parent_run_id = source["parent_physical_run_id"] if parent_source == "H4_R2" else source["physical_run_id"]
        successful_attempt_index = int(registry.get("successful_attempt_index", 0))
        rows.append(
            {
                **{key: source[key] for key in (
                    "physical_run_id", "dataset", "candidate_set_id", "candidate_set_hash",
                    "candidate_set_ordinal", "model_seed", "roles_json", "gpu_preference"
                )},
                "stage": "R3",
                "parent_source": parent_source,
                "parent_physical_run_id": parent_run_id,
                "parent_attempt_relative_path": parent_attempt,
                "parent_checkpoint_relative_path": checkpoint_relative,
                "parent_checkpoint_sha256": checkpoint_sha,
                "parent_representation_relative_path": representation_relative,
                "parent_representation_sha256": representation_sha,
                "continuation_from_update": 1000,
                "endpoint_update": 2000,
                "continuation_mode": "STRICT_CONTINUATION",
                "execution_action": "TRAIN_ALL_R2_STATES_TO_R3",
                "labels_mounted": False,
                "run_role": "SEARCH",
                "attempt_index": successful_attempt_index,
                "status": "PENDING_GPU",
            }
        )
    if len(rows) != len(r2_plan) or len(rows) > MAX_R3_IDENTITIES:
        raise RuntimeError("H5 R3 must contain every and only R2 identity")
    rows, seal = _seal_rows(rows, "R3")
    atomic_write_csv(result_root / "TC_LPMSO_H5_R3_RUN_PLAN.csv", rows, fieldnames=list(PLAN_FIELDS))
    atomic_write_json(result_root / "TC_LPMSO_H5_R3_PLAN_SEAL.json", seal)
    return rows, seal


def build_third_seed_plan(
    *, result_root: Path, h1_result_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate = read_json(result_root / "TC_LPMSO_H5_R3_GATE_DECISION.json")
    if gate.get("status") != "TC_LPMSO_H5_R3_GATE_PASS":
        raise RuntimeError("third seed is not eligible before all H5 R3 gates pass")
    pool = _read_csv(result_root / "TC_LPMSO_H5_R3_AUDIT_POOL_TABLE.csv")
    h1 = _h1_update300(h1_result_root)
    rows: list[dict[str, Any]] = []
    for candidate in pool:
        source_key = (
            candidate["dataset"],
            candidate["candidate_set_hash"],
            SEARCH_MODEL_SEEDS[0],
        )
        source = h1[source_key]
        rows.append(
            {
                "stage": "THIRD_SEED_R3_AUDIT",
                "physical_run_id": h5_physical_run_id(
                    candidate["dataset"], int(candidate["candidate_set_ordinal"]), THIRD_DEVELOPMENT_MODEL_SEED
                ),
                "dataset": candidate["dataset"],
                "candidate_set_id": candidate["candidate_set_id"],
                "candidate_set_hash": candidate["candidate_set_hash"],
                "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
                "model_seed": THIRD_DEVELOPMENT_MODEL_SEED,
                "parent_source": "H1_CONFIG_ONLY",
                "parent_physical_run_id": source["physical_run_id"],
                "parent_attempt_relative_path": source["attempt_root_relative_path"],
                "parent_checkpoint_relative_path": "",
                "parent_checkpoint_sha256": "",
                "parent_representation_relative_path": "",
                "parent_representation_sha256": "",
                "continuation_from_update": 0,
                "endpoint_update": 2000,
                "continuation_mode": "FROM_ZERO_THIRD_DEVELOPMENT_SEED",
                "execution_action": "TRAIN_THIRD_SEED_FROM_ZERO",
                "roles_json": candidate["roles_json"],
                "labels_mounted": False,
                "gpu_preference": int(candidate["candidate_set_ordinal"]) % 2,
                "run_role": "THIRD_DEVELOPMENT_SEED_AUDIT",
                "attempt_index": 0,
                "status": "PENDING_GPU",
            }
        )
    if len(rows) > MAX_THIRD_SEED_RUNS or len({row["physical_run_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"H5 third-seed plan exceeds 31 unique runs: {len(rows)}")
    rows, seal = _seal_rows(rows, "THIRD_SEED_R3_AUDIT")
    atomic_write_csv(result_root / "TC_LPMSO_H5_THIRD_SEED_RUN_PLAN.csv", rows, fieldnames=list(PLAN_FIELDS))
    atomic_write_json(result_root / "TC_LPMSO_H5_THIRD_SEED_PLAN_SEAL.json", seal)
    return rows, seal


def build_r4_plan(*, result_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    promotion = _read_csv(result_root / "TC_LPMSO_H5_R4_PROMOTION_TABLE.csv")
    promotion_seal = read_json(result_root / "TC_LPMSO_H5_R4_PROMOTION_SEAL.json")
    if promotion_seal.get("status") != "SEALED_BEFORE_R4":
        raise RuntimeError("H5 R4 promotion is not sealed")
    r3_plan = {
        (row["dataset"], row["candidate_set_hash"], int(row["model_seed"])): row
        for row in _read_csv(result_root / "TC_LPMSO_H5_R3_RUN_PLAN.csv")
    }
    third_plan = {
        (row["dataset"], row["candidate_set_hash"], int(row["model_seed"])): row
        for row in _read_csv(result_root / "TC_LPMSO_H5_THIRD_SEED_RUN_PLAN.csv")
    }
    r3_registry = _endpoint_registry(
        result_root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv",
        endpoint_update=2000,
    )
    third_registry = _endpoint_registry(
        result_root / "TC_LPMSO_H5_THIRD_SEED_CHECKPOINT_REGISTRY.csv",
        endpoint_update=2000,
    )
    rows: list[dict[str, Any]] = []
    for candidate in promotion:
        for seed in (*SEARCH_MODEL_SEEDS, THIRD_DEVELOPMENT_MODEL_SEED):
            key = (candidate["dataset"], candidate["candidate_set_hash"], seed)
            source = r3_plan.get(key) if seed in SEARCH_MODEL_SEEDS else third_plan.get(key)
            if source is None:
                raise RuntimeError(f"H5 R4 predecessor missing: {key}")
            registry = (r3_registry if seed in SEARCH_MODEL_SEEDS else third_registry).get(key)
            if registry is None or registry["source_root"] != "H5_RESULT_ROOT":
                raise RuntimeError(f"H5 R4 predecessor registry missing: {key}")
            checkpoint_relative = registry["checkpoint_relative_path"]
            representation_relative = registry["representation_relative_path"]
            checkpoint = result_root / checkpoint_relative
            representation = result_root / representation_relative
            if (
                not checkpoint.is_file()
                or not representation.is_file()
                or sha256_file(checkpoint) != registry["checkpoint_sha256"]
                or sha256_file(representation) != registry["representation_sha256"]
            ):
                raise RuntimeError(f"H5 R4 predecessor artifacts missing: {key}")
            attempt = Path(checkpoint_relative).parents[1]
            rows.append(
                {
                    "stage": "R4",
                    "physical_run_id": source["physical_run_id"],
                    "dataset": candidate["dataset"],
                    "candidate_set_id": candidate["candidate_set_id"],
                    "candidate_set_hash": candidate["candidate_set_hash"],
                    "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
                    "model_seed": seed,
                    "parent_source": "H5_R3" if seed in SEARCH_MODEL_SEEDS else "H5_THIRD_SEED",
                    "parent_physical_run_id": source["physical_run_id"],
                    "parent_attempt_relative_path": str(attempt).replace("\\", "/"),
                    "parent_checkpoint_relative_path": checkpoint_relative,
                    "parent_checkpoint_sha256": registry["checkpoint_sha256"],
                    "parent_representation_relative_path": representation_relative,
                    "parent_representation_sha256": registry["representation_sha256"],
                    "continuation_from_update": 2000,
                    "endpoint_update": 3000,
                    "continuation_mode": "STRICT_CONTINUATION",
                    "execution_action": "TRAIN_R4_TO_3000",
                    "roles_json": candidate["roles_json"],
                    "labels_mounted": False,
                    "gpu_preference": int(candidate["candidate_set_ordinal"]) % 2,
                    "run_role": (
                        "SEARCH"
                        if seed in SEARCH_MODEL_SEEDS
                        else "THIRD_DEVELOPMENT_SEED_AUDIT"
                    ),
                    "attempt_index": int(registry.get("successful_attempt_index", source["attempt_index"])),
                    "status": "PENDING_GPU",
                }
            )
    if len(rows) > MAX_R4_CONTINUATIONS or len({(row["physical_run_id"], row["model_seed"]) for row in rows}) != len(rows):
        raise RuntimeError(f"H5 R4 plan exceeds 42 continuations: {len(rows)}")
    rows, seal = _seal_rows(rows, "R4")
    atomic_write_csv(result_root / "TC_LPMSO_H5_R4_RUN_PLAN.csv", rows, fieldnames=list(PLAN_FIELDS))
    atomic_write_json(result_root / "TC_LPMSO_H5_R4_PLAN_SEAL.json", seal)
    return rows, seal
