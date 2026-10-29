"""H7 parent-state inventory and sealed canary/R4 planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file
from metachorus_v53.m4a_h4.continuation import checkpoint_state_audit

from .common import envelope, read_csv, safe_relative, seal_rows
from .constants import (
    CANARY_ENDPOINT,
    CONTROLS,
    DEVELOPMENT_MODEL_SEEDS,
    EXPECTED_PARENT_STATES,
    EXPECTED_R4_RUNS,
    EXPECTED_SHORTLIST_COUNTS,
    H5_R3_CHECKPOINT_REGISTRY_SHA256,
    PARENT_ARTIFACT_SHA256,
    PARENT_UPDATE,
    R4_ENDPOINT,
)


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
    "template_config_sha256",
    "path_ids_json",
    "continuation_from_update",
    "endpoint_update",
    "continuation_mode",
    "execution_action",
    "roles_json",
    "labels_mounted",
    "downstream_evaluations_during_training",
    "eligible_for_scientific_result",
    "eligible_for_selector_development",
    "reusable_for_science",
    "gpu_preference",
    "run_role",
    "attempt_index",
    "status",
    "plan_seal_sha256",
)

PARENT_INVENTORY_FIELDS = (
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
    "parent_checkpoint_size_bytes",
    "parent_representation_relative_path",
    "parent_representation_sha256",
    "parent_representation_size_bytes",
    "template_config_sha256",
    "resource_trace_sha256",
    "path_ids_json",
    "strict_state_complete",
    "optimizer_state_present",
    "target_networks_present",
    "projectors_present",
    "loss_weights_present",
    "rng_state_present",
    "augmentation_state_present",
    "parent_immutable",
    "status",
)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(value)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.casefold()
    )


def canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Restore CSV rows to the typed representation used by the plan seal."""

    value = dict(row)
    value.pop("plan_seal_sha256", None)
    for field in (
        "candidate_set_ordinal",
        "model_seed",
        "continuation_from_update",
        "endpoint_update",
        "downstream_evaluations_during_training",
        "gpu_preference",
        "attempt_index",
    ):
        if field in value:
            value[field] = int(value[field])
    for field in (
        "labels_mounted",
        "eligible_for_scientific_result",
        "eligible_for_selector_development",
        "reusable_for_science",
    ):
        if field in value:
            value[field] = _bool(value[field])
    return value


def h7_physical_run_id(
    stage: str,
    dataset: str,
    candidate_ordinal: int,
    model_seed: int,
) -> str:
    tokens = {
        DEVELOPMENT_MODEL_SEEDS[0]: "S00",
        DEVELOPMENT_MODEL_SEEDS[1]: "S01",
        DEVELOPMENT_MODEL_SEEDS[2]: "S02",
    }
    if int(model_seed) not in tokens:
        raise ValueError("unapproved H7 model seed")
    prefix = {"CANARY": "M4A-H7-CANARY", "R4": "M4A-H7-R4"}.get(stage)
    if prefix is None:
        raise ValueError(stage)
    if stage == "CANARY" and int(model_seed) != DEVELOPMENT_MODEL_SEEDS[2]:
        raise ValueError("H7 canary is frozen to S02")
    return f"{prefix}-{dataset.upper()}-C{int(candidate_ordinal):03d}-{tokens[int(model_seed)]}"


def _safe_artifact(root: Path, relative_value: str) -> Path:
    relative = safe_relative(relative_value)
    root_resolved = root.resolve(strict=True)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("H7 parent artifact symlink forbidden")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("H7 parent artifact escapes result root") from error
    if not resolved.is_file():
        raise ValueError("H7 parent artifact is not a file")
    return resolved


def _attempt_from_relative(relative_value: str) -> str:
    parts = safe_relative(relative_value).parts
    if "checkpoints" not in parts:
        raise ValueError("parent checkpoint path lacks checkpoints directory")
    index = parts.index("checkpoints")
    if index < 3 or not parts[index - 1].startswith("attempt_"):
        raise ValueError("parent checkpoint path lacks canonical attempt")
    return "/".join(parts[:index])


def collect_parent_states(
    shortlist_rows: Sequence[Mapping[str, Any]],
    path_audit_rows: Sequence[Mapping[str, Any]],
    *,
    h5_result_root: Path,
    h6_result_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind and validate all 63 exact update-2000 continuation sources."""

    h5_registry_path = h5_result_root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv"
    h6_registry_path = h6_result_root / "TC_LPMSO_H6_THIRD_SEED_CHECKPOINT_REGISTRY.csv"
    if sha256_file(h5_registry_path) != H5_R3_CHECKPOINT_REGISTRY_SHA256:
        raise ValueError("H5 R3 checkpoint registry hash mismatch")
    if sha256_file(h6_registry_path) != PARENT_ARTIFACT_SHA256[h6_registry_path.name]:
        raise ValueError("H6 third-seed checkpoint registry hash mismatch")

    shortlist = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): dict(row)
        for row in shortlist_rows
    }
    if len(shortlist) != sum(EXPECTED_SHORTLIST_COUNTS.values()):
        raise ValueError("H7 shortlist identity count mismatch")
    path_audit = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): dict(row)
        for row in path_audit_rows
    }
    if set(path_audit) != set(shortlist):
        raise ValueError("H7 path audit/shortlist identity mismatch")

    registries = (
        ("H5_RESULT_ROOT", h5_result_root, read_csv(h5_registry_path)),
        ("H6_RESULT_ROOT", h6_result_root, read_csv(h6_registry_path)),
    )
    selected: dict[tuple[str, str, int], tuple[str, Path, dict[str, str]]] = {}
    for source_name, source_root, rows in registries:
        for row in rows:
            key2 = (str(row["dataset"]), str(row["candidate_set_hash"]))
            seed = int(row["model_seed"])
            if key2 not in shortlist or seed not in DEVELOPMENT_MODEL_SEEDS:
                continue
            if source_name == "H5_RESULT_ROOT" and seed == DEVELOPMENT_MODEL_SEEDS[2]:
                continue
            if source_name == "H6_RESULT_ROOT" and seed != DEVELOPMENT_MODEL_SEEDS[2]:
                continue
            key = (*key2, seed)
            if key in selected:
                raise ValueError(f"duplicate H7 parent state: {key}")
            selected[key] = (source_name, source_root, row)
    expected = {
        (dataset, candidate_hash, seed)
        for dataset, candidate_hash in shortlist
        for seed in DEVELOPMENT_MODEL_SEEDS
    }
    if set(selected) != expected or len(selected) != EXPECTED_PARENT_STATES:
        raise RuntimeError("BLOCKED_R4_PARENT_STATE_INCOMPLETE: registry coverage")

    output: list[dict[str, Any]] = []
    for key in sorted(selected):
        dataset, candidate_hash, seed = key
        source_name, source_root, row = selected[key]
        parent = shortlist[(dataset, candidate_hash)]
        if (
            str(row["candidate_set_id"]) != str(parent["candidate_set_id"])
            or int(row["candidate_set_ordinal"]) != int(parent["candidate_set_ordinal"])
            or int(row["endpoint_update"]) != PARENT_UPDATE
            or str(row["status"]) != "PASS"
            or not _bool(row["strict_state_complete"])
        ):
            raise RuntimeError("BLOCKED_R4_PARENT_STATE_INCOMPLETE: identity/status")
        checkpoint_relative = str(row["checkpoint_relative_path"])
        representation_relative = str(row["representation_relative_path"])
        checkpoint = _safe_artifact(source_root, checkpoint_relative)
        representation = _safe_artifact(source_root, representation_relative)
        if (
            sha256_file(checkpoint) != str(row["checkpoint_sha256"])
            or checkpoint.stat().st_size != int(row["checkpoint_size_bytes"])
            or sha256_file(representation) != str(row["representation_sha256"])
            or representation.stat().st_size != int(row["representation_size_bytes"])
        ):
            raise RuntimeError("BLOCKED_R4_PARENT_STATE_INCOMPLETE: artifact hash")
        attempt_relative = _attempt_from_relative(checkpoint_relative)
        attempt = source_root / Path(*safe_relative(attempt_relative).parts)
        config = _safe_artifact(attempt, "RESOLVED_CONFIG.json")
        trace = _safe_artifact(attempt, "RESOURCE_TRACE.csv")
        config_payload = read_json(config)
        expected_paths = json.loads(str(path_audit[(dataset, candidate_hash)]["path_ids_json"]))
        if (
            config_payload.get("physical_run_id") != str(row["physical_run_id"])
            or config_payload.get("candidate_set_hash") != candidate_hash
            or int(config_payload.get("candidate_set_ordinal", -1)) != int(parent["candidate_set_ordinal"])
            or int(config_payload.get("model_seed", -1)) != seed
            or list(config_payload.get("path_ids", [])) != expected_paths
        ):
            raise RuntimeError("BLOCKED_R4_PARENT_STATE_INCOMPLETE: config lineage")
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=PARENT_UPDATE,
            expected_sha256=str(row["checkpoint_sha256"]),
        )
        required = {
            "strict_state_complete": "strict_state_complete",
            "optimizer_state_present": "optimizer_state_present",
            "target_networks_present": "target_encoders_present",
            "projectors_present": "projectors_present",
            "loss_weights_present": "loss_weight_parameters_present",
            "rng_state_present": "python_numpy_torch_rng_present",
            "augmentation_state_present": "augmentation_state_present",
        }
        if not all(bool(audit.get(source_field)) for source_field in required.values()):
            raise RuntimeError("BLOCKED_R4_PARENT_STATE_INCOMPLETE: strict state")
        output.append(
            {
                "dataset": dataset,
                "candidate_set_id": str(parent["candidate_set_id"]),
                "candidate_set_hash": candidate_hash,
                "candidate_set_ordinal": int(parent["candidate_set_ordinal"]),
                "model_seed": seed,
                "parent_source": source_name,
                "parent_physical_run_id": str(row["physical_run_id"]),
                "parent_attempt_relative_path": attempt_relative,
                "parent_checkpoint_relative_path": checkpoint_relative,
                "parent_checkpoint_sha256": str(row["checkpoint_sha256"]),
                "parent_checkpoint_size_bytes": checkpoint.stat().st_size,
                "parent_representation_relative_path": representation_relative,
                "parent_representation_sha256": str(row["representation_sha256"]),
                "parent_representation_size_bytes": representation.stat().st_size,
                "template_config_sha256": sha256_file(config),
                "resource_trace_sha256": sha256_file(trace),
                "path_ids_json": json.dumps(expected_paths, ensure_ascii=False),
                **{field: True for field in required},
                "parent_immutable": True,
                "status": "PASS",
            }
        )
    basis = {
        "parent_state_count": len(output),
        "acm_state_count": sum(row["dataset"] == "acm" for row in output),
        "freebase_state_count": sum(row["dataset"] == "freebase" for row in output),
        "model_seeds": list(DEVELOPMENT_MODEL_SEEDS),
        "parent_update": PARENT_UPDATE,
        "all_strict_state_complete": True,
        "parent_artifacts_immutable": True,
        "inventory_identity": canonical_json_hash(output),
        "h5_checkpoint_registry_sha256": H5_R3_CHECKPOINT_REGISTRY_SHA256,
        "h6_checkpoint_registry_sha256": PARENT_ARTIFACT_SHA256[h6_registry_path.name],
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    return output, envelope(
        "metachorus.v53.m4a-h7.r4-continuation-audit.v1",
        "PASS",
        basis,
    )


def _gpu_preference(dataset: str, candidate_hash: str, ordinal: int) -> int:
    return int(canonical_json_hash({"dataset": dataset, "hash": candidate_hash, "ordinal": ordinal})[-1], 16) % 2


def _plan_row(
    parent: Mapping[str, Any],
    path_row: Mapping[str, Any],
    *,
    stage: str,
    endpoint: int,
    canary: bool,
) -> dict[str, Any]:
    seed = int(parent["model_seed"])
    return {
        "stage": stage,
        "physical_run_id": h7_physical_run_id(
            stage,
            str(parent["dataset"]),
            int(parent["candidate_set_ordinal"]),
            seed,
        ),
        "dataset": str(parent["dataset"]),
        "candidate_set_id": str(parent["candidate_set_id"]),
        "candidate_set_hash": str(parent["candidate_set_hash"]),
        "candidate_set_ordinal": int(parent["candidate_set_ordinal"]),
        "model_seed": seed,
        "parent_source": str(parent["parent_source"]),
        "parent_physical_run_id": str(parent["parent_physical_run_id"]),
        "parent_attempt_relative_path": str(parent["parent_attempt_relative_path"]),
        "parent_checkpoint_relative_path": str(parent["parent_checkpoint_relative_path"]),
        "parent_checkpoint_sha256": str(parent["parent_checkpoint_sha256"]),
        "parent_representation_relative_path": str(parent["parent_representation_relative_path"]),
        "parent_representation_sha256": str(parent["parent_representation_sha256"]),
        "template_config_sha256": str(parent["template_config_sha256"]),
        "path_ids_json": str(path_row["path_ids_json"]),
        "continuation_from_update": PARENT_UPDATE,
        "endpoint_update": endpoint,
        "continuation_mode": "STRICT_CONTINUATION_STATE_COPY" if canary else "STRICT_CONTINUATION",
        "execution_action": "RUN_ENGINEERING_CANARY_FROM_STATE_COPY" if canary else "CONTINUE_ROBUST_SHORTLIST_TO_R4",
        "roles_json": str(path_row.get("components_json", "[]")),
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "eligible_for_scientific_result": not canary,
        "eligible_for_selector_development": not canary,
        "reusable_for_science": not canary,
        "gpu_preference": _gpu_preference(
            str(parent["dataset"]),
            str(parent["candidate_set_hash"]),
            int(parent["candidate_set_ordinal"]),
        ),
        "run_role": "M4A_H7_ENGINEERING_CANARY" if canary else "M4A_H7_R4_ROBUST_SHORTLIST",
        "attempt_index": 0,
        "status": "PENDING_GPU",
    }


def build_canary_plan(
    shortlist_rows: Sequence[Mapping[str, Any]],
    path_audit_rows: Sequence[Mapping[str, Any]],
    parent_inventory_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shortlist = [dict(row) for row in shortlist_rows]
    acm_manual_hash = CONTROLS["acm"]["manual"][1]
    freebase_controls = {value[1] for value in CONTROLS["freebase"].values()}
    freebase_hashes = sorted(
        str(row["candidate_set_hash"])
        for row in shortlist
        if str(row["dataset"]) == "freebase"
        and str(row["candidate_set_hash"]) not in freebase_controls
    )
    if not freebase_hashes:
        raise ValueError("H7 has no eligible Freebase engineering canary")
    identities = {("acm", acm_manual_hash), ("freebase", freebase_hashes[0])}
    path_map = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): dict(row)
        for row in path_audit_rows
    }
    parent_map = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"])): dict(row)
        for row in parent_inventory_rows
    }
    rows: list[dict[str, Any]] = []
    for dataset, candidate_hash in sorted(identities):
        parent = parent_map.get((dataset, candidate_hash, DEVELOPMENT_MODEL_SEEDS[2]))
        path_row = path_map.get((dataset, candidate_hash))
        if parent is None or path_row is None:
            raise ValueError("H7 canary parent/path identity missing")
        rows.append(
            _plan_row(
                parent,
                path_row,
                stage="CANARY",
                endpoint=CANARY_ENDPOINT,
                canary=True,
            )
        )
    sealed, seal = seal_rows(rows, stage="CANARY")
    validate_canary_plan_rows(sealed)
    return sealed, seal


def validate_canary_plan_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 2 or {str(row["dataset"]) for row in rows} != {"acm", "freebase"}:
        raise ValueError("H7 canary plan must contain one run per dataset")
    for row in rows:
        if (
            str(row["stage"]) != "CANARY"
            or int(row["model_seed"]) != DEVELOPMENT_MODEL_SEEDS[2]
            or int(row["continuation_from_update"]) != PARENT_UPDATE
            or int(row["endpoint_update"]) != CANARY_ENDPOINT
            or str(row["continuation_mode"]) != "STRICT_CONTINUATION_STATE_COPY"
            or str(row["run_role"]) != "M4A_H7_ENGINEERING_CANARY"
            or _bool(row["eligible_for_scientific_result"])
            or _bool(row["eligible_for_selector_development"])
            or _bool(row["reusable_for_science"])
            or _bool(row["labels_mounted"])
            or int(row["downstream_evaluations_during_training"]) != 0
            or int(row["attempt_index"]) != 0
            or str(row["status"]) != "PENDING_GPU"
        ):
            raise ValueError("invalid sealed H7 canary plan row")


def validate_canary_gate(result_root: Path) -> dict[str, Any]:
    decision = read_json(result_root / "M4A_H7_CANARY_GATE_DECISION.json")
    basis = decision.get("basis", {})
    if (
        decision.get("schema_version") != "metachorus.v53.m4a-h7.canary-gate-decision.v1"
        or decision.get("status") != "H7_CANARY_PASS"
        or decision.get("identity") != canonical_json_hash(basis)
        or basis.get("fanout_authorized") is not True
        or basis.get("fanout_started_before_decision") is not False
        or basis.get("scientific_metrics_observed") is not False
        or basis.get("labels_mounted") is not False
        or basis.get("h5_locked_report_access") is not False
        or basis.get("s_eval_started") is not False
    ):
        raise ValueError("invalid H7 canary gate")
    return decision


def build_r4_plan(
    *,
    result_root: Path,
    path_audit_rows: Sequence[Mapping[str, Any]],
    parent_inventory_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decision = validate_canary_gate(result_root)
    shortlist_seal = read_json(result_root / "TC_LPMSO_H7_ROBUST_SHORTLIST_SEAL.json")
    shortlist_basis = shortlist_seal.get("basis", {})
    if (
        shortlist_seal.get("schema_version")
        != "metachorus.v53.m4a-h7.robust-shortlist-seal.v1"
        or shortlist_seal.get("status") != "SEALED_BEFORE_R4"
        or shortlist_seal.get("identity") != canonical_json_hash(shortlist_basis)
        or shortlist_basis.get("r4_started") is not False
        or shortlist_basis.get("truth_read") is not False
        or shortlist_basis.get("h5_locked_report_access") is not False
        or shortlist_basis.get("s_eval_access") is not False
    ):
        raise ValueError("invalid H7 shortlist seal")
    path_map = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): dict(row)
        for row in path_audit_rows
    }
    rows = [
        _plan_row(
            parent,
            path_map[(str(parent["dataset"]), str(parent["candidate_set_hash"]))],
            stage="R4",
            endpoint=R4_ENDPOINT,
            canary=False,
        )
        for parent in parent_inventory_rows
    ]
    if len(rows) != EXPECTED_R4_RUNS:
        raise ValueError("H7 R4 plan must contain exactly 63 identities")
    sealed, seal = seal_rows(rows, stage="R4")
    basis = dict(seal["basis"])
    basis.update(
        {
            "canary_gate_identity": decision["identity"],
            "shortlist_seal_identity": shortlist_seal["identity"],
            "parent_state_inventory_sha256": sha256_file(
                result_root / "TC_LPMSO_H7_R4_PARENT_STATE_INVENTORY.csv"
            ),
        }
    )
    identity = canonical_json_hash(basis)
    sealed = [{**{key: row[key] for key in row if key != "plan_seal_sha256"}, "plan_seal_sha256": identity} for row in sealed]
    seal = envelope(
        "metachorus.v53.m4a-h7.execution-plan-seal.v1",
        "SEALED_BEFORE_GPU_TRAINING",
        basis,
    )
    validate_r4_plan_rows(sealed)
    return sealed, seal


def validate_r4_plan_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_R4_RUNS or len({str(row["physical_run_id"]) for row in rows}) != len(rows):
        raise ValueError("H7 R4 plan identity count mismatch")
    candidates: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["candidate_set_hash"]))
        candidates.setdefault(key, set()).add(int(row["model_seed"]))
        if (
            str(row["stage"]) != "R4"
            or str(row["physical_run_id"])
            != h7_physical_run_id(
                "R4",
                str(row["dataset"]),
                int(row["candidate_set_ordinal"]),
                int(row["model_seed"]),
            )
            or int(row["continuation_from_update"]) != PARENT_UPDATE
            or int(row["endpoint_update"]) != R4_ENDPOINT
            or str(row["continuation_mode"]) != "STRICT_CONTINUATION"
            or str(row["run_role"]) != "M4A_H7_R4_ROBUST_SHORTLIST"
            or not _bool(row["eligible_for_scientific_result"])
            or not _bool(row["eligible_for_selector_development"])
            or not _bool(row["reusable_for_science"])
            or _bool(row["labels_mounted"])
            or int(row["downstream_evaluations_during_training"]) != 0
            or not _valid_sha256(str(row["parent_checkpoint_sha256"]))
            or not _valid_sha256(str(row["parent_representation_sha256"]))
            or not _valid_sha256(str(row["template_config_sha256"]))
            or int(row["attempt_index"]) != 0
            or str(row["status"]) != "PENDING_GPU"
        ):
            raise ValueError("invalid H7 R4 plan row")
    if any(seeds != set(DEVELOPMENT_MODEL_SEEDS) for seeds in candidates.values()):
        raise ValueError("each H7 shortlist candidate requires all three seeds")
