"""Frozen H8 canary and 50-run S_eval planning."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file

from .common import read_csv, seal_rows, validate_envelope
from .constants import (
    CANARY_ENDPOINT,
    EXPECTED_ACM_RUNS,
    EXPECTED_CANARY_RUNS,
    EXPECTED_FREEBASE_RUNS,
    EXPECTED_SCIENCE_RUNS,
    FROZEN_PATH_SETS,
    FROZEN_PROJECTION_IDENTITIES,
    FREEBASE_H7_SELECTED_PROJECTION_HASHES,
    S_EVAL_MODEL_SEEDS,
    SCIENCE_ENDPOINT,
)


PLAN_FIELDS = (
    "stage",
    "physical_run_id",
    "dataset",
    "role",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "model_seed",
    "template_config_relative_path",
    "template_config_sha256",
    "path_ids_json",
    "projection_identity",
    "projection_hashes_json",
    "continuation_from_update",
    "endpoint_update",
    "continuation_mode",
    "execution_action",
    "trainer",
    "view_pipeline",
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

TEMPLATE_FIELDS = (
    "dataset",
    "role",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "path_ids_json",
    "projection_identity",
    "projection_hashes_json",
    "template_config_relative_path",
    "template_config_sha256",
    "source_h7_physical_run_id",
    "source_h7_marker_identity",
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
    raise ValueError(f"invalid boolean: {value!r}")


def _valid_sha256(value: Any) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None


def canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
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
        value[field] = int(value[field])
    for field in (
        "labels_mounted",
        "eligible_for_scientific_result",
        "eligible_for_selector_development",
        "reusable_for_science",
    ):
        value[field] = _bool(value[field])
    return value


def future_seed_token(seed: int) -> str:
    try:
        return f"F{S_EVAL_MODEL_SEEDS.index(int(seed)):02d}"
    except ValueError as error:
        raise ValueError("unapproved H8 S_eval model seed") from error


def h8_physical_run_id(
    stage: str,
    dataset: str,
    candidate_ordinal: int,
    model_seed: int,
) -> str:
    token = future_seed_token(model_seed)
    prefix = {
        "CANARY": "M4A-H8-CANARY",
        "S_EVAL": "M4A-H8-S-EVAL",
    }.get(stage)
    if prefix is None:
        raise ValueError(f"unapproved H8 stage: {stage}")
    if stage == "CANARY" and token != "F00":
        raise ValueError("H8 canaries are frozen to future seed F00")
    if dataset not in {"acm", "freebase"}:
        raise ValueError(dataset)
    return f"{prefix}-{dataset.upper()}-C{int(candidate_ordinal):03d}-{token}"


def _template_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    frozen_by_key = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): row
        for row in FROZEN_PATH_SETS
    }
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        value = dict(row)
        key = (str(value["dataset"]), str(value["candidate_set_hash"]))
        if key in output:
            raise ValueError("duplicate H8 template identity")
        frozen = frozen_by_key.get(key)
        path_ids = json.loads(str(value["path_ids_json"]))
        projection_hashes = json.loads(str(value["projection_hashes_json"]))
        expected_projection_hashes = (
            list(FREEBASE_H7_SELECTED_PROJECTION_HASHES)
            if frozen is not None
            and frozen["dataset"] == "freebase"
            and frozen["role"] == "H7_SELECTED"
            else []
        )
        if (
            frozen is None
            or
            not _valid_sha256(value.get("template_config_sha256"))
            or not _valid_sha256(value.get("projection_identity"))
            or str(value.get("status")) != "PASS"
            or not isinstance(path_ids, list)
            or not isinstance(projection_hashes, list)
            or str(value.get("role")) != str(frozen["role"])
            or str(value.get("candidate_set_id")) != str(frozen["candidate_set_id"])
            or int(value.get("candidate_set_ordinal", -1))
            != int(frozen["candidate_set_ordinal"])
            or tuple(path_ids) != tuple(frozen["path_ids"])
            or str(value.get("projection_identity"))
            != FROZEN_PROJECTION_IDENTITIES[str(frozen["dataset"])]
            or projection_hashes != expected_projection_hashes
        ):
            raise ValueError("invalid H8 template row")
        output[key] = value
    expected = {
        (str(row["dataset"]), str(row["candidate_set_hash"]))
        for row in FROZEN_PATH_SETS
    }
    if set(output) != expected:
        raise ValueError("H8 template inventory does not bind all five frozen sets")
    return output


def _plan_row(
    template: Mapping[str, Any],
    *,
    stage: str,
    model_seed: int,
    canary: bool,
) -> dict[str, Any]:
    dataset = str(template["dataset"])
    ordinal = int(template["candidate_set_ordinal"])
    return {
        "stage": stage,
        "physical_run_id": h8_physical_run_id(stage, dataset, ordinal, model_seed),
        "dataset": dataset,
        "role": str(template["role"]),
        "candidate_set_id": str(template["candidate_set_id"]),
        "candidate_set_hash": str(template["candidate_set_hash"]),
        "candidate_set_ordinal": ordinal,
        "model_seed": int(model_seed),
        "template_config_relative_path": str(template["template_config_relative_path"]),
        "template_config_sha256": str(template["template_config_sha256"]),
        "path_ids_json": str(template["path_ids_json"]),
        "projection_identity": str(template["projection_identity"]),
        "projection_hashes_json": str(template["projection_hashes_json"]),
        "continuation_from_update": 0,
        "endpoint_update": CANARY_ENDPOINT if canary else SCIENCE_ENDPOINT,
        "continuation_mode": "FROM_ZERO_ENGINEERING_CANARY" if canary else "FROM_ZERO_LOCKED_CONFIRMATION",
        "execution_action": "RUN_ENGINEERING_CANARY" if canary else "RUN_FROZEN_S_EVAL_CONFIRMATION",
        "trainer": "O0-CodeFaithful",
        "view_pipeline": "S0",
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "eligible_for_scientific_result": not canary,
        "eligible_for_selector_development": False,
        "reusable_for_science": not canary,
        "gpu_preference": S_EVAL_MODEL_SEEDS.index(int(model_seed)) % 2,
        "run_role": "M4A_H8_ENGINEERING_CANARY" if canary else "M4A_H8_S_EVAL_CONFIRMATION",
        "attempt_index": 0,
        "status": "PENDING_GPU",
    }


def build_canary_plan(
    template_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    templates = _template_map(template_rows)
    frozen = {
        (str(row["dataset"]), str(row["role"])): row for row in FROZEN_PATH_SETS
    }
    selected = (
        frozen[("acm", "CURRENT_AUTOMATIC")],
        frozen[("freebase", "H7_SELECTED")],
    )
    rows = [
        _plan_row(
            templates[(str(row["dataset"]), str(row["candidate_set_hash"]))],
            stage="CANARY",
            model_seed=S_EVAL_MODEL_SEEDS[0],
            canary=True,
        )
        for row in selected
    ]
    sealed, seal = seal_rows(rows, stage="CANARY")
    validate_canary_plan_rows(sealed)
    return sealed, seal


def build_science_draft(template_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    templates = _template_map(template_rows)
    rows: list[dict[str, Any]] = []
    for frozen in FROZEN_PATH_SETS:
        template = templates[(str(frozen["dataset"]), str(frozen["candidate_set_hash"]))]
        for seed in S_EVAL_MODEL_SEEDS:
            rows.append(
                _plan_row(template, stage="S_EVAL", model_seed=seed, canary=False)
            )
    validate_science_plan_rows(rows, require_seal=False)
    return rows


def seal_science_plan(
    rows: Sequence[Mapping[str, Any]], *, canary_gate_identity: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not _valid_sha256(canary_gate_identity):
        raise ValueError("H8 canary gate identity must be SHA-256")
    sealed, seal = seal_rows(
        rows,
        stage="S_EVAL",
        extra_basis={
            "canary_gate_identity": canary_gate_identity,
            "formal_fanout_authorized": True,
            "future_seed_count": len(S_EVAL_MODEL_SEEDS),
            "frozen_set_count": len(FROZEN_PATH_SETS),
        },
    )
    validate_science_plan_rows(sealed, require_seal=True)
    return sealed, seal


def validate_canary_plan_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_CANARY_RUNS or {str(row["dataset"]) for row in rows} != {
        "acm",
        "freebase",
    }:
        raise ValueError("H8 canary plan must contain exactly two datasets")
    expected_hashes = {
        next(
            str(item["candidate_set_hash"])
            for item in FROZEN_PATH_SETS
            if item["dataset"] == "acm" and item["role"] == "CURRENT_AUTOMATIC"
        ),
        next(
            str(item["candidate_set_hash"])
            for item in FROZEN_PATH_SETS
            if item["dataset"] == "freebase" and item["role"] == "H7_SELECTED"
        ),
    }
    if {str(row["candidate_set_hash"]) for row in rows} != expected_hashes:
        raise ValueError("H8 canary identities drifted")
    frozen_by_key = {
        (str(item["dataset"]), str(item["candidate_set_hash"])): item
        for item in FROZEN_PATH_SETS
        if (item["dataset"], item["role"])
        in {("acm", "CURRENT_AUTOMATIC"), ("freebase", "H7_SELECTED")}
    }
    for row in rows:
        frozen = frozen_by_key.get(
            (str(row["dataset"]), str(row["candidate_set_hash"]))
        )
        if (
            frozen is None
            or str(row["stage"]) != "CANARY"
            or str(row["physical_run_id"])
            != h8_physical_run_id(
                "CANARY",
                str(row["dataset"]),
                int(row["candidate_set_ordinal"]),
                int(row["model_seed"]),
            )
            or int(row["model_seed"]) != S_EVAL_MODEL_SEEDS[0]
            or int(row["continuation_from_update"]) != 0
            or int(row["endpoint_update"]) != CANARY_ENDPOINT
            or str(row["continuation_mode"]) != "FROM_ZERO_ENGINEERING_CANARY"
            or str(row["run_role"]) != "M4A_H8_ENGINEERING_CANARY"
            or _bool(row["eligible_for_scientific_result"])
            or _bool(row["eligible_for_selector_development"])
            or _bool(row["reusable_for_science"])
            or _bool(row["labels_mounted"])
            or int(row["downstream_evaluations_during_training"]) != 0
            or str(row["trainer"]) != "O0-CodeFaithful"
            or str(row["view_pipeline"]) != "S0"
            or int(row["attempt_index"]) != 0
            or str(row["status"]) != "PENDING_GPU"
            or not _valid_sha256(row["template_config_sha256"])
            or str(row["role"]) != str(frozen["role"])
            or str(row["candidate_set_id"]) != str(frozen["candidate_set_id"])
            or int(row["candidate_set_ordinal"]) != int(frozen["candidate_set_ordinal"])
            or tuple(json.loads(str(row["path_ids_json"]))) != tuple(frozen["path_ids"])
            or str(row["projection_identity"])
            != FROZEN_PROJECTION_IDENTITIES[str(frozen["dataset"])]
            or json.loads(str(row["projection_hashes_json"]))
            != (
                list(FREEBASE_H7_SELECTED_PROJECTION_HASHES)
                if frozen["dataset"] == "freebase"
                else []
            )
        ):
            raise ValueError("invalid sealed H8 canary plan row")


def validate_science_plan_rows(
    rows: Sequence[Mapping[str, Any]], *, require_seal: bool = True
) -> None:
    if len(rows) != EXPECTED_SCIENCE_RUNS:
        raise ValueError("H8 S_eval plan must contain exactly 50 runs")
    if len({str(row["physical_run_id"]) for row in rows}) != EXPECTED_SCIENCE_RUNS:
        raise ValueError("H8 S_eval plan has duplicate physical identities")
    counts = {
        dataset: sum(str(row["dataset"]) == dataset for row in rows)
        for dataset in ("acm", "freebase")
    }
    if counts != {"acm": EXPECTED_ACM_RUNS, "freebase": EXPECTED_FREEBASE_RUNS}:
        raise ValueError("H8 S_eval dataset counts drifted")
    by_set: dict[tuple[str, str], set[int]] = {}
    frozen_by_key = {
        (str(item["dataset"]), str(item["candidate_set_hash"])): item
        for item in FROZEN_PATH_SETS
    }
    for row in rows:
        key = (str(row["dataset"]), str(row["candidate_set_hash"]))
        frozen = frozen_by_key.get(key)
        by_set.setdefault(key, set()).add(int(row["model_seed"]))
        if (
            frozen is None
            or
            str(row["stage"]) != "S_EVAL"
            or str(row["physical_run_id"])
            != h8_physical_run_id(
                "S_EVAL",
                str(row["dataset"]),
                int(row["candidate_set_ordinal"]),
                int(row["model_seed"]),
            )
            or int(row["continuation_from_update"]) != 0
            or int(row["endpoint_update"]) != SCIENCE_ENDPOINT
            or str(row["continuation_mode"]) != "FROM_ZERO_LOCKED_CONFIRMATION"
            or str(row["run_role"]) != "M4A_H8_S_EVAL_CONFIRMATION"
            or not _bool(row["eligible_for_scientific_result"])
            or _bool(row["eligible_for_selector_development"])
            or not _bool(row["reusable_for_science"])
            or _bool(row["labels_mounted"])
            or int(row["downstream_evaluations_during_training"]) != 0
            or str(row["trainer"]) != "O0-CodeFaithful"
            or str(row["view_pipeline"]) != "S0"
            or int(row["attempt_index"]) != 0
            or str(row["status"]) != "PENDING_GPU"
            or not _valid_sha256(row["template_config_sha256"])
            or str(row["role"]) != str(frozen["role"])
            or str(row["candidate_set_id"]) != str(frozen["candidate_set_id"])
            or int(row["candidate_set_ordinal"]) != int(frozen["candidate_set_ordinal"])
            or tuple(json.loads(str(row["path_ids_json"]))) != tuple(frozen["path_ids"])
            or str(row["projection_identity"])
            != FROZEN_PROJECTION_IDENTITIES[str(frozen["dataset"])]
            or json.loads(str(row["projection_hashes_json"]))
            != (
                list(FREEBASE_H7_SELECTED_PROJECTION_HASHES)
                if frozen["dataset"] == "freebase" and frozen["role"] == "H7_SELECTED"
                else []
            )
            or (require_seal and not _valid_sha256(row.get("plan_seal_sha256")))
        ):
            raise ValueError("invalid sealed H8 S_eval plan row")
    if set(by_set) != set(frozen_by_key) or any(
        seeds != set(S_EVAL_MODEL_SEEDS) for seeds in by_set.values()
    ):
        raise ValueError("every H8 frozen set requires all ten future seeds")


def validate_plan_seal(
    plan_path: Path, seal_path: Path, *, stage: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(plan_path)
    seal = validate_envelope(
        read_json(seal_path),
        schema_version="metachorus.v53.m4a-h8.execution-plan-seal.v1",
        status="SEALED_BEFORE_GPU_TRAINING",
    )
    canonical = [canonical_plan_row(row) for row in rows]
    basis = seal["basis"]
    if (
        basis.get("stage") != stage
        or int(basis.get("row_count", -1)) != len(rows)
        or basis.get("rows_identity") != canonical_json_hash(canonical)
        or basis.get("labels_mounted") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
        or basis.get("run_level_parallel_only") is not True
        or int(basis.get("maximum_gpu_count", -1)) != 2
        or basis.get("multi_gpu_single_run") is not False
        or basis.get("h5_locked_report_access") is not False
        or basis.get("locked_evaluation_started") is not False
        or any(str(row.get("plan_seal_sha256")) != seal["identity"] for row in rows)
    ):
        raise ValueError("H8 plan seal mismatch")
    if stage == "CANARY":
        validate_canary_plan_rows(rows)
    elif stage == "S_EVAL":
        validate_science_plan_rows(rows)
    else:
        raise ValueError(stage)
    if sha256_file(plan_path) != str(basis.get("plan_sha256", sha256_file(plan_path))):
        raise ValueError("H8 plan SHA-256 mismatch")
    return rows, seal
