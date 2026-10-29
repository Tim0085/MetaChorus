"""Label-free H8 from-zero canary and S_eval training execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from metachorus_v53.audit.common import (
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
)
from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession
from metachorus_v53.m4a_h4.continuation import checkpoint_state_audit

from .common import append_jsonl_once, safe_relative
from .constants import (
    CANARY_CHECKPOINT_SCHEDULE,
    CANARY_ENDPOINT,
    ROLLING_INTERMEDIATE_LIMIT,
    S_EVAL_MODEL_SEEDS,
    SCIENCE_CHECKPOINT_SCHEDULE,
    SCIENCE_ENDPOINT,
)
from .planning import future_seed_token, h8_physical_run_id


def _git(repo_root: Path, expression: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", expression],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _json_list(value: Any) -> list[str]:
    observed = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(observed, list) or not observed or not all(
        isinstance(item, str) and item for item in observed
    ):
        raise ValueError("H8 path_ids_json must be a non-empty string list")
    return list(observed)


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid H8 boolean value: {value!r}")


def _runtime_domain() -> dict[str, str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("H8 run process must expose exactly one GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    command = ["nvidia-smi"]
    if visible and "," not in visible:
        command.extend(["-i", visible])
    command.extend(["--query-gpu=uuid", "--format=csv,noheader"])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    uuids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(uuids) != 1 or not uuids[0].startswith("GPU-"):
        raise RuntimeError("H8 could not bind the run to exactly one GPU UUID")
    basis = {
        "gpu_uuid": uuids[0],
        "gpu_name": torch.cuda.get_device_name(0),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "visible_gpu_count": torch.cuda.device_count(),
    }
    return {
        "gpu_uuid": uuids[0],
        "runtime_domain_identity": canonical_json_hash(basis),
    }


def build_h8_config(
    template: Mapping[str, Any],
    *,
    repo_root: Path,
    dataset_cache_root: Path,
    output_root: Path,
    plan_row: Mapping[str, Any],
) -> O0RunConfig:
    stage = str(plan_row["stage"])
    canary = stage == "CANARY"
    if stage not in {"CANARY", "S_EVAL"}:
        raise ValueError(stage)
    seed = int(plan_row["model_seed"])
    if seed not in S_EVAL_MODEL_SEEDS:
        raise ValueError("H8 configuration received an unapproved future seed")
    if (
        str(template.get("candidate_set_hash")) != str(plan_row["candidate_set_hash"])
        or str(template.get("path_set_id")) != str(plan_row["candidate_set_id"])
        or int(template.get("candidate_set_ordinal", -1))
        != int(plan_row["candidate_set_ordinal"])
        or tuple(template.get("path_ids", [])) != tuple(_json_list(plan_row["path_ids_json"]))
        or str(template.get("projection_identity")) != str(plan_row["projection_identity"])
    ):
        raise ValueError("H8 template and sealed plan identity mismatch")
    value = dict(template)
    value.update(
        {
            "dataset": str(plan_row["dataset"]),
            "cache_root": str(dataset_cache_root.resolve()),
            "output_root": str(output_root.resolve()),
            "physical_run_id": str(plan_row["physical_run_id"]),
            "path_set_role": "Candidate",
            "path_set_id": str(plan_row["candidate_set_id"]),
            "candidate_set_hash": str(plan_row["candidate_set_hash"]),
            "candidate_set_ordinal": int(plan_row["candidate_set_ordinal"]),
            "path_ids": _json_list(plan_row["path_ids_json"]),
            "model_seed": seed,
            "max_updates": CANARY_ENDPOINT if canary else SCIENCE_ENDPOINT,
            "checkpoint_schedule": list(
                CANARY_CHECKPOINT_SCHEDULE if canary else SCIENCE_CHECKPOINT_SCHEDULE
            ),
            "capture_mode": "on",
            "run_role": (
                "M4A_H8_ENGINEERING_CANARY"
                if canary
                else "M4A_H8_S_EVAL_CONFIRMATION"
            ),
            "eligible_for_scientific_result": not canary,
            "eligible_for_selector_development": False,
            "augmentation_namespace": canonical_json_hash(
                {
                    "stage": stage,
                    "physical_run_id": str(plan_row["physical_run_id"]),
                    "future_seed_token": future_seed_token(seed),
                    "method_frozen": True,
                }
            ),
            "source_commit": _git(repo_root, "HEAD"),
            "source_tree": _git(repo_root, "HEAD^{tree}"),
            "plan_seal_hash": str(plan_row["plan_seal_sha256"]),
        }
    )
    config = O0RunConfig.from_mapping(value)
    config.validate()
    return config


def _artifact_paths(run_root: Path, update: int) -> list[Path]:
    return [
        run_root / "checkpoints" / f"update_{update:06d}.pt",
        run_root / "representations" / f"update_{update:06d}.npy",
        run_root / "representations" / "view_specific" / f"update_{update:06d}",
    ]


def _retire_intermediate(run_root: Path, *, retired_update: int, retained_update: int) -> None:
    retired = _artifact_paths(run_root, retired_update)
    hashes: list[dict[str, Any]] = []
    for path in retired:
        if path.is_file():
            hashes.append(
                {
                    "relative_path": path.relative_to(run_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            path.unlink()
        elif path.is_dir():
            members = sorted(member for member in path.rglob("*") if member.is_file())
            hashes.extend(
                {
                    "relative_path": member.relative_to(run_root).as_posix(),
                    "size_bytes": member.stat().st_size,
                    "sha256": sha256_file(member),
                }
                for member in members
            )
            shutil.rmtree(path)
    if not hashes:
        raise RuntimeError("H8 rolling retention expected a complete retired capture")
    append_jsonl_once(
        run_root / "H8_ROLLING_CHECKPOINT_RETENTION.jsonl",
        {
            "event_id": f"RETIRE_UPDATE_{retired_update:06d}",
            "status": "RETIRED_AFTER_NEXT_DURABLE_BOUNDARY",
            "retired_update": int(retired_update),
            "retained_intermediate_update": int(retained_update),
            "retired_members": hashes,
            "retired_member_count": len(hashes),
            "labels_read": False,
            "downstream_evaluation_executed": False,
        },
    )


def validate_rolling_retention(run_root: Path, *, endpoint: int) -> dict[str, Any]:
    checkpoint_updates = sorted(
        int(path.stem.rsplit("_", 1)[-1])
        for path in (run_root / "checkpoints").glob("update_*.pt")
    )
    representation_updates = sorted(
        int(path.stem.rsplit("_", 1)[-1])
        for path in (run_root / "representations").glob("update_*.npy")
    )
    if endpoint == CANARY_ENDPOINT:
        expected = [CANARY_ENDPOINT]
        retired = []
    else:
        expected = [SCIENCE_CHECKPOINT_SCHEDULE[-2], SCIENCE_ENDPOINT]
        retired = [SCIENCE_CHECKPOINT_SCHEDULE[0]]
    if checkpoint_updates != expected or representation_updates != expected:
        raise RuntimeError("H8 rolling checkpoint retention invariant failed")
    if len([value for value in checkpoint_updates if value < endpoint]) > ROLLING_INTERMEDIATE_LIMIT:
        raise RuntimeError("H8 retained more than one intermediate checkpoint")
    return {
        "retained_checkpoint_updates": checkpoint_updates,
        "retained_representation_updates": representation_updates,
        "retired_checkpoint_updates": retired,
        "intermediate_checkpoint_count": sum(value < endpoint for value in checkpoint_updates),
        "rolling_intermediate_limit": ROLLING_INTERMEDIATE_LIMIT,
        "final_checkpoint_retained": endpoint in checkpoint_updates,
        "status": "PASS",
    }


def _marker_name(stage: str) -> str:
    return (
        "H8_CANARY_000020_COMPLETE.json"
        if stage == "CANARY"
        else "H8_STAGE_003000_COMPLETE.json"
    )


def _canary_engineering_audit(
    *,
    trained_snapshot: Mapping[str, Any],
    run_root: Path,
    result_root: Path,
    repo_root: Path,
    enforce_target_gpu: bool,
    template_path: Path,
    plan_row: Mapping[str, Any],
    config: O0RunConfig,
) -> dict[str, Any]:
    endpoint = CANARY_ENDPOINT
    checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    sidecar_path = run_root / "SIDECAR_CAPTURE_RECEIPT.json"
    sidecar = read_json(sidecar_path)
    checkpoints = list(sidecar.get("checkpoints", []))
    representations = list(sidecar.get("representations", []))
    checkpoint_io = list(sidecar.get("checkpoint_io", []))
    if (
        sidecar.get("schema_version") != "metachorus.v53.o0-sidecar-capture.v1"
        or list(sidecar.get("checkpoint_schedule", [])) != [endpoint]
        or len(checkpoints) != 1
        or len(representations) != 1
        or len(checkpoint_io) != 1
        or int(checkpoints[0].get("update", -1)) != endpoint
        or int(representations[0].get("update", -1)) != endpoint
        or int(checkpoint_io[0].get("update", -1)) != endpoint
        or checkpoints[0].get("path") != checkpoint.name
        or checkpoints[0].get("sha256") != sha256_file(checkpoint)
        or representations[0].get("path") != representation.name
        or representations[0].get("sha256") != sha256_file(representation)
        or sidecar.get("labels_read") is not False
        or sidecar.get("downstream_evaluation_executed") is not False
    ):
        raise RuntimeError("H8 canary checkpoint write/capture audit failed")
    strict_state = checkpoint_state_audit(
        checkpoint,
        expected_update=endpoint,
        expected_sha256=sha256_file(checkpoint),
    )
    if strict_state.get("strict_state_complete") is not True:
        raise RuntimeError("H8 canary checkpoint is not strict-state complete")
    restored_session = O0TrainingSession(
        config,
        repo_root=repo_root,
        enforce_target_gpu=enforce_target_gpu,
    )
    loaded = restored_session.resume(checkpoint)
    restored_snapshot = restored_session.state_snapshot()
    state_hash_fields = (
        "model_state_sha256",
        "optimizer_state_sha256",
        "target_state_sha256",
    )
    optimizer_state = loaded.get("payload", {}).get("optimizer_state", {})
    state_rows = optimizer_state.get("state", {}) if isinstance(optimizer_state, Mapping) else {}
    step_values: list[int] = []
    if isinstance(state_rows, Mapping):
        for state in state_rows.values():
            if not isinstance(state, Mapping) or "step" not in state:
                raise RuntimeError("H8 canary optimizer state lacks step counters")
            step = state["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    raise RuntimeError("H8 canary optimizer step counter is not scalar")
                step = step.detach().cpu().item()
            step_values.append(int(step))
    if (
        int(loaded.get("payload", {}).get("exact_update", -1)) != endpoint
        or loaded.get("rng_state_roundtrip") is not True
        or restored_session.current_update != endpoint
        or any(
            trained_snapshot[field] != restored_snapshot[field]
            for field in state_hash_fields
        )
        or trained_snapshot["state_dict_keys"] != restored_snapshot["state_dict_keys"]
        or not step_values
        or set(step_values) != {endpoint}
    ):
        raise RuntimeError("H8 canary checkpoint read/restore audit failed")
    result_resolved = result_root.resolve(strict=True)
    try:
        template_relative = template_path.resolve(strict=True).relative_to(result_resolved)
    except ValueError as error:
        raise RuntimeError("H8 canary template escaped the authorized result root") from error
    expected_paths = _json_list(plan_row["path_ids_json"])
    if (
        template_relative.as_posix() != safe_relative(
            str(plan_row["template_config_relative_path"])
        ).as_posix()
        or sha256_file(template_path) != str(plan_row["template_config_sha256"])
        or list(config.path_ids) != expected_paths
        or config.path_set_id != str(plan_row["candidate_set_id"])
        or config.candidate_set_hash != str(plan_row["candidate_set_hash"])
        or config.projection_identity != str(plan_row["projection_identity"])
    ):
        raise RuntimeError("H8 canary path resolution audit failed")
    basis = {
        "checkpoint_write_pass": True,
        "checkpoint_read_pass": True,
        "checkpoint_exact_update_verified": True,
        "checkpoint_strict_state_complete": True,
        "checkpoint_rng_roundtrip": True,
        "checkpoint_state_roundtrip": True,
        "checkpoint_restored_in_fresh_session": True,
        "path_resolution_pass": True,
        "optimizer_step_pass": True,
        "optimizer_steps": endpoint,
        "optimizer_state_entry_count": len(step_values),
        "optimizer_step_values": sorted(set(step_values)),
        "template_config_relative_path": template_relative.as_posix(),
        "template_config_sha256": sha256_file(template_path),
        "resolved_config_sha256": sha256_file(run_root / "RESOLVED_CONFIG.json"),
        "sidecar_receipt_sha256": sha256_file(sidecar_path),
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
    }
    return {**basis, "identity": canonical_json_hash(basis), "status": "PASS"}


def validate_stage_marker(
    marker_path: Path,
    *,
    plan_row: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    value = read_json(marker_path)
    basis = value.get("basis")
    endpoint = int(plan_row["endpoint_update"])
    checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    canary_audit = basis.get("canary_engineering_audit") if isinstance(basis, Mapping) else None
    canary_audit_valid = (
        str(plan_row["stage"]) != "CANARY"
        or (
            isinstance(canary_audit, Mapping)
            and canary_audit.get("status") == "PASS"
            and canary_audit.get("identity")
            == canonical_json_hash(
                {key: item for key, item in canary_audit.items() if key not in {"identity", "status"}}
            )
            and canary_audit.get("checkpoint_write_pass") is True
            and canary_audit.get("checkpoint_read_pass") is True
            and canary_audit.get("checkpoint_state_roundtrip") is True
            and canary_audit.get("checkpoint_restored_in_fresh_session") is True
            and canary_audit.get("checkpoint_strict_state_complete") is True
            and canary_audit.get("checkpoint_rng_roundtrip") is True
            and canary_audit.get("path_resolution_pass") is True
            and canary_audit.get("optimizer_step_pass") is True
            and int(canary_audit.get("optimizer_steps", -1)) == CANARY_ENDPOINT
            and int(canary_audit.get("optimizer_state_entry_count", 0)) > 0
            and list(canary_audit.get("optimizer_step_values", [])) == [CANARY_ENDPOINT]
        )
    )
    if (
        value.get("schema_version") != "metachorus.v53.m4a-h8.stage-run-complete.v1"
        or value.get("status") != "PASS"
        or not isinstance(basis, Mapping)
        or value.get("identity") != canonical_json_hash(basis)
        or basis.get("physical_run_id") != str(plan_row["physical_run_id"])
        or basis.get("dataset") != str(plan_row["dataset"])
        or basis.get("candidate_set_hash") != str(plan_row["candidate_set_hash"])
        or int(basis.get("model_seed", -1)) != int(plan_row["model_seed"])
        or int(basis.get("attempt_index", -1)) != int(plan_row.get("attempt_index", 0))
        or int(basis.get("continuation_from_update", -1)) != 0
        or int(basis.get("endpoint_update", -1)) != endpoint
        or int(basis.get("optimizer_steps", -1)) != endpoint
        or basis.get("labels_mounted") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
        or basis.get("h5_locked_report_access") is not False
        or not (
            str(basis.get("gpu_uuid", "")).startswith("GPU-")
            or basis.get("gpu_uuid") == "LOCAL_TEST_NO_GPU"
        )
        or re.fullmatch(r"[0-9a-f]{64}", str(basis.get("runtime_domain_identity", "")))
        is None
        or not checkpoint.is_file()
        or not representation.is_file()
        or sha256_file(checkpoint) != basis.get("checkpoint_sha256")
        or sha256_file(representation) != basis.get("representation_sha256")
        or basis.get("retention", {}).get("status") != "PASS"
        or not canary_audit_valid
    ):
        raise ValueError("invalid H8 stage marker")
    return dict(value)


def run_plan_row(
    *,
    plan_row: Mapping[str, Any],
    repo_root: Path,
    cache_root: Path,
    result_root: Path,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    stage = str(plan_row["stage"])
    endpoint = int(plan_row["endpoint_update"])
    role = str(plan_row["run_role"])
    approved = {
        ("CANARY", 0, CANARY_ENDPOINT, "M4A_H8_ENGINEERING_CANARY"),
        ("S_EVAL", 0, SCIENCE_ENDPOINT, "M4A_H8_S_EVAL_CONFIRMATION"),
    }
    if (
        stage,
        int(plan_row["continuation_from_update"]),
        endpoint,
        role,
    ) not in approved:
        raise ValueError("unapproved H8 training boundary")
    if _strict_bool(plan_row.get("labels_mounted", False)) or int(
        plan_row.get("downstream_evaluations_during_training", -1)
    ) != 0:
        raise ValueError("H8 training process cannot mount labels or evaluate downstream")
    run_id = str(plan_row["physical_run_id"])
    if run_id != h8_physical_run_id(
        stage,
        str(plan_row["dataset"]),
        int(plan_row["candidate_set_ordinal"]),
        int(plan_row["model_seed"]),
    ):
        raise ValueError("H8 physical run identity mismatch")
    attempt_index = int(plan_row.get("attempt_index", 0))
    base = "canary_runs" if stage == "CANARY" else "runs"
    run_root = result_root / base / run_id / f"attempt_{attempt_index:02d}"
    marker_path = run_root / _marker_name(stage)
    if marker_path.is_file():
        return validate_stage_marker(marker_path, plan_row=plan_row, run_root=run_root)
    if run_root.exists():
        raise RuntimeError("preserve the failed H8 attempt and allocate a new attempt")
    relative = safe_relative(str(plan_row["template_config_relative_path"]))
    template_path = result_root.joinpath(*relative.parts)
    result_resolved = result_root.resolve(strict=True)
    if template_path.is_symlink() or not template_path.is_file():
        raise ValueError("H8 template config must be a regular file")
    try:
        template_path.resolve(strict=True).relative_to(result_resolved)
    except ValueError as error:
        raise ValueError("H8 template config escapes the result root") from error
    if sha256_file(template_path) != str(plan_row["template_config_sha256"]):
        raise ValueError("H8 template config hash mismatch")
    config = build_h8_config(
        read_json(template_path),
        repo_root=repo_root,
        dataset_cache_root=cache_root / str(plan_row["dataset"]),
        output_root=run_root,
        plan_row=plan_row,
    )
    runtime_domain = _runtime_domain() if enforce_target_gpu else {
        "gpu_uuid": "LOCAL_TEST_NO_GPU",
        "runtime_domain_identity": canonical_json_hash({"local_test_no_gpu": True}),
    }
    session = O0TrainingSession(
        config, repo_root=repo_root, enforce_target_gpu=enforce_target_gpu
    )
    if stage == "CANARY":
        run = session.run(stop_at_update=CANARY_ENDPOINT)
    else:
        run = session.run(stop_at_update=SCIENCE_CHECKPOINT_SCHEDULE[0])
        first = run_root / "checkpoints" / "update_001000.pt"
        run = session.run(
            resume_checkpoint=first,
            stop_at_update=SCIENCE_CHECKPOINT_SCHEDULE[1],
        )
        _retire_intermediate(
            run_root,
            retired_update=SCIENCE_CHECKPOINT_SCHEDULE[0],
            retained_update=SCIENCE_CHECKPOINT_SCHEDULE[1],
        )
        second = run_root / "checkpoints" / "update_002000.pt"
        run = session.run(resume_checkpoint=second, stop_at_update=SCIENCE_ENDPOINT)
    if (
        run.get("labels_mounted") is not False
        or int(run.get("downstream_evaluations", -1)) != 0
    ):
        raise RuntimeError("H8 runner reported label access or downstream evaluation during training")
    atomic_write_json(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    retention = validate_rolling_retention(run_root, endpoint=endpoint)
    endpoint_checkpoint = run_root / "checkpoints" / f"update_{endpoint:06d}.pt"
    endpoint_representation = run_root / "representations" / f"update_{endpoint:06d}.npy"
    representation = np.load(endpoint_representation, allow_pickle=False)
    if representation.ndim != 2 or representation.size == 0 or not np.isfinite(representation).all():
        raise RuntimeError("H8 endpoint representation is invalid")
    trained_snapshot = session.state_snapshot() if stage == "CANARY" else None
    if stage == "CANARY":
        del session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    canary_audit = (
        _canary_engineering_audit(
            trained_snapshot=trained_snapshot,
            run_root=run_root,
            result_root=result_root,
            repo_root=repo_root,
            enforce_target_gpu=enforce_target_gpu,
            template_path=template_path,
            plan_row=plan_row,
            config=config,
        )
        if stage == "CANARY"
        else None
    )
    basis = {
        "physical_run_id": run_id,
        "dataset": str(plan_row["dataset"]),
        "role": str(plan_row["role"]),
        "candidate_set_id": str(plan_row["candidate_set_id"]),
        "candidate_set_hash": str(plan_row["candidate_set_hash"]),
        "candidate_set_ordinal": int(plan_row["candidate_set_ordinal"]),
        "model_seed": int(plan_row["model_seed"]),
        "attempt_index": attempt_index,
        "continuation_from_update": 0,
        "endpoint_update": endpoint,
        "optimizer_steps": int(run["optimizer_steps"]),
        "optimizer_steps_this_invocation": endpoint,
        "model_total_seconds": float(run["model_total_seconds"]),
        "peak_gpu_allocated_bytes": int(run["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(run["peak_gpu_reserved_bytes"]),
        "gpu_uuid": runtime_domain["gpu_uuid"],
        "runtime_domain_identity": runtime_domain["runtime_domain_identity"],
        "checkpoint_relative_path": endpoint_checkpoint.relative_to(result_root).as_posix(),
        "checkpoint_sha256": sha256_file(endpoint_checkpoint),
        "checkpoint_size_bytes": endpoint_checkpoint.stat().st_size,
        "representation_relative_path": endpoint_representation.relative_to(result_root).as_posix(),
        "representation_sha256": sha256_file(endpoint_representation),
        "representation_size_bytes": endpoint_representation.stat().st_size,
        "representation_shape": list(representation.shape),
        "resource_trace_relative_path": (run_root / "RESOURCE_TRACE.csv").relative_to(result_root).as_posix(),
        "resource_trace_sha256": sha256_file(run_root / "RESOURCE_TRACE.csv"),
        "continuation_mode": str(plan_row["continuation_mode"]),
        "run_role": role,
        "plan_seal_sha256": str(plan_row["plan_seal_sha256"]),
        "eligible_for_scientific_result": stage == "S_EVAL",
        "eligible_for_selector_development": False,
        "reusable_for_science": stage == "S_EVAL",
        "labels_mounted": run["labels_mounted"],
        "downstream_evaluations_during_training": int(run["downstream_evaluations"]),
        "h5_locked_report_access": False,
        "retention": retention,
        "canary_engineering_audit": canary_audit,
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h8.stage-run-complete.v1",
        "status": "PASS",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }
    atomic_write_json(marker_path, payload)
    return payload
