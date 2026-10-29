from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import traceback
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluate_representation import evaluate_representation
from m9_common import (
    MODEL_SEEDS,
    atomic_write_json,
    canonical_hash,
    git_identity,
    load_json,
    read_csv,
    sha256_file,
)


REGISTRY_FIELDS = (
    "event_identity", "event_time_utc", "physical_run_id", "attempt_index",
    "run_kind", "dataset", "p_e", "p_f", "model_seed", "set_id", "set_hash",
    "frozen_config_sha", "status", "optimizer_steps", "representation_sha256",
    "checkpoint_sha256", "error_type", "error_message",
)


def _append_registry_event(project_root: Path, event: Mapping[str, Any]) -> None:
    import fcntl

    path = project_root / "runs" / "AUGMENTATION_RUN_REGISTRY.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field: event.get(field, "") for field in REGISTRY_FIELDS if field != "event_identity"}
    payload["event_identity"] = canonical_hash(payload)
    ordered = {field: payload.get(field, "") for field in REGISTRY_FIELDS}
    with path.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = {row["event_identity"] for row in csv.DictReader(handle)} if handle.readable() else set()
        if ordered["event_identity"] not in existing:
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(ordered)
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _one_visible_4090() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("cell worker requires exactly one visible CUDA device")
    name = torch.cuda.get_device_name(0)
    if "4090" not in name:
        raise RuntimeError(f"cell worker requires RTX 4090, observed {name}")
    return {
        "gpu_name": name,
        "gpu_uuid": os.environ.get("M9_GPU_UUID", "ATTESTED_BY_SCHEDULER"),
        "runtime_domain": os.environ.get("M9_RUNTIME_DOMAIN", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _row_by_id(matrix_path: Path, physical_run_id: str) -> dict[str, str]:
    matches = [row for row in read_csv(matrix_path) if row["physical_run_id"] == physical_run_id]
    if len(matches) != 1:
        raise ValueError("physical_run_id must identify exactly one sealed matrix row")
    return matches[0]


def _frozen_by_dataset(path: Path, dataset: str) -> dict[str, str]:
    matches = [row for row in read_csv(path) if row["dataset"] == dataset]
    if len(matches) != 1:
        raise ValueError("dataset must identify exactly one frozen config row")
    return matches[0]


def _validate_m9_config(config: Any, row: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    canary = row["run_kind"] == "ENGINEERING_CANARY"
    endpoint = 20 if canary else 3000
    expected_schedule = (20,) if canary else (1000, 2000, 3000)
    expected_representation = (20,) if canary else (3000,)
    if hasattr(config, "resolved_representation_schedule"):
        representation_schedule_ok = tuple(config.resolved_representation_schedule()) == expected_representation
    else:
        representation_schedule_ok = tuple(config.checkpoint_schedule) == expected_schedule
    checks = {
        "dataset": config.dataset == row["dataset"] == frozen["dataset"],
        "set_id": config.path_set_id == frozen["final_set_id"] == row["set_id"],
        "set_hash": config.candidate_set_hash == frozen["final_set_hash"] == row["set_hash"],
        "seed": int(config.model_seed) == int(row["model_seed"]) and int(config.model_seed) in MODEL_SEEDS,
        "endpoint": int(config.max_updates) == endpoint,
        "checkpoint_schedule": tuple(config.checkpoint_schedule) == expected_schedule,
        "representation_schedule": representation_schedule_ok,
        "trainer": config.trainer_id == "O0-CodeFaithful",
        "combine": config.combine == frozen["combine"],
        "num_layers": int(config.num_layers) == int(frozen["num_layers"]),
        "hid_dim": int(config.hid_dim) == int(frozen["hidden_dim"]),
        "lr": float(config.lr) == float(frozen["learning_rate"]),
        "tau": float(config.tau) == float(frozen["tau"]),
        "alpha": float(config.alpha) == float(frozen["alpha"]),
        "weight_decay": float(config.weight_decay) == float(frozen["weight_decay"]),
        "ema": float(config.moving_average_decay) == float(frozen["EMA_decay"]),
        "p_e": float(config.edge_drop_rate) == float(row["p_e"]),
        "p_f": float(config.feature_drop_rate) == float(row["p_f"]),
        "run_role": config.run_role == frozen["base_run_role"],
        "capture": config.capture_mode == "on",
        "physical_run_id": config.physical_run_id == row["physical_run_id"],
        "scientific_role": config.eligible_for_scientific_result is (not canary),
        "selector_role": config.eligible_for_selector_development is False,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"M9 config contract failed: {failed}")


def run_cell(
    *, row: Mapping[str, Any], frozen: Mapping[str, Any], sources: Mapping[str, Any],
    project_root: Path, attempt_index: int, validate_only: bool = False,
) -> dict[str, Any]:
    if attempt_index not in (0, 1):
        raise ValueError("M9 permits at most one engineering recovery attempt")
    gpu = _one_visible_4090()
    dataset = row["dataset"]
    entry = sources["datasets"][dataset]
    identity = git_identity(entry["source_root"])
    if identity["commit"] != frozen["source_commit"] or identity["tree"] != frozen["source_tree"] or not identity["clean"]:
        raise RuntimeError("frozen source identity mismatch")
    source_root = Path(entry["source_root"]).resolve()
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "src"))
    try:
        from metachorus_v53.training.o0_codefaithful_runner import O0RunConfig, O0TrainingSession

        @dataclass(frozen=True)
        class M9RunConfig(O0RunConfig):
            def validate(self) -> None:
                _validate_m9_config(self, row, frozen)

        run_root = project_root / "runs" / row["physical_run_id"] / f"attempt_{attempt_index:02d}"
        engine_root = run_root / "engine"
        failure_path = project_root / "qa" / "failed_attempts" / f"{row['physical_run_id']}_attempt_{attempt_index:02d}.json"
        if run_root.exists():
            marker = run_root / "M9_VIS_P1_RUN_COMPLETE.json"
            if marker.is_file():
                prior = load_json(marker)
                if prior.get("status") == "PASS" and prior.get("row_identity") == row["row_identity"]:
                    return prior
            if engine_root.exists() or not failure_path.is_file():
                raise FileExistsError(f"refusing to overwrite non-bootstrap attempt root: {run_root}")
        elif not validate_only:
            run_root.mkdir(parents=True)
        canary = row["run_kind"] == "ENGINEERING_CANARY"
        config_kwargs = dict(
            dataset=dataset,
            path_set_role=entry["final_method_role"],
            path_set_id=frozen["final_set_id"],
            path_ids=tuple(ast.literal_eval(frozen["path_ids"]) if isinstance(frozen["path_ids"], str) else frozen["path_ids"]),
            candidate_set_hash=frozen["final_set_hash"],
            model_seed=int(row["model_seed"]),
            max_updates=20 if canary else 3000,
            capture_mode="on",
            checkpoint_schedule=(20,) if canary else (1000, 2000, 3000),
            cache_root=frozen["cache_root"],
            output_root=str(engine_root),
            device="cuda:0",
            source_commit=frozen["source_commit"],
            source_tree=frozen["source_tree"],
            cache_identity=frozen["cache_identity"],
            augmentation_namespace=f"m9-vis-p1/{row['physical_run_id']}",
            run_role=frozen["base_run_role"],
            eligible_for_scientific_result=not canary,
            eligible_for_selector_development=False,
            trainer_id="O0-CodeFaithful",
            combine=frozen["combine"],
            num_layers=int(frozen["num_layers"]),
            hid_dim=int(frozen["hidden_dim"]),
            lr=float(frozen["learning_rate"]),
            tau=float(frozen["tau"]),
            edge_drop_rate=float(row["p_e"]),
            feature_drop_rate=float(row["p_f"]),
            alpha=float(frozen["alpha"]),
            weight_decay=float(frozen["weight_decay"]),
            moving_average_decay=float(frozen["EMA_decay"]),
            physical_run_id=row["physical_run_id"],
            candidate_set_ordinal=int(frozen["final_set_ordinal"]),
            source_data_identity=frozen["source_data_identity"],
            projection_identity=frozen["projection_identity"],
            candidate_bank_hash=frozen["candidate_bank_hash"],
            plan_seal_hash=row["row_identity"],
        )
        if "representation_schedule" in {field.name for field in fields(O0RunConfig)}:
            config_kwargs["representation_schedule"] = (20,) if canary else (3000,)
        config = M9RunConfig(**config_kwargs)
        config.validate()
        session = O0TrainingSession(config, repo_root=source_root, enforce_target_gpu=True)
        if validate_only:
            value = {
                "status": "VALIDATED_NO_TRAINING",
                "physical_run_id": row["physical_run_id"],
                "attempt_index": attempt_index,
                "dataset": dataset,
                "config_identity": config.identity(),
                "base_config_supports_representation_schedule": "representation_schedule" in {field.name for field in fields(O0RunConfig)},
            }
            value["identity"] = canonical_hash(value)
            return value
        started_at = datetime.now(timezone.utc).isoformat()
        started = {
            "schema_version": "metachorus.m9-vis-p1.run-started.v1",
            "status": "STARTED",
            "event_time_utc": started_at,
            "physical_run_id": row["physical_run_id"],
            "attempt_index": attempt_index,
            "run_kind": row["run_kind"],
            "dataset": dataset,
            "p_e": float(row["p_e"]),
            "p_f": float(row["p_f"]),
            "model_seed": int(row["model_seed"]),
            "set_id": row["set_id"],
            "set_hash": row["set_hash"],
            "frozen_config_sha": row["frozen_config_sha"],
        }
        started["identity"] = canonical_hash(started)
        started_path = run_root / "M9_VIS_P1_RUN_STARTED.json"
        if not started_path.exists():
            atomic_write_json(started_path, started)
        _append_registry_event(project_root, started)
        run = session.run(stop_at_update=config.max_updates)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    if (
        run.get("labels_mounted") is not False
        or int(run.get("downstream_evaluations", -1)) != 0
        or int(run.get("optimizer_steps", -1)) != int(config.max_updates)
    ):
        raise RuntimeError("O0 run violated training isolation")
    atomic_write_json(run_root / "RESOLVED_CONFIG.json", dict(config.__dict__))
    representation = engine_root / "representations" / f"update_{config.max_updates:06d}.npy"
    checkpoint = engine_root / "checkpoints" / f"update_{config.max_updates:06d}.pt"
    if not representation.is_file() or not checkpoint.is_file():
        raise RuntimeError("endpoint checkpoint/representation missing")
    evaluation = None
    if not canary:
        evaluation = evaluate_representation(
            representation, dataset=dataset, entry=entry
        )
        atomic_write_json(run_root / "M9_VIS_P1_EVALUATION.json", evaluation)
    marker = {
        "schema_version": "metachorus.m9-vis-p1.run-complete.v1",
        "status": "PASS",
        "row_identity": row["row_identity"],
        "attempt_index": attempt_index,
        "engine_output_relative_path": engine_root.relative_to(project_root).as_posix(),
        "bootstrap_reentry_same_attempt": failure_path.is_file(),
        "run_kind": row["run_kind"],
        "physical_run_id": row["physical_run_id"],
        "dataset": dataset,
        "p_e": float(row["p_e"]),
        "p_f": float(row["p_f"]),
        "model_seed": int(row["model_seed"]),
        "set_id": row["set_id"],
        "set_hash": row["set_hash"],
        "frozen_config_sha": row["frozen_config_sha"],
        "optimizer_steps": int(run["optimizer_steps"]),
        "training_from_zero": True,
        "labels_mounted_during_training": False,
        "downstream_evaluations_during_training": 0,
        "representation_sha256": sha256_file(representation),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_total_seconds": float(run.get("model_total_seconds", 0.0)),
        "gpu": gpu,
        "evaluation_identity": evaluation.get("evaluation_identity") if evaluation else None,
        "event_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker["identity"] = canonical_hash(marker)
    atomic_write_json(run_root / "M9_VIS_P1_RUN_COMPLETE.json", marker)
    _append_registry_event(project_root, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--physical-run-id", required=True)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    row = _row_by_id(Path(args.matrix), args.physical_run_id)
    frozen = _frozen_by_dataset(Path(args.frozen_config), row["dataset"])
    if row["frozen_config_sha"] != frozen["config_sha256"]:
        raise SystemExit("matrix/frozen config identity mismatch")
    try:
        result = run_cell(
            row=row, frozen=frozen, sources=load_json(args.sources),
            project_root=project_root, attempt_index=args.attempt_index,
            validate_only=args.validate_only,
        )
    except Exception as error:
        failure_root = project_root / "qa" / ("adapter_preflight_failures" if args.validate_only else "failed_attempts")
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = failure_root / f"{args.physical_run_id}_attempt_{args.attempt_index:02d}.json"
        if not failure_path.exists():
            atomic_write_json(failure_path, {
                "schema_version": "metachorus.m9-vis-p1.failed-attempt.v1",
                "status": "FAILED",
                "physical_run_id": args.physical_run_id,
                "attempt_index": args.attempt_index,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback_sha256": canonical_hash(traceback.format_exc()),
            })
        if not args.validate_only and "row" in locals():
            _append_registry_event(project_root, {
                "event_time_utc": datetime.now(timezone.utc).isoformat(),
                "physical_run_id": args.physical_run_id,
                "attempt_index": args.attempt_index,
                "run_kind": row.get("run_kind", ""),
                "dataset": row.get("dataset", ""),
                "p_e": row.get("p_e", ""),
                "p_f": row.get("p_f", ""),
                "model_seed": row.get("model_seed", ""),
                "set_id": row.get("set_id", ""),
                "set_hash": row.get("set_hash", ""),
                "frozen_config_sha": row.get("frozen_config_sha", ""),
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
        raise
    print({"status": result["status"], "physical_run_id": result["physical_run_id"], "identity": result["identity"]})


if __name__ == "__main__":
    main()
