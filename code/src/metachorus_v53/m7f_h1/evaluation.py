"""Post-training evaluators with explicit GLOBAL/LOCKED label boundaries."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash, sha256_file
from metachorus_v53.m6f_m7f_fast.evaluation import (
    PreliminaryLabelSession,
    evaluate_preliminary_representation,
    evaluate_search_representation,
    open_search_mount,
)

from .common import envelope, read_csv, safe_relative, write_csv_once, write_json_once
from .constants import (
    BUDGET_MODEL_SEEDS,
    EVALUATION_MODEL_SEEDS,
    IMDB_FINAL_SET,
    IMDB_MANUAL_SET,
    MANUAL_DATASETS,
)
from .trajectory import summarize_locked_metrics, validate_selection_seal


def _load_representation(
    result_root: Path, row: Mapping[str, Any]
) -> tuple[np.ndarray, Path]:
    relative = str(
        row.get("relative_path", row.get("representation_relative_path", ""))
    )
    governed = safe_relative(relative)
    candidate = result_root.joinpath(*governed.parts)
    cursor = result_root
    for part in governed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("representation symlink chains are forbidden")
    source = candidate.resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        source.relative_to(result_root)
    except ValueError as exc:
        raise ValueError("representation escapes the governed result root") from exc
    if sha256_file(source) != row.get(
        "sha256", row.get("representation_sha256")
    ):
        raise ValueError("representation SHA-256 differs from the sealed row")
    values = np.load(source, allow_pickle=False)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("representation must be a finite 2D array")
    return values, source


def _flatten_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    metrics = value.get("metrics", {})
    required = ("Micro_F1", "Macro_F1", "NMI", "ARI", "F_cls", "F_clu", "F_joint")
    if not isinstance(metrics, Mapping) or any(name not in metrics for name in required):
        raise ValueError("evaluator did not return the frozen metric schema")
    output = {name: float(metrics[name]) for name in required}
    if any(not np.isfinite(number) for number in output.values()):
        raise ValueError("evaluator returned non-finite metrics")
    return output


def _existing_metric_update(row: Mapping[str, Any]) -> int:
    available = [name for name in ("update", "endpoint_update") if name in row]
    if not available:
        raise ValueError("existing metric row lacks its update identity")
    updates = {int(row[name]) for name in available}
    if len(updates) != 1:
        raise ValueError("existing metric update columns disagree")
    return next(iter(updates))


def evaluate_trajectory_search_pool(
    *,
    result_root: str | Path,
    run_markers: Sequence[Mapping[str, Any]],
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(result_root).resolve(strict=True)
    search = open_search_mount(
        role_manifest_path=role_manifest_path,
        role_seal_path=role_seal_path,
        private_mount_audit_path=private_mount_audit_path,
        private_search_mount_root=private_search_mount_root,
        dataset="imdb",
    )
    rows: list[dict[str, Any]] = []
    for marker in run_markers:
        basis = marker.get("basis", {})
        if (
            marker.get("status") != "PASS"
            or marker.get("identity") != canonical_json_hash(basis)
            or basis.get("stage") != "IMDB_TRAJECTORY"
            or basis.get("dataset") != "imdb"
            or basis.get("labels_mounted") is not False
        ):
            raise ValueError("IMDb trajectory marker is invalid")
        representations = list(basis.get("representations", []))
        if len(representations) not in {4, 60}:
            raise ValueError("IMDb trajectory must expose 4 existing or 60 rerun updates")
        for representation in representations:
            values, source = _load_representation(root, representation)
            started = time.perf_counter()
            evaluated = evaluate_search_representation(
                values, dataset="imdb", search_mount=search
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "dataset": "imdb",
                    "physical_run_id": basis["physical_run_id"],
                    "path_set_role": basis["path_set_role"],
                    "candidate_set_id": basis["candidate_set_id"],
                    "candidate_set_hash": basis["candidate_set_hash"],
                    "model_seed": int(basis["model_seed"]),
                    "update": int(representation["update"]),
                    **_flatten_metrics(evaluated),
                    "representation_relative_path": source.relative_to(root).as_posix(),
                    "representation_sha256": representation["sha256"],
                    "metric_source_relative_path": source.relative_to(root).as_posix(),
                    "metric_source_sha256": representation["sha256"],
                    "evidence_type": "EVALUATED_REPRESENTATION",
                    "evaluation_wall_time_seconds": elapsed,
                    "metric_role": "GLOBAL_SEARCH_POOL",
                    "prelim_locked_report_access": False,
                    "raw_labels_persisted": False,
                    "status": "PASS",
                }
            )
    return rows


def evaluate_existing_trajectory_evidence(
    *,
    existing_result_root: str | Path,
    inventory_rows: Sequence[Mapping[str, Any]],
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
) -> list[dict[str, Any]]:
    """Normalize exact parent metrics, filling gaps from retained representations."""

    root = Path(existing_result_root).resolve(strict=True)
    search = open_search_mount(
        role_manifest_path=role_manifest_path,
        role_seal_path=role_seal_path,
        private_mount_audit_path=private_mount_audit_path,
        private_search_mount_root=private_search_mount_root,
        dataset="imdb",
    )
    normalized_inventory = [dict(row) for row in inventory_rows]
    by_key: dict[tuple[str, int], dict[str, Any]] = {}

    def identity(run_id: str) -> tuple[str, Mapping[str, Any], int]:
        matched = re.fullmatch(
            r"M7F-H1-IMDB-TRAJECTORY-C(000|014)-E(0[0-2])", run_id
        )
        if matched is None:
            raise ValueError("existing IMDb trajectory run identity is invalid")
        candidate = IMDB_FINAL_SET if matched.group(1) == "014" else IMDB_MANUAL_SET
        role = "FINAL_FROZEN_METHOD" if matched.group(1) == "014" else "MANUAL"
        seed = EVALUATION_MODEL_SEEDS[int(matched.group(2))]
        return role, candidate, seed

    for inventory in normalized_inventory:
        if inventory.get("usable") not in {True, "True", "true"}:
            continue
        run_id = str(inventory["physical_run_id"])
        role, candidate, seed = identity(run_id)
        relative = str(inventory["remote_relative_path"])
        suffix = Path(relative).suffix.casefold()
        if suffix == ".csv":
            source = _load_governed_text(root, relative, str(inventory["sha256"]))
            for raw in read_csv(source):
                if str(raw.get("physical_run_id", inventory.get("source_physical_run_id", ""))) not in {
                    run_id,
                    str(inventory.get("source_physical_run_id", "")),
                }:
                    continue
                update = _existing_metric_update(raw)
                if update not in {300, 1000, 2000, 3000}:
                    continue
                evaluated = {"metrics": raw}
                by_key[(run_id, update)] = {
                    "dataset": "imdb",
                    "physical_run_id": run_id,
                    "path_set_role": role,
                    "candidate_set_id": candidate["candidate_set_id"],
                    "candidate_set_hash": candidate["candidate_set_hash"],
                    "model_seed": seed,
                    "update": update,
                    **_flatten_metrics(evaluated),
                    "representation_relative_path": str(
                        raw.get("representation_relative_path", "")
                    ),
                    "representation_sha256": str(
                        raw.get("representation_sha256", "")
                    ),
                    "metric_source_relative_path": relative,
                    "metric_source_sha256": str(inventory["sha256"]),
                    "evidence_type": "EXISTING_METRIC",
                    "metric_role": "GLOBAL_SEARCH_POOL",
                    "prelim_locked_report_access": False,
                    "raw_labels_persisted": False,
                    "status": "PASS",
                }
    for inventory in normalized_inventory:
        if inventory.get("usable") not in {True, "True", "true"}:
            continue
        relative = str(inventory["remote_relative_path"])
        if Path(relative).suffix.casefold() != ".npy":
            continue
        update_match = re.search(r"update_(\d{6})\.npy$", relative)
        if update_match is None:
            continue
        update = int(update_match.group(1))
        run_id = str(inventory["physical_run_id"])
        role, candidate, seed = identity(run_id)
        values, source = _load_representation(
            root, {"relative_path": relative, "sha256": inventory["sha256"]}
        )
        evaluated = evaluate_search_representation(
            values, dataset="imdb", search_mount=search
        )
        key = (run_id, update)
        if (
            key in by_key
            and by_key[key].get("evidence_type") == "EVALUATED_REPRESENTATION"
        ):
            raise ValueError(
                "existing IMDb evidence has duplicate retained representations"
            )
        
        
        
        by_key[key] = {
            "dataset": "imdb",
            "physical_run_id": run_id,
            "path_set_role": role,
            "candidate_set_id": candidate["candidate_set_id"],
            "candidate_set_hash": candidate["candidate_set_hash"],
            "model_seed": seed,
            "update": update,
            **_flatten_metrics(evaluated),
            "representation_relative_path": source.relative_to(root).as_posix(),
            "representation_sha256": str(inventory["sha256"]),
            "metric_source_relative_path": source.relative_to(root).as_posix(),
            "metric_source_sha256": str(inventory["sha256"]),
            "evidence_type": "EVALUATED_REPRESENTATION",
            "metric_role": "GLOBAL_SEARCH_POOL",
            "prelim_locked_report_access": False,
            "raw_labels_persisted": False,
            "status": "PASS",
        }
    expected = {
        (f"M7F-H1-IMDB-TRAJECTORY-C{candidate:03d}-E{ordinal:02d}", update)
        for candidate in (0, 14)
        for ordinal in range(3)
        for update in (300, 1000, 2000, 3000)
    }
    if set(by_key) != expected:
        raise ValueError("existing IMDb evidence does not cover exact six-by-four rows")
    return [by_key[key] for key in sorted(by_key)]


def _load_governed_text(root: Path, relative: str, expected_sha256: str) -> Path:
    governed = safe_relative(relative)
    candidate = root.joinpath(*governed.parts)
    cursor = root
    for part in governed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("existing metric source symlink chains are forbidden")
    source = candidate.resolve(strict=True)
    source.relative_to(root)
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError("existing metric source identity changed")
    return source


def _training_registry_seal(
    *,
    active_datasets: Sequence[str],
    run_markers: Sequence[Mapping[str, Any]],
    evidence_identities: Sequence[str] = (),
) -> dict[str, Any]:
    marker_identities = [str(marker.get("identity", "")) for marker in run_markers]
    marker_identities.extend(map(str, evidence_identities))
    if not marker_identities or any(len(value) != 64 for value in marker_identities):
        raise ValueError("locked evaluator requires exact training completion identities")
    return envelope(
        "metachorus.v53.m7f-h1.training-registry-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        {
            "active_datasets": list(active_datasets),
            "run_marker_identities": sorted(marker_identities),
            "run_count": len(marker_identities),
            "all_status_pass": True,
            "labels_mounted": False,
            "preliminary_evaluation_started": False,
        },
    )


def evaluate_locked_selections(
    *,
    result_root: str | Path,
    selection_rows: Sequence[Mapping[str, Any]],
    selection_seal: Mapping[str, Any],
    trajectory_markers: Sequence[Mapping[str, Any]],
    role_manifest_path: str | Path,
    private_search_mount_root: str | Path,
    label_source: str | Path,
    label_source_root: str | Path,
    representation_root: str | Path | None = None,
    trajectory_evidence_identities: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selections = [dict(row) for row in selection_rows]
    validate_selection_seal(selections, selection_seal)
    root = Path(result_root).resolve(strict=True)
    representation_base = (
        root
        if representation_root is None
        else Path(representation_root).resolve(strict=True)
    )
    registry = _training_registry_seal(
        active_datasets=("imdb",),
        run_markers=trajectory_markers,
        evidence_identities=trajectory_evidence_identities,
    )
    session = PreliminaryLabelSession(
        role_manifest_path=role_manifest_path,
        private_search_mount_root=private_search_mount_root,
        label_sources={"imdb": label_source},
        label_source_roots={"imdb": label_source_root},
        training_registry_seal=registry,
    )
    roles = session.open_dataset("imdb")
    locked_rows: list[dict[str, Any]] = []
    for selected in selections:
        values, source = _load_representation(representation_base, selected)
        evaluated = evaluate_preliminary_representation(
            values, dataset="imdb", roles=roles
        )
        locked_rows.append(
            {
                "dataset": "imdb",
                "selector": selected["selector"],
                "path_set_role": selected["path_set_role"],
                "candidate_set_id": selected["candidate_set_id"],
                "candidate_set_hash": selected["candidate_set_hash"],
                "model_seed": int(selected["model_seed"]),
                "selected_update": int(selected["selected_update"]),
                **_flatten_metrics(evaluated),
                "representation_relative_path": source.relative_to(
                    representation_base
                ).as_posix(),
                "representation_sha256": selected["representation_sha256"],
                "metric_role": "PRELIM_LOCKED_REPORT",
                "selection_role": "GLOBAL_SEARCH_POOL_ONLY",
                "selection_seal_identity": selection_seal["identity"],
                "raw_labels_persisted": False,
                "status": "PASS",
            }
        )
    summaries, deltas = summarize_locked_metrics(
        locked_rows,
        selection_rows=selections,
        selection_seal=selection_seal,
    )
    if session.access_count != 1 or session.per_dataset_access_count["imdb"] != 1:
        raise RuntimeError("IMDb locked labels were not opened exactly once")
    audit = envelope(
        "metachorus.v53.m7f-h1.locked-access-audit.v1",
        "PASS",
        {
            "selection_seal_identity": selection_seal["identity"],
            "training_registry_seal_identity": registry["identity"],
            "active_datasets": ["imdb"],
            "locked_report_access_count": 1,
            "per_dataset_access_count": session.per_dataset_access_count,
            "labels_mounted_during_training": False,
            "raw_labels_persisted": False,
            "checkpoint_reselected_after_locked_evaluation": False,
            "locked_metrics_rows_identity": canonical_json_hash(locked_rows),
        },
    )
    return locked_rows, summaries, deltas, audit


def evaluate_manual_baselines(
    *,
    result_root: str | Path,
    run_markers: Sequence[Mapping[str, Any]],
    role_manifest_path: str | Path,
    private_search_mount_root: str | Path,
    label_sources: Mapping[str, str | Path],
    label_source_roots: Mapping[str, str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(result_root).resolve(strict=True)
    registry = _training_registry_seal(
        active_datasets=MANUAL_DATASETS, run_markers=run_markers
    )
    session = PreliminaryLabelSession(
        role_manifest_path=role_manifest_path,
        private_search_mount_root=private_search_mount_root,
        label_sources=label_sources,
        label_source_roots=label_source_roots,
        training_registry_seal=registry,
    )
    metric_rows: list[dict[str, Any]] = []
    for marker in run_markers:
        basis = marker.get("basis", {})
        dataset = str(basis.get("dataset", ""))
        if (
            marker.get("status") != "PASS"
            or marker.get("identity") != canonical_json_hash(basis)
            or basis.get("stage") != "MANUAL_BASELINE"
            or dataset not in MANUAL_DATASETS
            or basis.get("path_set_role") != "MANUAL"
            or basis.get("labels_mounted") is not False
        ):
            raise ValueError("Manual baseline marker is invalid")
        representations = list(basis.get("representations", []))
        if len(representations) != 1 or int(representations[0]["update"]) != 3000:
            raise ValueError("Manual baseline requires exactly the update-3000 representation")
        values, source = _load_representation(root, representations[0])
        evaluated = evaluate_preliminary_representation(
            values, dataset=dataset, roles=session.open_dataset(dataset)
        )
        metric_rows.append(
            {
                "dataset": dataset,
                "physical_run_id": basis["physical_run_id"],
                "candidate_set_id": basis["candidate_set_id"],
                "candidate_set_hash": basis["candidate_set_hash"],
                "model_seed": int(basis["model_seed"]),
                "update": 3000,
                **_flatten_metrics(evaluated),
                "representation_relative_path": source.relative_to(root).as_posix(),
                "representation_sha256": representations[0]["sha256"],
                "classification_fit_role": "GLOBAL_SEARCH_POOL",
                "classification_metric_role": "PRELIM_LOCKED_REPORT",
                "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
                "clustering_metric_role": "PRELIM_LOCKED_REPORT",
                "labels_mounted_during_training": False,
                "raw_labels_persisted": False,
                "status": "PASS",
            }
        )
    expected = {
        (dataset, seed)
        for dataset in MANUAL_DATASETS
        for seed in EVALUATION_MODEL_SEEDS
    }
    if {
        (row["dataset"], row["model_seed"]) for row in metric_rows
    } != expected or len(metric_rows) != 6:
        raise ValueError("Manual baseline evaluator requires exact 2×3 runs")
    summary_rows: list[dict[str, Any]] = []
    for dataset in MANUAL_DATASETS:
        group = [row for row in metric_rows if row["dataset"] == dataset]
        summary = {
            "dataset": dataset,
            "path_set_role": "MANUAL",
            "model_seed_count": 3,
            "status": "MANUAL_BASELINE_COMPLETE",
        }
        for metric in ("Micro_F1", "Macro_F1", "NMI", "ARI", "F_cls", "F_clu", "F_joint"):
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        summary_rows.append(summary)
    if session.access_count != 2 or any(
        session.per_dataset_access_count[name] != 1 for name in MANUAL_DATASETS
    ):
        raise RuntimeError("Manual locked labels were not opened once per dataset")
    audit = envelope(
        "metachorus.v53.m7f-h1.manual-locked-access-audit.v1",
        "PASS",
        {
            "training_registry_seal_identity": registry["identity"],
            "active_datasets": list(MANUAL_DATASETS),
            "locked_report_access_count": 2,
            "per_dataset_access_count": session.per_dataset_access_count,
            "labels_mounted_during_training": False,
            "raw_labels_persisted": False,
            "metric_rows_identity": canonical_json_hash(metric_rows),
        },
    )
    return metric_rows, summary_rows, audit


def evaluate_budget_canaries(
    *,
    result_root: str | Path,
    run_markers: Sequence[Mapping[str, Any]],
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
) -> list[dict[str, Any]]:
    """Measure evaluator overhead without publishing canary science metrics."""

    root = Path(result_root).resolve(strict=True)
    mounts = {
        dataset: open_search_mount(
            role_manifest_path=role_manifest_path,
            role_seal_path=role_seal_path,
            private_mount_audit_path=private_mount_audit_path,
            private_search_mount_root=private_search_mount_root,
            dataset=dataset,
        )
        for dataset in MANUAL_DATASETS
    }
    rows: list[dict[str, Any]] = []
    for marker in run_markers:
        basis = marker.get("basis", {})
        dataset = str(basis.get("dataset", ""))
        representations = list(basis.get("representations", []))
        if (
            marker.get("status") != "PASS"
            or marker.get("identity") != canonical_json_hash(basis)
            or basis.get("stage") != "BUDGET_CANARY"
            or dataset not in MANUAL_DATASETS
            or basis.get("path_set_role")
            not in {"MANUAL", "BUDGET_SHORTEST", "BUDGET_LONGEST"}
            or int(basis.get("model_seed", -1)) not in BUDGET_MODEL_SEEDS
            or basis.get("budget_canary_not_scientific_selection") is not True
            or basis.get("labels_mounted") is not False
            or len(representations) != 1
            or int(representations[0].get("update", -1)) != 300
        ):
            raise ValueError("budget canary completion marker is invalid")
        values, _ = _load_representation(root, representations[0])
        started = time.perf_counter()
        evaluated = evaluate_search_representation(
            values, dataset=dataset, search_mount=mounts[dataset]
        )
        elapsed = time.perf_counter() - started
        if evaluated.get("prelim_locked_report_access") is not False:
            raise RuntimeError("budget evaluator crossed the locked-report boundary")
        rows.append(
            {
                "physical_run_id": basis["physical_run_id"],
                "dataset": dataset,
                "path_set_role": basis["path_set_role"],
                "candidate_set_id": basis["candidate_set_id"],
                "candidate_set_hash": basis["candidate_set_hash"],
                "model_seed": int(basis["model_seed"]),
                "endpoint_update": 300,
                "model_total_seconds": float(basis["model_total_seconds"]),
                "checkpoint_overhead_seconds": float(
                    basis.get("checkpoint_overhead_seconds", 0.0)
                ),
                "evaluation_overhead_seconds": elapsed,
                "run_output_bytes": int(basis["run_output_bytes"]),
                "path_resolution_pass": True,
                "labels_mounted": False,
                "budget_canary_not_scientific_selection": True,
                "nan_or_inf_observed": False,
                "scientific_metrics_persisted": False,
                "status": "PASS",
            }
        )
    expected = {
        (dataset, role, seed)
        for dataset in MANUAL_DATASETS
        for role in ("MANUAL", "BUDGET_SHORTEST", "BUDGET_LONGEST")
        for seed in BUDGET_MODEL_SEEDS
    }
    if {
        (row["dataset"], row["path_set_role"], row["model_seed"])
        for row in rows
    } != expected or len(rows) != 12:
        raise ValueError("budget evaluator requires exact 2×3×2 canaries")
    return rows


def write_locked_outputs(
    result_root: str | Path,
    *,
    locked_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    access_audit: Mapping[str, Any],
) -> None:
    root = Path(result_root)
    write_csv_once(
        root / "IMDB_LOCKED_METRICS_BY_SELECTOR_AND_SEED.csv",
        locked_rows,
        fieldnames=list(locked_rows[0]),
    )
    write_csv_once(
        root / "IMDB_SELECTOR_SUMMARY.csv",
        summaries,
        fieldnames=list(summaries[0]),
    )
    write_csv_once(
        root / "IMDB_SELECTED_VS_MANUAL_DELTAS.csv",
        deltas,
        fieldnames=list(deltas[0]),
    )
    write_json_once(root / "M7F_H1_LOCKED_ACCESS_AUDIT.json", access_audit)


__all__ = [
    "evaluate_trajectory_search_pool",
    "evaluate_existing_trajectory_evidence",
    "evaluate_locked_selections",
    "evaluate_manual_baselines",
    "evaluate_budget_canaries",
    "write_locked_outputs",
]
