from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from m9_common import DATASETS, GRID_VALUES, MODEL_SEEDS, canonical_hash, load_json, read_csv, require_exact_grid, write_csv


def collect_results(project_root: Path) -> list[dict[str, Any]]:
    matrix = [row for row in read_csv(project_root / "qa" / "VISUALIZATION_GPU_RUN_MATRIX_PENDING.csv") if row["run_kind"] == "AUGMENTATION_GRID"]
    results = []
    for row in matrix:
        run_root = project_root / "runs" / row["physical_run_id"] / "attempt_00"
        marker = load_json(run_root / "M9_VIS_P1_RUN_COMPLETE.json")
        evaluation = load_json(run_root / "M9_VIS_P1_EVALUATION.json")
        if marker["status"] != "PASS" or evaluation["status"] != "PASS" or marker["row_identity"] != row["row_identity"]:
            raise RuntimeError(f"invalid PASS evidence: {row['physical_run_id']}")
        results.append({
            "dataset": row["dataset"],
            "p_e": float(row["p_e"]),
            "p_f": float(row["p_f"]),
            "model_seed": int(row["model_seed"]),
            "Micro_F1": float(evaluation["Micro_F1"]),
            "Macro_F1": float(evaluation["Macro_F1"]),
            "NMI": float(evaluation["NMI"]),
            "ARI": float(evaluation["ARI"]),
            "set_id": row["set_id"],
            "set_hash": row["set_hash"],
            "frozen_config_sha": row["frozen_config_sha"],
            "run_id": row["physical_run_id"],
            "GPU_UUID": marker["gpu"]["gpu_uuid"],
            "runtime_domain": marker["gpu"]["runtime_domain"],
            "training_seconds": marker["model_total_seconds"],
            "representation_sha256": marker["representation_sha256"],
            "status": "PASS",
        })
    require_exact_grid(results)
    return results


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["p_e"], row["p_f"]), []).append(row)
    expected = {(d, pe, pf) for d in DATASETS for pe in GRID_VALUES for pf in GRID_VALUES}
    if set(groups) != expected:
        raise RuntimeError("grid aggregation keys are incomplete")
    summary = []
    for key in sorted(groups):
        values = groups[key]
        if len(values) != 3 or {row["model_seed"] for row in values} != set(MODEL_SEEDS):
            raise RuntimeError(f"grid cell lacks exactly three PASS seeds: {key}")
        micro = [row["Micro_F1"] for row in values]
        summary.append({
            "dataset": key[0], "p_e": key[1], "p_f": key[2],
            "Micro_F1_mean": statistics.mean(micro),
            "Micro_F1_sample_SD": statistics.stdev(micro),
            "Micro_F1_minimum": min(micro),
            "Micro_F1_maximum": max(micro),
            "seed_count": 3,
        })
    ranges = []
    for dataset in DATASETS:
        subset = [row for row in summary if row["dataset"] == dataset]
        minimum = min(row["Micro_F1_mean"] for row in subset)
        maximum = max(row["Micro_F1_mean"] for row in subset)
        best = [{"p_e": row["p_e"], "p_f": row["p_f"]} for row in subset if row["Micro_F1_mean"] == maximum]
        ranges.append({
            "dataset": dataset,
            "minimum_mean_Micro_F1": minimum,
            "maximum_mean_Micro_F1": maximum,
            "range": maximum - minimum,
            "best_grid_cells": json.dumps(best, sort_keys=True),
        })
    return summary, ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    runs = collect_results(root)
    summary, ranges = aggregate(runs)
    write_csv(root / "outputs" / "AUGMENTATION_RESULTS_BY_RUN.csv", runs)
    write_csv(root / "outputs" / "AUGMENTATION_GRID_SUMMARY.csv", summary)
    write_csv(root / "outputs" / "AUGMENTATION_RANGE_SUMMARY.csv", ranges)
    print({"status": "PASS", "run_rows": len(runs), "grid_cells": len(summary)})


if __name__ == "__main__":
    main()
