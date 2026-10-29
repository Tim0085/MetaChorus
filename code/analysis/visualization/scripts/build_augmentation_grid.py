from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from m9_common import (
    DATASETS,
    GRID_VALUES,
    MODEL_SEEDS,
    canonical_hash,
    read_csv,
    require_exact_grid,
    safe_float_token,
    write_csv,
)


GRID_FIELDS = (
    "dataset", "p_e", "p_f", "model_seed", "set_id", "set_hash",
    "frozen_config_sha", "endpoint", "physical_run_id", "group_id",
    "estimated_seconds", "run_status", "row_identity",
)


def build_grid(frozen_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_dataset = {str(row["dataset"]).casefold(): row for row in frozen_rows}
    if set(by_dataset) != set(DATASETS):
        raise ValueError("frozen config table must contain exactly five datasets")
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        frozen = by_dataset[dataset]
        for p_e in GRID_VALUES:
            for p_f in GRID_VALUES:
                for seed in MODEL_SEEDS:
                    basis = {
                        "dataset": dataset,
                        "p_e": p_e,
                        "p_f": p_f,
                        "model_seed": seed,
                        "set_id": frozen["final_set_id"],
                        "set_hash": frozen["final_set_hash"],
                        "frozen_config_sha": frozen["config_sha256"],
                        "endpoint": 3000,
                    }
                    basis["physical_run_id"] = (
                        f"M9-VIS-P1-GRID-{dataset.upper()}-PE{safe_float_token(p_e)}-"
                        f"PF{safe_float_token(p_f)}-S{seed}"
                    )
                    basis["group_id"] = f"{dataset}::{seed}"
                    basis["estimated_seconds"] = float(frozen["historical_median_seconds"])
                    basis["run_status"] = "PENDING"
                    basis["row_identity"] = canonical_hash(basis)
                    rows.append(basis)
    require_exact_grid(rows)
    return rows


def lpt_schedule(grid_rows: Sequence[Mapping[str, Any]], gpu_count: int = 4) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in grid_rows:
        group = groups.setdefault(
            str(row["group_id"]),
            {
                "group_id": row["group_id"],
                "dataset": row["dataset"],
                "model_seed": int(row["model_seed"]),
                "row_count": 0,
                "estimated_seconds": 0.0,
            },
        )
        group["row_count"] += 1
        group["estimated_seconds"] += float(row["estimated_seconds"])
    loads = [0.0] * gpu_count
    scheduled = []
    for group in sorted(groups.values(), key=lambda item: (-item["estimated_seconds"], item["group_id"])):
        gpu_slot = min(range(gpu_count), key=lambda index: (loads[index], index))
        group = dict(group)
        group["preferred_gpu_slot"] = gpu_slot
        group["planned_runtime_domain"] = f"M9-VIS-P1-GPU{gpu_slot}"
        group["scheduled_start_seconds"] = loads[gpu_slot]
        loads[gpu_slot] += group["estimated_seconds"]
        group["scheduled_end_seconds"] = loads[gpu_slot]
        scheduled.append(group)
    return scheduled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output-grid", required=True)
    parser.add_argument("--output-schedule", required=True)
    args = parser.parse_args()
    grid = build_grid(read_csv(args.frozen_config))
    schedule = lpt_schedule(grid)
    write_csv(args.output_grid, grid, GRID_FIELDS)
    write_csv(args.output_schedule, schedule)
    print({"status": "PASS", "grid_rows": len(grid), "groups": len(schedule)})


if __name__ == "__main__":
    main()
