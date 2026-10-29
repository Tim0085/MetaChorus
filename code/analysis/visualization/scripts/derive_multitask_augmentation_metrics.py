from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from fix1_common import (
    ALL_METRICS,
    DATASETS,
    DERIVED_METRICS,
    GRID_VALUES,
    SOURCE_METRICS,
    derived_values,
    float_text,
    group_metric_rows,
    load_json,
    read_csv,
    sha256_file,
    summary_stats,
    validate_source_results,
    write_csv,
    write_json,
)


SOURCE_FIELDS = [
    "dataset", "p_e", "p_f", "model_seed", "Micro_F1", "Macro_F1", "NMI", "ARI",
    "set_id", "set_hash", "frozen_config_sha", "run_id", "GPU_UUID", "runtime_domain",
    "training_seconds", "representation_sha256", "status",
]
DERIVED_FIELDS = [
    "dataset", "p_e", "p_f", "model_seed", "Micro_F1", "Macro_F1", "NMI", "ARI",
    "F_cls", "F_clu", "F_joint", "set_id", "set_hash", "frozen_config_sha", "run_id",
    "GPU_UUID", "runtime_domain", "training_seconds", "representation_sha256", "status",
]


def freeze_scientific_files(root: Path, contract: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    failures = []
    for item in contract["frozen_scientific_files"]:
        path = root / item["relative_path"]
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if path.suffix.casefold() == ".csv" and path.is_file():
            parsed = read_csv(path)
            row_count = len(parsed)
            column_count = len(parsed[0]) if parsed else 0
        else:
            row_count = ""
            column_count = ""
        status = "PASS" if actual == item["sha256"] else "FAIL"
        rows.append({
            "relative_path": item["relative_path"],
            "expected_sha256": item["sha256"],
            "actual_sha256": actual,
            "size_bytes": path.stat().st_size if path.is_file() else "",
            "row_count": row_count,
            "column_count": column_count,
            "byte_identity_status": status,
            "scientific_role": item["scientific_role"],
            "included_in_final_package": str(bool(item.get("include_byte_identical"))).lower(),
        })
        if status != "PASS":
            failures.append(item["relative_path"])
        expected_rows = item.get("rows")
        if expected_rows is not None and row_count != expected_rows:
            failures.append(f"{item['relative_path']}:rows={row_count}")
    write_csv(root / "qa_fix1" / "M9_VIS_P1_FIX1_SCIENTIFIC_FILE_FREEZE.csv", rows)
    if failures:
        raise RuntimeError(f"scientific freeze failed: {failures}")
    return rows


def derive(root: Path) -> dict[str, object]:
    contract = load_json(root / "configs" / "fix1_contract.json")
    freeze_rows = freeze_scientific_files(root, contract)
    source_path = root / "outputs" / "AUGMENTATION_RESULTS_BY_RUN.csv"
    source_rows = read_csv(source_path)
    validate_source_results(source_rows)

    derived_rows: list[dict[str, str]] = []
    max_formula_error = 0.0
    for source in source_rows:
        values = derived_values(source)
        row = {field: source[field] for field in SOURCE_FIELDS}
        row.update({name: float_text(values[name]) for name in DERIVED_METRICS})
        ordered = {field: row[field] for field in DERIVED_FIELDS}
        derived_rows.append(ordered)
        f_cls_expected = (float(source["Micro_F1"]) + float(source["Macro_F1"])) / 2.0
        f_clu_expected = (float(source["NMI"]) + float(source["ARI"])) / 2.0
        f_joint_expected = (f_cls_expected + f_clu_expected) / 2.0
        max_formula_error = max(
            max_formula_error,
            abs(float(ordered["F_cls"]) - f_cls_expected),
            abs(float(ordered["F_clu"]) - f_clu_expected),
            abs(float(ordered["F_joint"]) - f_joint_expected),
        )
    derived_path = root / "outputs" / "AUGMENTATION_MULTITASK_RESULTS_BY_RUN.csv"
    write_csv(derived_path, derived_rows, DERIVED_FIELDS)

    groups = group_metric_rows(derived_rows)
    if len(groups) != 125:
        raise RuntimeError(f"derived grid must contain 125 cells, found {len(groups)}")
    summary_fields = ["dataset", "p_e", "p_f"]
    for metric in ALL_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_sample_SD", f"{metric}_minimum", f"{metric}_maximum"])
    summary_fields.append("seed_count")
    summary_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for pe in GRID_VALUES:
            for pf in GRID_VALUES:
                group = groups[(dataset, pe, pf)]
                output: dict[str, str] = {"dataset": dataset, "p_e": float_text(pe), "p_f": float_text(pf)}
                for metric in ALL_METRICS:
                    stats = summary_stats([float(row[metric]) for row in group])
                    for suffix in ("mean", "sample_SD", "minimum", "maximum"):
                        output[f"{metric}_{suffix}"] = float_text(float(stats[suffix]))
                output["seed_count"] = "3"
                summary_rows.append(output)
    summary_path = root / "outputs" / "AUGMENTATION_MULTITASK_GRID_SUMMARY.csv"
    write_csv(summary_path, summary_rows, summary_fields)

    parent_micro = read_csv(root / "outputs" / "AUGMENTATION_GRID_SUMMARY.csv")
    parent_by_key = {(row["dataset"], float(row["p_e"]), float(row["p_f"])): row for row in parent_micro}
    micro_parity_max_error = 0.0
    for row in summary_rows:
        key = (row["dataset"], float(row["p_e"]), float(row["p_f"]))
        parent = parent_by_key[key]
        mapping = {
            "Micro_F1_mean": "Micro_F1_mean",
            "Micro_F1_sample_SD": "Micro_F1_sample_SD",
            "Micro_F1_minimum": "Micro_F1_minimum",
            "Micro_F1_maximum": "Micro_F1_maximum",
        }
        for left, right in mapping.items():
            micro_parity_max_error = max(micro_parity_max_error, abs(float(row[left]) - float(parent[right])))
        if row["seed_count"] != parent["seed_count"]:
            raise RuntimeError(f"Micro-F1 seed-count mismatch for {key}")
    if micro_parity_max_error > 1e-12:
        raise RuntimeError(f"Micro-F1 parent summary parity failed: {micro_parity_max_error}")

    range_fields = ["dataset", "metric", "minimum_mean", "maximum_mean", "range", "all_best_grid_cells", "all_worst_grid_cells"]
    range_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        subset = [row for row in summary_rows if row["dataset"] == dataset]
        for metric in ALL_METRICS:
            field = f"{metric}_mean"
            values = [float(row[field]) for row in subset]
            minimum = min(values)
            maximum = max(values)
            best = sorted((float(row["p_e"]), float(row["p_f"])) for row in subset if float(row[field]) == maximum)
            worst = sorted((float(row["p_e"]), float(row["p_f"])) for row in subset if float(row[field]) == minimum)
            encode = lambda cells: json.dumps([{"p_e": pe, "p_f": pf} for pe, pf in cells], separators=(",", ":"))
            range_rows.append({
                "dataset": dataset,
                "metric": metric,
                "minimum_mean": float_text(minimum),
                "maximum_mean": float_text(maximum),
                "range": float_text(maximum - minimum),
                "all_best_grid_cells": encode(best),
                "all_worst_grid_cells": encode(worst),
            })
    range_path = root / "outputs" / "AUGMENTATION_MULTITASK_RANGE_SUMMARY.csv"
    write_csv(range_path, range_rows, range_fields)

    extreme_fields = ["dataset", "metric", "p_e", "p_f", "seed_values_json", "mean", "sample_SD", "minimum", "maximum", "tie_count"]
    extreme_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        subset = [row for row in summary_rows if row["dataset"] == dataset]
        for metric in ("Micro_F1", "NMI", "ARI", "F_clu", "F_joint"):
            sd_field = f"{metric}_sample_SD"
            maximum_sd = max(float(row[sd_field]) for row in subset)
            tied = [row for row in subset if math.isclose(float(row[sd_field]), maximum_sd, rel_tol=0.0, abs_tol=1e-15)]
            for row in tied:
                key = (dataset, float(row["p_e"]), float(row["p_f"]))
                seed_values = [
                    {"model_seed": int(item["model_seed"]), "value": float(item[metric])}
                    for item in sorted(groups[key], key=lambda item: int(item["model_seed"]))
                ]
                extreme_rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "p_e": row["p_e"],
                    "p_f": row["p_f"],
                    "seed_values_json": json.dumps(seed_values, separators=(",", ":")),
                    "mean": row[f"{metric}_mean"],
                    "sample_SD": row[sd_field],
                    "minimum": row[f"{metric}_minimum"],
                    "maximum": row[f"{metric}_maximum"],
                    "tie_count": str(len(tied)),
                })
    extremes_path = root / "outputs" / "AUGMENTATION_SEED_DISPERSION_EXTREMES.csv"
    write_csv(extremes_path, extreme_rows, extreme_fields)

    result = {
        "schema_version": "metachorus.m9-vis-p1-fix1.multitask-derivation.v1",
        "status": "PASS",
        "scientific_rerun": False,
        "source_file": "outputs/AUGMENTATION_RESULTS_BY_RUN.csv",
        "source_file_sha256": sha256_file(source_path),
        "source_rows": len(source_rows),
        "derived_rows": len(derived_rows),
        "derived_cells": len(summary_rows),
        "formula_max_absolute_error": max_formula_error,
        "micro_f1_parent_summary_max_absolute_error": micro_parity_max_error,
        "frozen_files_checked": len(freeze_rows),
        "outputs": {
            path.name: sha256_file(path)
            for path in (derived_path, summary_path, range_path, extremes_path)
        },
    }
    write_json(root / "qa_fix1" / "M9_VIS_P1_FIX1_DERIVATION_RECEIPT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    result = derive(Path(args.project_root).resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
