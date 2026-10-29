from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")

from export_publication_artwork import save_artwork
from fix1_common import (
    DATASETS,
    DISPLAY_NAMES,
    GRID_VALUES,
    configure_matplotlib,
    metric_grid,
    read_csv,
    sha256_file,
    write_json,
)


RASTER_DPI = 600
FIGURES = (
    ("F_clu_mean", "AugmentationSensitivity_Fclu_mean_heatmaps", r"$F_{clu}$ mean"),
    ("F_joint_mean", "AugmentationSensitivity_Fjoint_mean_heatmaps", r"$F_{joint}$ mean"),
    ("F_clu_sample_SD", "AugmentationSensitivity_Fclu_SD_heatmaps", r"$F_{clu}$ sample SD"),
    ("F_joint_sample_SD", "AugmentationSensitivity_Fjoint_SD_heatmaps", r"$F_{joint}$ sample SD"),
)
STATIC_PUBLICATION_CONTRACT = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "formats": (".svg", ".pdf", ".tiff", ".png"),
    "dpi": RASTER_DPI,
}


def render(root: Path) -> dict[str, object]:
    import matplotlib.pyplot as plt

    font = configure_matplotlib()
    source_path = root / "outputs" / "AUGMENTATION_MULTITASK_GRID_SUMMARY.csv"
    rows = read_csv(source_path)
    anchor_path = root / "qa" / "AUGMENTATION_FROZEN_ANCHOR_POINTS.csv"
    anchor_rows = read_csv(anchor_path)
    if len(rows) != 125 or any(int(row["seed_count"]) != 3 for row in rows):
        raise RuntimeError("multi-task summary must contain 125 complete three-seed cells")
    figure_metadata: dict[str, object] = {}
    for metric_field, prefix_name, colorbar_label in FIGURES:
        fig, axes = plt.subplots(2, 3, figsize=(183.0 / 25.4, 105.0 / 25.4), facecolor="white")
        ranges: dict[str, dict[str, float]] = {}
        for index, dataset in enumerate(DATASETS):
            ax = axes.flat[index]
            values = metric_grid(rows, dataset, metric_field)
            vmin, vmax = float(values.min()), float(values.max())
            image = ax.imshow(
                values, origin="lower", interpolation="nearest", cmap="viridis", rasterized=True,
                extent=(0.0, 1.0, 0.0, 1.0), aspect="equal", vmin=vmin, vmax=vmax,
            )
            ax.set_xticks(GRID_VALUES, [f"{value:.1f}" for value in GRID_VALUES], fontsize=6)
            ax.set_yticks(GRID_VALUES, [f"{value:.1f}" for value in GRID_VALUES], fontsize=6)
            ax.set_xlabel(r"$p_e$", fontsize=7)
            ax.set_ylabel(r"$p_f$", fontsize=7)
            ax.set_title(f"({chr(97 + index)}) {DISPLAY_NAMES[dataset]}", fontsize=7.5, pad=3)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
            colorbar.ax.tick_params(labelsize=6)
            colorbar.set_label(colorbar_label, fontsize=7)
            ranges[dataset] = {"vmin": vmin, "vmax": vmax, "range": vmax - vmin}
        axes.flat[5].axis("off")
        fig.subplots_adjust(left=0.055, right=0.90, bottom=0.10, top=0.93, wspace=0.65, hspace=0.60)
        prefix = root / "outputs" / prefix_name
        output = save_artwork(fig, prefix, width_mm=183.0, height_mm=105.0)
        plt.close(fig)
        figure_metadata[prefix_name] = {
            "metric": metric_field,
            "colorbar_label": colorbar_label,
            "interpolation_used": False,
            "interpolation": "nearest",
            "origin": "lower",
            "cross_panel_color_comparison_allowed": False,
            "per_panel_ranges": ranges,
            "output": output,
        }
    metadata = {
        "schema_version": "metachorus.m9-vis-p1-fix1.multitask-heatmaps.v1",
        "status": "PASS",
        "scientific_rerun": False,
        "source_file": "outputs/AUGMENTATION_MULTITASK_GRID_SUMMARY.csv",
        "source_file_sha256": sha256_file(source_path),
        "source_cells": len(rows),
        "color_mapping": "viridis; independent per-dataset scale",
        "frozen_anchor_outside_grid": {
            row["dataset"]: {
                "p_e": float(row["p_e"]),
                "p_f": float(row["p_f"]),
                "outside_surface_grid": row["anchor_in_surface_grid"].casefold() != "true",
            }
            for row in anchor_rows
        },
        "frozen_anchor_file_sha256": sha256_file(anchor_path),
        "figure_width_mm": 183.0,
        "figure_height_mm": 105.0,
        "actual_font": font,
        "static_publication_contract": STATIC_PUBLICATION_CONTRACT,
        "figures": figure_metadata,
    }
    write_json(root / "outputs" / "AugmentationSensitivity_Multitask_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    result = render(Path(args.project_root).resolve())
    print(json.dumps({"status": result["status"], "figures": len(result["figures"])}, sort_keys=True))


if __name__ == "__main__":
    main()
