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
    import numpy as np
    font = configure_matplotlib()
    source_path = root / "outputs" / "AUGMENTATION_GRID_SUMMARY.csv"
    rows = read_csv(source_path)
    anchor_path = root / "qa" / "AUGMENTATION_FROZEN_ANCHOR_POINTS.csv"
    anchor_rows = read_csv(anchor_path)
    if len(rows) != 125 or any(int(row["seed_count"]) != 3 for row in rows):
        raise RuntimeError("Micro-F1 source must contain 125 complete three-seed cells")
    fig = plt.figure(figsize=(183.0 / 25.4, 132.0 / 25.4), facecolor="white")
    slots = (
        (0.07, 0.525, 0.36, 0.425),
        (0.57, 0.525, 0.36, 0.425),
        (0.02, 0.045, 0.25, 0.405),
        (0.36, 0.045, 0.25, 0.405),
        (0.70, 0.045, 0.25, 0.405),
    )
    x, y = np.meshgrid(GRID_VALUES, GRID_VALUES)
    ranges: dict[str, dict[str, float]] = {}
    for index, (dataset, slot) in enumerate(zip(DATASETS, slots)):
        ax = fig.add_axes(slot, projection="3d")
        z = metric_grid(rows, dataset, "Micro_F1_mean")
        ax.plot_surface(
            x, y, z, cmap="viridis", edgecolor="black", linewidth=0.35,
            antialiased=True, rcount=5, ccount=5, rasterized=True,
        )
        ax.view_init(elev=34, azim=-60)
        ax.set_box_aspect((1.0, 1.0, 0.72))
        ax.set_xlabel(r"$p_e$", labelpad=-1)
        ax.set_ylabel(r"$p_f$", labelpad=-1)
        ax.set_zlabel("Micro-F1", labelpad=1)
        ax.set_xticks(GRID_VALUES)
        ax.set_yticks(GRID_VALUES)
        ax.tick_params(labelsize=6, pad=0)
        zmin, zmax = float(z.min()), float(z.max())
        pad = max((zmax - zmin) * 0.08, 0.001)
        ax.set_zlim(zmin - pad, zmax + pad)
        ax.set_title(f"({chr(97 + index)}) {DISPLAY_NAMES[dataset]}", fontsize=7.5, pad=-2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ranges[dataset] = {"vmin": zmin, "vmax": zmax, "range": zmax - zmin}
    prefix = root / "outputs" / "AugmentationSensitivity_5Datasets"
    output = save_artwork(fig, prefix, width_mm=183.0, height_mm=132.0)
    plt.close(fig)
    metadata = {
        "schema_version": "metachorus.m9-vis-p1-fix1.augmentation-surface.v1",
        "status": "PASS",
        "scientific_rerun": False,
        "source_file": "outputs/AUGMENTATION_GRID_SUMMARY.csv",
        "source_file_sha256": sha256_file(source_path),
        "source_cells": len(rows),
        "seed_count_per_cell": 3,
        "metric": "Micro_F1_mean",
        "grid_values": list(GRID_VALUES),
        "view": {"elev": 34, "azim": -60},
        "interpolation_used": False,
        "best_point_markers": False,
        "color_mapping": "viridis with black surface mesh",
        "cross_panel_slope_comparison_allowed": False,
        "per_panel_ranges": ranges,
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
        "figure_height_mm": 132.0,
        "actual_font": font,
        "static_publication_contract": STATIC_PUBLICATION_CONTRACT,
        "output": output,
    }
    write_json(root / "outputs" / "AugmentationSensitivity_metadata_FIX1.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    result = render(Path(args.project_root).resolve())
    print(json.dumps({"status": result["status"], "source_cells": result["source_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
