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

    font = configure_matplotlib()
    source_path = root / "outputs" / "AUGMENTATION_GRID_SUMMARY.csv"
    rows = read_csv(source_path)
    anchor_path = root / "qa" / "AUGMENTATION_FROZEN_ANCHOR_POINTS.csv"
    anchor_rows = read_csv(anchor_path)
    if len(rows) != 125 or any(int(row["seed_count"]) != 3 for row in rows):
        raise RuntimeError("Micro-F1 source must contain 125 complete three-seed cells")
    fig, axes = plt.subplots(2, 3, figsize=(183.0 / 25.4, 105.0 / 25.4), facecolor="white")
    ranges: dict[str, dict[str, float]] = {}
    for index, dataset in enumerate(DATASETS):
        ax = axes.flat[index]
        values = metric_grid(rows, dataset, "Micro_F1_sample_SD")
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
        colorbar.set_label("Micro-F1 sample SD", fontsize=7)
        ranges[dataset] = {"vmin": vmin, "vmax": vmax, "range": vmax - vmin}
    axes.flat[5].axis("off")
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.10, top=0.93, wspace=0.65, hspace=0.60)
    prefix = root / "outputs" / "AugmentationSensitivity_SD_heatmaps"
    output = save_artwork(fig, prefix, width_mm=183.0, height_mm=105.0)
    plt.close(fig)
    metadata = {
        "schema_version": "metachorus.m9-vis-p1-fix1.micro-f1-sd-heatmaps.v1",
        "status": "PASS",
        "scientific_rerun": False,
        "source_file": "outputs/AUGMENTATION_GRID_SUMMARY.csv",
        "source_file_sha256": sha256_file(source_path),
        "metric": "Micro_F1_sample_SD",
        "interpolation_used": False,
        "interpolation": "nearest",
        "origin": "lower",
        "color_mapping": "viridis; independent per-dataset scale",
        "cross_panel_color_comparison_allowed": False,
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
        "figure_height_mm": 105.0,
        "actual_font": font,
        "static_publication_contract": STATIC_PUBLICATION_CONTRACT,
        "output": output,
    }
    write_json(root / "outputs" / "AugmentationSensitivity_SD_heatmaps_metadata_FIX1.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    result = render(Path(args.project_root).resolve())
    print(json.dumps({"status": result["status"], "metric": result["metric"]}, sort_keys=True))


if __name__ == "__main__":
    main()
