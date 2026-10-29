from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")

from export_publication_artwork import save_artwork
from fix1_common import (
    configure_matplotlib,
    read_csv,
    sha256_file,
    validate_embedding_coordinates,
    write_json,
)


RASTER_DPI = 600
PANEL_TITLES = {
    "a": "Manual",
    "b": "MetaChorus",
    "c": "Classification-only selection",
    "d": "Clustering-only selection",
}
STATIC_PUBLICATION_CONTRACT = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "formats": (".svg", ".pdf", ".tiff", ".png"),
    "dpi": RASTER_DPI,
}


def render(root: Path) -> dict[str, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    font = configure_matplotlib()
    coordinates_path = root / "outputs" / "Embedding_DBLP_4Panels_coordinates.csv"
    diagnostics_path = root / "outputs" / "Embedding_DBLP_4Panels_diagnostics.csv"
    qa_path = root / "outputs" / "EMBEDDING_TSNE_QA_BY_VISUAL_SEED.csv"
    rows = read_csv(coordinates_path)
    diagnostics_rows = read_csv(diagnostics_path)
    validate_embedding_coordinates(rows)
    if len(diagnostics_rows) != 4 or {row["panel"] for row in diagnostics_rows} != {"a", "b", "c", "d"}:
        raise RuntimeError("embedding diagnostics must contain exactly four panels")
    diagnostics = {row["panel"]: row for row in diagnostics_rows}
    qa_rows = read_csv(qa_path)
    if len(qa_rows) != 12 or {int(row["visualization_seed"]) for row in qa_rows} != {2025, 2026, 2027}:
        raise RuntimeError("frozen t-SNE QA must contain four panels by three visualization seeds")

    by_panel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_panel[row["panel"]].append(row)
    labels_unique = sorted({int(row["label"]) for row in rows})
    cmap = mpl.colormaps["viridis"].resampled(len(labels_unique))
    color_by_label = {label: cmap(index) for index, label in enumerate(labels_unique)}
    color_mapping = {f"Class {label}": mpl.colors.to_hex(color_by_label[label], keep_alpha=False) for label in labels_unique}
    outputs: dict[str, object] = {}
    for show_sc, suffix in ((True, ""), (False, "_NoSC")):
        fig, axes = plt.subplots(2, 2, figsize=(183.0 / 25.4, 142.0 / 25.4), facecolor="white")
        for ax, panel in zip(axes.flat, ("a", "b", "c", "d")):
            panel_rows = by_panel[panel]
            x = np.asarray([float(row["tsne_x_display"]) for row in panel_rows], dtype=float)
            y = np.asarray([float(row["tsne_y_display"]) for row in panel_rows], dtype=float)
            colors = [color_by_label[int(row["label"])] for row in panel_rows]
            ax.scatter(x, y, s=2.2, c=colors, alpha=0.72, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            title = f"({panel}) {PANEL_TITLES[panel]}"
            if show_sc:
                value = float(diagnostics[panel]["silhouette_label_highdim_euclidean"])
                title += f"\nSC_HD = {value:.4f}"
            ax.set_title(title, fontsize=7.5, pad=2)
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=4, color=color_by_label[label], label=f"Class {label}")
            for label in labels_unique
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.012), fontsize=7)
        fig.subplots_adjust(left=0.018, right=0.982, top=0.955, bottom=0.095, wspace=0.05, hspace=0.105)
        prefix = root / "outputs" / f"Embedding_DBLP_4Panels{suffix}"
        outputs[prefix.name] = save_artwork(fig, prefix, width_mm=183.0, height_mm=142.0)
        plt.close(fig)

    metadata = {
        "schema_version": "metachorus.m9-vis-p1-fix1.embedding-figure.v1",
        "status": "PASS",
        "scientific_rerun": False,
        "tsne_rerun": False,
        "source_coordinate_file": "outputs/Embedding_DBLP_4Panels_coordinates.csv",
        "source_coordinate_sha256": sha256_file(coordinates_path),
        "source_diagnostics_sha256": sha256_file(diagnostics_path),
        "source_tsne_qa_sha256": sha256_file(qa_path),
        "coordinate_rows": len(rows),
        "nodes_per_panel": 4057,
        "primary_visualization_seed": 2025,
        "panel_labels": PANEL_TITLES,
        "source_roles_preserved": {panel: by_panel[panel][0]["role"] for panel in ("a", "b", "c", "d")},
        "semantic_label_names_status": "NOT_AVAILABLE",
        "legend_labels": [f"Class {label}" for label in labels_unique],
        "color_mapping": color_mapping,
        "silhouette_definition": "original high-dimensional embedding space; Euclidean metric; ground-truth labels",
        "orientation_warning": "Absolute direction, rotation, and scale are not scientifically comparable across t-SNE panels.",
        "figure_width_mm": 183.0,
        "figure_height_mm": 142.0,
        "frozen_anchor_outside_grid": "NOT_APPLICABLE",
        "actual_font": font,
        "static_publication_contract": STATIC_PUBLICATION_CONTRACT,
        "outputs": outputs,
    }
    write_json(root / "outputs" / "Embedding_DBLP_4Panels_metadata_FIX1.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    result = render(Path(args.project_root).resolve())
    print(json.dumps({"status": result["status"], "coordinate_rows": result["coordinate_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
