from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183.0
DATASETS = ("acm", "dblp", "aminer", "imdb", "freebase")
DISPLAY_NAMES = {
    "acm": "ACM",
    "dblp": "DBLP",
    "aminer": "AMiner",
    "imdb": "IMDb",
    "freebase": "Freebase",
}
GRID_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
MODEL_SEEDS = (3276665818, 1910071238, 610349320)
SOURCE_METRICS = ("Micro_F1", "Macro_F1", "NMI", "ARI")
DERIVED_METRICS = ("F_cls", "F_clu", "F_joint")
ALL_METRICS = SOURCE_METRICS + DERIVED_METRICS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def data_row_count(path: Path) -> int:
    if path.suffix.casefold() != ".csv":
        raise ValueError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def strict_float(value: str | float | int, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def float_text(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"cannot serialize non-finite float: {value!r}")
    return repr(float(value))


def validate_source_results(rows: Sequence[Mapping[str, str]]) -> None:
    required = (
        "dataset", "p_e", "p_f", "model_seed", "Micro_F1", "Macro_F1", "NMI", "ARI",
        "set_id", "set_hash", "frozen_config_sha", "run_id", "GPU_UUID", "runtime_domain",
        "training_seconds", "representation_sha256", "status",
    )
    if len(rows) != 375:
        raise RuntimeError(f"source run table must contain 375 rows, found {len(rows)}")
    keys: set[tuple[str, float, float, int]] = set()
    by_dataset: dict[str, int] = defaultdict(int)
    by_cell: dict[tuple[str, float, float], int] = defaultdict(int)
    for index, row in enumerate(rows, start=1):
        missing = [field for field in required if field not in row]
        if missing:
            raise RuntimeError(f"row {index} missing fields: {missing}")
        dataset = row["dataset"]
        pe = strict_float(row["p_e"], name="p_e")
        pf = strict_float(row["p_f"], name="p_f")
        seed = int(row["model_seed"])
        if dataset not in DATASETS or pe not in GRID_VALUES or pf not in GRID_VALUES or seed not in MODEL_SEEDS:
            raise RuntimeError(f"row {index} outside frozen grid contract")
        if row["status"] != "PASS":
            raise RuntimeError(f"row {index} is not PASS")
        for metric in SOURCE_METRICS:
            strict_float(row[metric], name=metric)
        strict_float(row["training_seconds"], name="training_seconds")
        key = (dataset, pe, pf, seed)
        if key in keys:
            raise RuntimeError(f"duplicate source run key: {key}")
        keys.add(key)
        by_dataset[dataset] += 1
        by_cell[(dataset, pe, pf)] += 1
    if set(by_dataset) != set(DATASETS) or any(by_dataset[name] != 75 for name in DATASETS):
        raise RuntimeError(f"dataset row counts invalid: {dict(by_dataset)}")
    expected_cells = {(dataset, pe, pf) for dataset in DATASETS for pe in GRID_VALUES for pf in GRID_VALUES}
    if set(by_cell) != expected_cells or any(by_cell[key] != 3 for key in expected_cells):
        raise RuntimeError("source grid does not contain exactly three seeds for all 125 cells")


def derived_values(row: Mapping[str, str]) -> dict[str, float]:
    micro = strict_float(row["Micro_F1"], name="Micro_F1")
    macro = strict_float(row["Macro_F1"], name="Macro_F1")
    nmi = strict_float(row["NMI"], name="NMI")
    ari = strict_float(row["ARI"], name="ARI")
    f_cls = (micro + macro) / 2.0
    f_clu = (nmi + ari) / 2.0
    return {"F_cls": f_cls, "F_clu": f_clu, "F_joint": (f_cls + f_clu) / 2.0}


def group_metric_rows(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, float, float], list[Mapping[str, str]]]:
    groups: dict[tuple[str, float, float], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], float(row["p_e"]), float(row["p_f"]))].append(row)
    return groups


def summary_stats(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("each grid cell must contain exactly three finite values")
    return {
        "mean": statistics.mean(values),
        "sample_SD": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "seed_count": len(values),
    }


def metric_grid(rows: Sequence[Mapping[str, str]], dataset: str, field: str) -> Any:
    import numpy as np

    subset = [row for row in rows if row["dataset"] == dataset]
    keys = {(float(row["p_e"]), float(row["p_f"])) for row in subset}
    expected = {(pe, pf) for pe in GRID_VALUES for pf in GRID_VALUES}
    if len(subset) != 25 or keys != expected:
        raise RuntimeError(f"{dataset}/{field} is not a complete 5x5 grid")
    result = np.empty((5, 5), dtype=float)
    for y, pf in enumerate(GRID_VALUES):
        for x, pe in enumerate(GRID_VALUES):
            matches = [row for row in subset if float(row["p_e"]) == pe and float(row["p_f"]) == pf]
            value = strict_float(matches[0][field], name=field)
            result[y, x] = value
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite grid: {dataset}/{field}")
    return result


def configure_matplotlib() -> dict[str, str]:
    import matplotlib as mpl
    from matplotlib import font_manager

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
    })
    resolved_path = font_manager.findfont(font_manager.FontProperties(family=["Arial", "Helvetica", "DejaVu Sans"]), fallback_to_default=True)
    actual_name = font_manager.FontProperties(fname=resolved_path).get_name()
    return {"requested": "Arial, Helvetica, DejaVu Sans", "actual_name": actual_name, "actual_file_name": Path(resolved_path).name}


def export_publication_figure(fig: Any, prefix: Path, *, width_mm: float, height_mm: float) -> dict[str, Any]:
    from PIL import Image

    if abs(width_mm - FIGURE_WIDTH_MM) > 1e-9:
        raise ValueError(f"publication width must be {FIGURE_WIDTH_MM} mm")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH, forward=True)
    common = {"facecolor": "white", "edgecolor": "white", "transparent": False, "bbox_inches": None, "pad_inches": 0}
    fig.savefig(prefix.with_suffix(".pdf"), dpi=600, **common)
    fig.savefig(prefix.with_suffix(".svg"), dpi=600, **common)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, **common)
    with tempfile.NamedTemporaryFile(prefix="fix1_", suffix=".png", dir=prefix.parent, delete=False) as handle:
        temporary_png = Path(handle.name)
    try:
        fig.savefig(temporary_png, dpi=600, **common)
        with Image.open(temporary_png) as image:
            rgb = image.convert("RGB")
            rgb.save(prefix.with_suffix(".tiff"), format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    finally:
        temporary_png.unlink(missing_ok=True)
    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "savefig_dpi_pdf_svg": 600,
        "png_dpi": 300,
        "tiff_dpi": 600,
        "tiff_mode": "RGB",
        "tiff_compression": "tiff_lzw",
        "files": {
            suffix: {"sha256": sha256_file(prefix.with_suffix(suffix)), "size_bytes": prefix.with_suffix(suffix).stat().st_size}
            for suffix in (".pdf", ".svg", ".png", ".tiff")
        },
    }


def validate_embedding_coordinates(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != 16228:
        raise RuntimeError(f"embedding coordinates must contain 16228 rows, found {len(rows)}")
    panels: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        if row["panel"] not in {"a", "b", "c", "d"}:
            raise RuntimeError(f"unexpected panel: {row['panel']}")
        if int(row["visualization_seed"]) != 2025:
            raise RuntimeError("FIX1 coordinate source must contain only primary visualization seed 2025")
        for field in ("tsne_x_raw", "tsne_y_raw", "tsne_x_display", "tsne_y_display"):
            strict_float(row[field], name=field)
        panels[row["panel"]].append((row["node_id"], row["label"]))
    if set(panels) != {"a", "b", "c", "d"} or any(len(panels[name]) != 4057 for name in panels):
        raise RuntimeError("each embedding panel must contain 4057 rows")
    anchor = panels["a"]
    if any(panels[name] != anchor for name in ("b", "c", "d")):
        raise RuntimeError("node IDs or labels are not aligned across embedding panels")


def files_manifest(paths: Iterable[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(relative_to).as_posix().encode("utf-8")):
        rows.append({
            "relative_path": path.relative_to(relative_to).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows

