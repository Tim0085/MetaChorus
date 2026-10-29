from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE = "M9-VIS-P1"
RUN_MODES = {"PRECHECK_ONLY", "EXECUTE_GPU"}
DATASETS = ("acm", "dblp", "aminer", "imdb", "freebase")
MODEL_SEEDS = (3276665818, 1910071238, 610349320)
GRID_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=target.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=target.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def parse_sha256_sidecar(path: str | Path) -> tuple[str, str]:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    parts = text.split(maxsplit=1)
    if len(parts) != 2 or len(parts[0]) != 64:
        raise ValueError("invalid sha256 sidecar")
    return parts[0].casefold(), parts[1].strip().lstrip("*")


def require_taskbook_identity(taskbook: Path, sidecar: Path) -> str:
    recorded, filename = parse_sha256_sidecar(sidecar)
    actual = sha256_file(taskbook)
    if filename != taskbook.name or recorded != actual:
        raise RuntimeError("taskbook identity mismatch")
    return actual


def run_text(command: Sequence[str]) -> tuple[int, str]:
    result = subprocess.run(
        list(command), check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode, result.stdout.strip()


def git_identity(root: str | Path) -> dict[str, Any]:
    root = str(Path(root).resolve())
    rc1, commit = run_text(["git", "-C", root, "rev-parse", "HEAD"])
    rc2, tree = run_text(["git", "-C", root, "rev-parse", "HEAD^{tree}"])
    rc3, status = run_text(["git", "-C", root, "status", "--porcelain"])
    if rc1 or rc2 or rc3:
        raise RuntimeError(f"cannot attest git identity: {root}")
    return {"root": root, "commit": commit, "tree": tree, "clean": not bool(status)}


def package_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return None


def no_gpu_attestation() -> dict[str, Any]:
    rc, output = run_text(["nvidia-smi", "-L"])
    nvidia_lines = [line for line in output.splitlines() if line.lstrip().startswith("GPU ")]
    torch_count: int | None = None
    torch_available: bool | None = None
    try:
        import torch

        torch_count = int(torch.cuda.device_count())
        torch_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    proven = len(nvidia_lines) == 0 and (torch_count in (0, None))
    return {
        "status": "PASS" if proven else "FAIL",
        "nvidia_smi_returncode": rc,
        "nvidia_smi_gpu_count": len(nvidia_lines),
        "torch_cuda_device_count": torch_count,
        "torch_cuda_is_available": torch_available,
        "CUDA_VISIBLE_DEVICES": visible,
        "precheck_no_gpu_proven": proven,
    }


def npy_metadata(path: str | Path, *, require_finite: bool = True) -> dict[str, Any]:
    import numpy as np

    source = Path(path)
    array = np.load(source, mmap_mode="r", allow_pickle=False)
    finite = bool(np.isfinite(array).all()) if require_finite else None
    return {
        "path": str(source.resolve()),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
        "finite": finite,
    }


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"existing link targets unexpected file: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to overwrite existing input: {link}")
    link.symlink_to(target.resolve())


def safe_float_token(value: float) -> str:
    return f"{int(round(value * 10)):02d}"


def require_exact_grid(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (dataset, p_e, p_f, seed)
        for dataset in DATASETS
        for p_e in GRID_VALUES
        for p_f in GRID_VALUES
        for seed in MODEL_SEEDS
    }
    observed = {
        (
            str(row["dataset"]).casefold(),
            float(row["p_e"]),
            float(row["p_f"]),
            int(row["model_seed"]),
        )
        for row in rows
    }
    if len(rows) != 375 or observed != expected:
        raise ValueError("augmentation grid is not the exact 375-row contract")


def manifest_for_files(paths: Iterable[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths, key=lambda p: p.as_posix()):
        if path.is_file() and not path.is_symlink():
            rows.append({
                "relative_path": path.relative_to(relative_to).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows

