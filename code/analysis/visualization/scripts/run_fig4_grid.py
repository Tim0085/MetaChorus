from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from m9_common import canonical_hash, git_identity, load_json, read_csv, sha256_file


def parse_gpu_inventory(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        index, uuid, name, bus = [part.strip() for part in line.split(",", 3)]
        rows.append({"index": index, "uuid": uuid, "name": name, "pci_bus_id": bus})
    if len(rows) != 4 or any("4090" not in row["name"] for row in rows):
        raise RuntimeError("EXECUTE_GPU requires exactly four visible RTX 4090 GPUs")
    if len({row["uuid"] for row in rows}) != 4:
        raise RuntimeError("GPU UUIDs must be unique")
    return rows


def attest_four_4090() -> list[dict[str, str]]:
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("nvidia-smi attestation failed")
    return parse_gpu_inventory(result.stdout)


def validate_precheck(project_root: Path) -> dict[str, Any]:
    qa = project_root / "qa"
    seal = load_json(qa / "VISUALIZATION_PRECHECK_SEAL.json")
    decision = load_json(qa / "VISUALIZATION_PRECHECK_DECISION.json")
    if seal["status"] != "PRECHECK_COMPLETE_GPU_PENDING" or decision["status"] != "PRECHECK_COMPLETE_GPU_PENDING":
        raise RuntimeError("PRECHECK is not sealed for GPU execution")
    for member in seal["members"]:
        path = qa / member["filename"]
        if sha256_file(path) != member["sha256"]:
            raise RuntimeError(f"PRECHECK member identity mismatch: {path.name}")
    inventory = read_csv(qa / "VISUALIZATION_PRECHECK_INPUT_INVENTORY.csv")
    mismatches: dict[str, tuple[str, str]] = {}
    for row in inventory:
        path = project_root / row["relative_path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"sealed input unavailable: {row['relative_path']}")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            mismatches[row["relative_path"]] = (row["sha256"], actual)
    if mismatches:
        expected = {relative: old_sha for relative, (old_sha, _) in mismatches.items()}
        previous_identity = None
        applied = 0
        latest: dict[str, Any] | None = None
        for index in range(1, 100):
            correction_path = qa / f"VISUALIZATION_RUNTIME_ADAPTER_CORRECTION_{index:03d}.json"
            if not correction_path.is_file():
                break
            correction = load_json(correction_path)
            identity = correction.get("correction_identity")
            basis = dict(correction)
            basis.pop("correction_identity", None)
            if (
                correction.get("status") != "PASS"
                or correction.get("base_precheck_seal_identity") != seal["seal_identity"]
                or correction.get("base_correction_identity") != previous_identity
                or canonical_hash(basis) != identity
            ):
                raise RuntimeError(f"runtime adapter correction {index:03d} identity invalid")
            for relative_path, hashes in correction.get("changed_files", {}).items():
                if relative_path not in expected or hashes.get("old_sha256") != expected[relative_path]:
                    raise RuntimeError(f"runtime adapter correction chain mismatch: {relative_path}")
                expected[relative_path] = hashes["new_sha256"]
            previous_identity = identity
            applied += 1
            latest = correction
        if not applied or latest is None:
            raise RuntimeError(f"sealed input identity mismatch without correction: {sorted(mismatches)}")
        for relative_path, (_, actual_sha) in mismatches.items():
            if expected[relative_path] != actual_sha:
                raise RuntimeError(f"runtime adapter correction final hash mismatch: {relative_path}")
        matrix_member = next(item for item in seal["members"] if item["filename"] == "VISUALIZATION_GPU_RUN_MATRIX_PENDING.csv")
        if latest.get("gpu_matrix_sha256") != matrix_member["sha256"]:
            raise RuntimeError("runtime adapter correction changed the sealed GPU matrix")
    project_identity = load_json(qa / "VISUALIZATION_PROJECT_AND_STORAGE_IDENTITY.json")
    for expected in project_identity["source_git_identities"]:
        observed = git_identity(expected["root"])
        if observed != expected:
            raise RuntimeError(f"frozen source Git identity changed: {expected['root']}")
    return seal


def _run_slot(
    slot: int,
    rows: list[dict[str, str]],
    *,
    project_root: Path,
    sources_path: Path,
    gpu: dict[str, str],
) -> list[str]:
    completed: list[str] = []
    for row in rows:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
        env["M9_GPU_UUID"] = gpu["uuid"]
        env["M9_RUNTIME_DOMAIN"] = row["planned_runtime_domain"]
        command = [
            sys.executable,
            str(project_root / "scripts" / "run_augmentation_cell.py"),
            "--project-root", str(project_root),
            "--sources", str(sources_path),
            "--matrix", str(project_root / "qa" / "VISUALIZATION_GPU_RUN_MATRIX_PENDING.csv"),
            "--frozen-config", str(project_root / "qa" / "AUGMENTATION_FROZEN_CONFIG_BY_DATASET.csv"),
            "--physical-run-id", row["physical_run_id"],
            "--attempt-index", str(row.get("_attempt_index", 0)),
        ]
        result = subprocess.run(command, env=env, check=False)
        if result.returncode:
            raise RuntimeError(f"GPU slot {slot} failed at {row['physical_run_id']}")
        completed.append(row["physical_run_id"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--kind", choices=("ENGINEERING_CANARY", "ANCHOR_VALIDATION", "AUGMENTATION_GRID", "ALL"), default="ALL")
    parser.add_argument("--attempt-map")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    validate_precheck(project_root)
    gpus = attest_four_4090()
    matrix = read_csv(project_root / "qa" / "VISUALIZATION_GPU_RUN_MATRIX_PENDING.csv")
    kinds = {args.kind} if args.kind != "ALL" else {"ENGINEERING_CANARY", "ANCHOR_VALIDATION", "AUGMENTATION_GRID"}
    selected = [row for row in matrix if row["run_kind"] in kinds and row["run_status"] == "PENDING"]
    attempt_map: dict[str, int] = {}
    if args.attempt_map:
        attempt_rows = read_csv(args.attempt_map)
        attempt_map = {row["physical_run_id"]: int(row["attempt_index"]) for row in attempt_rows}
        if set(attempt_map) != {row["physical_run_id"] for row in selected}:
            raise RuntimeError("attempt map must cover exactly the selected rows")
    for row in selected:
        row["_attempt_index"] = attempt_map.get(row["physical_run_id"], 0)
    by_slot = {slot: [] for slot in range(4)}
    for row in selected:
        by_slot[int(row["preferred_gpu_slot"])].append(row)
    completed: list[str] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="m9-gpu") as executor:
        futures = {
            executor.submit(
                _run_slot,
                slot,
                rows,
                project_root=project_root,
                sources_path=Path(args.sources).resolve(),
                gpu=gpus[slot],
            ): slot
            for slot, rows in by_slot.items()
            if rows
        }
        for future in as_completed(futures):
            try:
                completed.extend(future.result())
            except Exception as error:
                failures.append(str(error))
    if failures:
        raise RuntimeError(f"GPU execution paused after worker failure: {failures}")
    print({"status": "PASS", "executed_rows": len(completed), "kind": args.kind, "parallel_gpu_workers": len([rows for rows in by_slot.values() if rows])})


if __name__ == "__main__":
    main()
