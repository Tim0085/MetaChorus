from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from metachorus_v53.m8_f1.common import envelope, read_json, write_json_once
from metachorus_v53.m8_f1.training import _gpu_runtime, run_dynamic_worker


def _verify_source_identity(
    repo_root: Path, source_commit: str, source_tree: str
) -> None:
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
    ):
        raise ValueError("M8-F1 source commit/tree must be lowercase 40-hex identities")
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    actual_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != source_commit or actual_tree != source_tree:
        raise RuntimeError("M8-F1 installed source commit/tree identity changed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen M8-F1 remote GPU worker")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cache-parent", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument(
        "--logical-stage", required=True, choices=("CANARY", "FINAL10")
    )
    parser.add_argument("--worker-index", required=True, type=int, choices=range(4))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-pci-bus-id", required=True)
    parser.add_argument("--log-path", required=True, type=Path)
    args = parser.parse_args()
    if args.repo_root.resolve() != REPO_ROOT.resolve():
        parser.error("remote worker must execute from the installed frozen source")
    if re.fullmatch(r"[0-9a-f]{32}", args.launch_nonce) is None:
        parser.error("launch nonce must contain 32 lowercase hexadecimal characters")
    _verify_source_identity(
        args.repo_root.resolve(strict=True), args.source_commit, args.source_tree
    )

    runtime = _gpu_runtime(enforce_target_gpu=True)
    if (
        runtime["gpu_uuid"] != args.expected_gpu_uuid
        or runtime["gpu_pci_bus_id"].casefold()
        != args.expected_gpu_pci_bus_id.casefold()
        or os.environ.get("M8_F1_LAUNCH_NONCE") != args.launch_nonce
    ):
        raise RuntimeError("M8-F1 remote worker GPU/PCI/nonce handshake failed")

    result = args.result_root.resolve(strict=True)
    seal = read_json(args.seal)
    launch_root = result / "launches" / args.logical_stage.casefold() / args.launch_nonce
    started = envelope(
        "metachorus.v53.m8-f1.remote-worker-started.v1",
        "STARTED",
        {
            "logical_stage": args.logical_stage,
            "worker_index": args.worker_index,
            "worker_count": 4,
            "launch_nonce": args.launch_nonce,
            "plan_seal_identity": seal["identity"],
            "source_commit": args.source_commit,
            "source_tree": args.source_tree,
            **runtime,
            "labels_mounted_during_training": False,
            "one_process_per_gpu": True,
            "ddp_used": False,
            "data_parallel_used": False,
            "torchrun_used": False,
        },
    )
    write_json_once(launch_root / f"worker_{args.worker_index}_STARTED.json", started)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            args.log_path.open("a", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            receipt = run_dynamic_worker(
                repo_root=args.repo_root,
                cache_parent=args.cache_parent,
                result_root=result,
                plan_path=args.plan,
                seal_path=args.seal,
                logical_stage=args.logical_stage,
                worker_index=args.worker_index,
                worker_count=4,
                source_commit=args.source_commit,
                source_tree=args.source_tree,
                launch_nonce=args.launch_nonce,
                enforce_target_gpu=True,
            )
            print(
                {"status": receipt["status"], "identity": receipt["identity"]},
                flush=True,
            )
    except BaseException as exc:
        failure = envelope(
            "metachorus.v53.m8-f1.remote-worker-fatal.v1",
            "FAILED_PRESERVED",
            {
                "logical_stage": args.logical_stage,
                "worker_index": args.worker_index,
                "launch_nonce": args.launch_nonce,
                "plan_seal_identity": seal["identity"],
                "source_commit": args.source_commit,
                "source_tree": args.source_tree,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback_sha256": hashlib.sha256(
                    traceback.format_exc().encode("utf-8")
                ).hexdigest(),
                "scientific_metric_triggered_retry": False,
            },
        )
        write_json_once(launch_root / f"worker_{args.worker_index}_FATAL.json", failure)
        raise


if __name__ == "__main__":
    main()
