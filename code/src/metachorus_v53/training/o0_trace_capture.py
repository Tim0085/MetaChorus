"""RNG-neutral trace, checkpoint, and representation sidecar."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from metachorus_v53.audit.common import atomic_write_json, sha256_file
from .o0_checkpoint_io import (
    capture_rng_state,
    restore_rng_state,
    rng_state_sha256,
    save_checkpoint,
)


TRACE_FIELDS = (
    "update",
    "augmentation_id",
    "recon_structure_loss",
    "l_cn",
    "l_cv",
    "l_gl",
    "weight_0",
    "weight_1",
    "weight_2",
    "weight_3",
    "total_loss",
    "gradient_norm",
    "learning_rate",
    "step_wall_time_seconds",
    "gpu_event_time_ms",
    "gpu_allocated_bytes",
    "gpu_reserved_bytes",
    "cpu_rss_bytes",
    "view_mode",
    "view_bank_step_identity",
    "view_bank_trace_entry_count",
    "augmentation_sample_count",
    "encoder_forward_count",
    "projector_forward_count",
    "representation_cache_bytes",
    "s1_expected_encoder_forward_count",
    "pcpa_profile_id",
    "pcpa_rate_manifest_sha256",
    "pcpa_profile_registry_sha256",
    "pcpa_schedule_t",
    "pcpa_warmup_weight",
    "pcpa_edge_rate_mean",
    "pcpa_edge_rate_min",
    "pcpa_edge_rate_max",
    "pcpa_feature_rate_mean",
    "pcpa_feature_rate_min",
    "pcpa_feature_rate_max",
    "pcpa_operation_trace_update_hash",
    "pcpa_operation_trace_row_count",
    "pcpa_rng_state_before_update_sha256",
    "rng_state_sha256",
    "finite",
    "pcgrad_variant_id",
    "pcgrad_projected_scope",
    "pcgrad_raw_conflict_count",
    "pcgrad_raw_pair_count",
    "pcgrad_raw_conflict_rate",
    "pcgrad_raw_pair_cosines",
    "pcgrad_raw_conflict_pairs",
    "pcgrad_projection_order",
    "pcgrad_extra_wall_time_seconds",
    "pcgrad_baseline_shared_gradient_norm",
    "pcgrad_merged_shared_gradient_norm",
    "pcgrad_no_conflict_baseline_delta_norm",
)


class TraceWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        resume: bool = False,
        expected_last_update: int | None = None,
        durable_flush_interval: int = 1,
    ):
        if (
            isinstance(durable_flush_interval, bool)
            or not isinstance(durable_flush_interval, int)
            or durable_flush_interval < 1
        ):
            raise ValueError("durable_flush_interval must be a positive integer")
        self.path = Path(path)
        self.durable_flush_interval = durable_flush_interval
        self._pending_rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not resume:
            raise FileExistsError(self.path)
        if resume:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            with self.path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError("cannot resume an empty resource trace")
            updates = [int(row["update"]) for row in rows]
            if updates[0] != 1:
                raise ValueError("resumed resource trace must start at update 1")
            if updates != list(range(updates[0], updates[-1] + 1)):
                raise ValueError("resource trace updates are not contiguous")
            if (
                expected_last_update is not None
                and updates[-1] != int(expected_last_update)
            ):
                raise ValueError(
                    "resource trace/checkpoint mismatch: "
                    f"{updates[-1]} != {expected_last_update}"
                )
        self._stream = self.path.open(
            "a" if resume else "x",
            encoding="utf-8",
            newline="",
        )
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=list(TRACE_FIELDS),
            extrasaction="raise",
            lineterminator="\n",
        )
        if not resume:
            self._writer.writeheader()
            self._pending_rows = 1
            self.flush()

    def flush(self) -> None:
        """Durably persist all rows written since the previous flush."""
        if self._stream.closed or self._pending_rows == 0:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._pending_rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow(dict(row))
        self._pending_rows += 1
        if self._pending_rows >= self.durable_flush_interval:
            self.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self.flush()
            self._stream.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *unused: Any) -> None:
        self.close()


def _atomic_numpy_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_representation(
    destination: str | Path,
    *,
    model: torch.nn.Module,
    features: torch.Tensor,
    edge_indices: Sequence[torch.Tensor],
    combine: str,
) -> dict[str, Any]:
    path = Path(destination)
    rng_before = capture_rng_state()
    rng_hash_before = rng_state_sha256(rng_before)
    was_training = bool(model.training)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            if edge_indices:
                view_tensors = [
                    model.online_encoder(features, edge_index)
                    for edge_index in edge_indices
                ]
                if combine == "concat":
                    fused_tensor = torch.concat(view_tensors, dim=-1)
                elif combine == "mean":
                    fused_tensor = torch.stack(view_tensors).mean(dim=0)
                else:
                    raise ValueError(
                        f"unsupported representation combine: {combine}"
                    )
            else:
                
                
                
                
                view_tensors = []
                fused_tensor = model(features, edge_indices, combine)
            representation = (
                fused_tensor.detach().cpu().contiguous().numpy()
            )
            view_representations = [
                value.detach().cpu().contiguous().numpy()
                for value in view_tensors
            ]
        rng_after = capture_rng_state()
        rng_hash_after = rng_state_sha256(rng_after)
    finally:
        model.train(was_training)
        restore_rng_state(rng_before)
    elapsed = time.perf_counter() - started
    if rng_hash_before != rng_hash_after:
        raise RuntimeError("representation capture consumed training RNG")
    if rng_state_sha256() != rng_hash_before:
        raise RuntimeError("representation capture failed to restore RNG")
    _atomic_numpy_save(path, representation)
    view_receipts: list[dict[str, Any]] = []
    view_root = path.parent / "view_specific" / path.stem
    for index, view in enumerate(view_representations):
        view_path = view_root / f"view_{index:02d}.npy"
        _atomic_numpy_save(view_path, view)
        view_receipts.append(
            {
                "view_index": index,
                "path": view_path.relative_to(path.parent).as_posix(),
                "shape": list(view.shape),
                "dtype": str(view.dtype),
                "size_bytes": view_path.stat().st_size,
                "sha256": sha256_file(view_path),
            }
        )
    return {
        "path": path.name,
        "shape": list(representation.shape),
        "dtype": str(representation.dtype),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "capture_wall_time_seconds": elapsed,
        "detached": True,
        "no_grad": True,
        "labels_read": False,
        "downstream_evaluation_executed": False,
        "rng_unchanged": True,
        "view_specific": view_receipts,
        "view_specific_count": len(view_receipts),
    }


class O0Sidecar:
    def __init__(
        self,
        output_root: str | Path,
        *,
        checkpoint_schedule: Sequence[int],
        representation_schedule: Sequence[int] | None = None,
        metadata: Mapping[str, Any],
        combine: str,
        resume_receipt: str | Path | None = None,
        expected_resume_update: int | None = None,
    ):
        self.output_root = Path(output_root)
        normalized_checkpoints = [int(value) for value in checkpoint_schedule]
        normalized_representations = [
            int(value)
            for value in (
                checkpoint_schedule
                if representation_schedule is None
                else representation_schedule
            )
        ]
        for name, schedule in (
            ("checkpoint", normalized_checkpoints),
            ("representation", normalized_representations),
        ):
            if schedule != sorted(schedule) or len(schedule) != len(set(schedule)):
                raise ValueError(f"{name} schedule must be strictly increasing")
        self.checkpoint_schedule = frozenset(normalized_checkpoints)
        self.representation_schedule = frozenset(normalized_representations)
        
        self.schedule = self.checkpoint_schedule
        self.metadata = dict(metadata)
        self.combine = combine
        self.checkpoint_receipts: list[dict[str, Any]] = []
        self.representation_receipts: list[dict[str, Any]] = []
        self.io_rows: list[dict[str, Any]] = []
        if resume_receipt is not None:
            from metachorus_v53.audit.common import read_json

            prior = read_json(resume_receipt)
            if prior.get("schema_version") != (
                "metachorus.v53.o0-sidecar-capture.v1"
            ):
                raise ValueError("unsupported prior sidecar receipt")
            try:
                legacy_coupled_receipt = "representation_schedule" not in prior
                prior_schedule = [
                    int(value) for value in prior.get("checkpoint_schedule", [])
                ]
                prior_representation_schedule = [
                    int(value)
                    for value in prior.get(
                        "representation_schedule", prior_schedule
                    )
                ]
                prior_checkpoints = list(prior.get("checkpoints", []))
                prior_representations = list(prior.get("representations", []))
                prior_io_rows = list(prior.get("checkpoint_io", []))
                checkpoint_updates = [
                    int(row["update"]) for row in prior_checkpoints
                ]
                representation_updates = [
                    int(row["update"]) for row in prior_representations
                ]
                io_updates = [int(row["update"]) for row in prior_io_rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("malformed prior sidecar receipt") from exc
            if (
                not checkpoint_updates
                or prior_schedule != sorted(prior_schedule)
                or len(prior_schedule) != len(set(prior_schedule))
                or prior_representation_schedule
                != sorted(prior_representation_schedule)
                or len(prior_representation_schedule)
                != len(set(prior_representation_schedule))
                or checkpoint_updates != sorted(checkpoint_updates)
                or representation_updates != sorted(representation_updates)
                or io_updates != sorted(io_updates)
                or io_updates
                != sorted(set(checkpoint_updates) | set(representation_updates))
                or (
                    legacy_coupled_receipt
                    and checkpoint_updates != representation_updates
                )
                or len(checkpoint_updates) != len(set(checkpoint_updates))
                or len(representation_updates) != len(set(representation_updates))
                or len(io_updates) != len(set(io_updates))
            ):
                raise ValueError("prior sidecar receipt has incomplete captures")
            resume_update = checkpoint_updates[-1]
            if (
                expected_resume_update is not None
                and resume_update != int(expected_resume_update)
            ):
                raise ValueError("sidecar receipt endpoint does not match resume update")
            prior_completed_schedule = [
                update for update in prior_schedule if update <= resume_update
            ]
            current_completed_schedule = [
                update
                for update in sorted(self.checkpoint_schedule)
                if update <= resume_update
            ]
            prior_completed_representation_schedule = [
                update
                for update in prior_representation_schedule
                if update <= resume_update
            ]
            current_completed_representation_schedule = [
                update
                for update in sorted(self.representation_schedule)
                if update <= resume_update
            ]
            if (
                prior_completed_schedule != current_completed_schedule
                or checkpoint_updates
                != current_completed_schedule[-len(checkpoint_updates) :]
            ):
                raise ValueError(
                    "sidecar completed checkpoint schedule changed on resume"
                )
            if (
                prior_completed_representation_schedule
                != current_completed_representation_schedule
                or representation_updates
                != current_completed_representation_schedule[
                    -len(representation_updates) :
                ]
            ):
                raise ValueError(
                    "sidecar completed representation schedule changed on resume"
                )
            self.checkpoint_receipts.extend(prior_checkpoints)
            self.representation_receipts.extend(prior_representations)
            self.io_rows.extend(prior_io_rows)

    def maybe_capture(
        self,
        *,
        update: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        features: torch.Tensor,
        edge_indices: Sequence[torch.Tensor],
    ) -> None:
        update = int(update)
        checkpoint_due = update in self.checkpoint_schedule
        representation_due = update in self.representation_schedule
        if not checkpoint_due and not representation_due:
            return
        if checkpoint_due and any(
            int(row["update"]) == update for row in self.checkpoint_receipts
        ):
            raise RuntimeError(f"duplicate sidecar checkpoint update {update}")
        if representation_due and any(
            int(row["update"]) == update for row in self.representation_receipts
        ):
            raise RuntimeError(f"duplicate sidecar representation update {update}")
        checkpoint_path = self.output_root / "checkpoints" / f"update_{update:06d}.pt"
        representation_path = (
            self.output_root / "representations" / f"update_{update:06d}.npy"
        )
        checkpoint = None
        representation = None
        if checkpoint_due:
            checkpoint = save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                exact_update=update,
                metadata=self.metadata,
                scheduler=None,
            )
        if representation_due:
            try:
                representation = capture_representation(
                    representation_path,
                    model=model,
                    features=features,
                    edge_indices=edge_indices,
                    combine=self.combine,
                )
            except Exception:
                
                
                
                if checkpoint_due:
                    checkpoint_path.unlink(missing_ok=True)
                representation_path.unlink(missing_ok=True)
                view_root = (
                    representation_path.parent
                    / "view_specific"
                    / representation_path.stem
                )
                if view_root.is_dir():
                    shutil.rmtree(view_root)
                raise
        if checkpoint is not None:
            self.checkpoint_receipts.append(
                {
                    "update": checkpoint.update,
                    "path": checkpoint.path.name,
                    "size_bytes": checkpoint.size_bytes,
                    "sha256": checkpoint.sha256,
                    "save_wall_time_seconds": checkpoint.save_wall_time_seconds,
                    "hash_wall_time_seconds": checkpoint.hash_wall_time_seconds,
                }
            )
        if representation is not None:
            self.representation_receipts.append({"update": update, **representation})
        self.io_rows.append(
            {
                "update": update,
                "checkpoint_bytes": (
                    checkpoint.size_bytes if checkpoint is not None else 0
                ),
                "save_wall_time_seconds": (
                    checkpoint.save_wall_time_seconds
                    if checkpoint is not None
                    else ""
                ),
                "load_wall_time_seconds": "",
                "hash_wall_time_seconds": (
                    checkpoint.hash_wall_time_seconds
                    if checkpoint is not None
                    else ""
                ),
                "representation_bytes": (
                    representation["size_bytes"]
                    if representation is not None
                    else 0
                ),
                "trace_bytes": 0,
            }
        )

    def publish_receipt(
        self,
        *,
        filename: str = "SIDECAR_CAPTURE_RECEIPT.json",
    ) -> dict[str, Any]:
        result = {
            "schema_version": "metachorus.v53.o0-sidecar-capture.v1",
            "checkpoint_schedule": sorted(self.checkpoint_schedule),
            "representation_schedule": sorted(self.representation_schedule),
            "checkpoints": self.checkpoint_receipts,
            "representations": self.representation_receipts,
            "checkpoint_io": self.io_rows,
            "labels_read": False,
            "downstream_evaluation_executed": False,
        }
        atomic_write_json(
            self.output_root / filename,
            result,
        )
        return result
