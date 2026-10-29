"""Evaluation-only label cache, physically separate from training inputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from metachorus_v53.audit.common import (
    atomic_write_json,
    atomic_write_text,
    build_checksums,
    canonical_json_hash,
    sha256_file,
)
from .array_identity import array_summary, write_npy_deterministic


LABEL_CACHE_SCHEMA = "metachorus.evaluation-label-cache.v1"


def write_label_cache(
    output_parent: str | Path,
    labels: Any,
    *,
    source_data_sha256: str,
    target_node_type: str,
    dataset_id: str = "acm",
    target_node_ordering: str = "source_processed_tensor_order",
) -> dict[str, Any]:
    summary = array_summary(labels)
    if len(summary["shape"]) != 1:
        raise ValueError("evaluation label array must be one-dimensional")
    normalized_dataset = dataset_id.casefold()
    if not normalized_dataset or not normalized_dataset.replace("_", "").isalnum():
        raise ValueError("dataset_id must be a non-empty safe identifier")
    array_name = f"{target_node_type.casefold()}_y.npy"
    histogram = {
        str(key): int(value)
        for key, value in sorted(
            Counter(map(int, labels.tolist())).items(), key=lambda item: item[0]
        )
    }
    identity_basis = {
        "schema_version": LABEL_CACHE_SCHEMA,
        "dataset_id": normalized_dataset,
        "source_data_sha256": source_data_sha256,
        "target_node_type": target_node_type,
        "target_node_ordering": target_node_ordering,
        "array_semantic_sha256": summary["semantic_sha256"],
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "class_histogram": histogram,
    }
    label_cache_id = canonical_json_hash(identity_basis)
    root = Path(output_parent) / f"{normalized_dataset}_label_cache_{label_cache_id[:24]}"
    if root.exists():
        raise FileExistsError(f"append-only label cache exists: {root}")
    (root / "labels").mkdir(parents=True)
    (root / "metadata").mkdir()
    array_record = write_npy_deterministic(root / "labels" / array_name, labels)
    evaluation_identity = {
        **identity_basis,
        "evaluation_only_class_count": len(histogram),
        "label_cache_id": label_cache_id,
        "local_only": True,
        "training_mount_allowed": False,
    }
    atomic_write_json(
        root / "metadata" / "evaluation_identity.json", evaluation_identity
    )
    files = {
        f"labels/{array_name}": array_record,
        "metadata/evaluation_identity.json": {
            "raw_file_sha256": sha256_file(
                root / "metadata" / "evaluation_identity.json"
            ),
            "size_bytes": (
                root / "metadata" / "evaluation_identity.json"
            ).stat().st_size,
        },
    }
    manifest = {
        "schema_version": LABEL_CACHE_SCHEMA,
        "label_cache_id": label_cache_id,
        "dataset_id": normalized_dataset,
        "source_data_sha256": source_data_sha256,
        "target_node_type": target_node_type,
        "target_node_ordering": target_node_ordering,
        "class_count": len(histogram),
        "class_histogram": histogram,
        "files": files,
        "local_only": True,
        "training_mount_allowed": False,
    }
    atomic_write_json(root / "LABEL_CACHE_MANIFEST.json", manifest)
    atomic_write_text(
        root / "LABEL_CACHE_CHECKSUMS.sha256",
        build_checksums(
            root,
            exclude_names={"LABEL_CACHE_CHECKSUMS.sha256"},
        ),
    )
    return {
        "root": root,
        "label_cache_id": label_cache_id,
        "manifest": manifest,
        "manifest_sha256": sha256_file(root / "LABEL_CACHE_MANIFEST.json"),
        "checksums_sha256": sha256_file(root / "LABEL_CACHE_CHECKSUMS.sha256"),
        "array_semantic_sha256": summary["semantic_sha256"],
        "shape": summary["shape"],
        "dtype": summary["dtype"],
    }
