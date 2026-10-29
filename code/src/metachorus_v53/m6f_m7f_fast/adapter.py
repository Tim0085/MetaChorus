"""Strict numeric-only training adapter for fast-stage sanitized caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from metachorus_v53.audit.common import (
    canonical_json_hash,
    load_checksums,
    read_json,
    sha256_file,
)
from metachorus_v53.data.array_identity import array_summary, load_npy_safe

from .cache import CACHE_SCHEMA


class FastSanitizedAdapter:
    """Expose structural inputs without a supervised-information API."""

    __slots__ = (
        "_root",
        "_manifest",
        "_dataset_schema",
        "_relation_schema",
        "_meta_path_schema",
        "_path_library",
        "_all_sets",
        "_fixed_sets",
        "_data_identity",
        "_projection_identity",
    )

    def __init__(self, cache_root: str | Path, *, expected_dataset: str | None = None) -> None:
        root = Path(cache_root).resolve()
        if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("fast sanitized cache must be a regular, symlink-free directory")
        complete = read_json(root / "SANITIZED_CACHE_COMPLETE.json")
        manifest = read_json(root / "SANITIZED_CACHE_MANIFEST.json")
        if (
            complete.get("schema_version") != CACHE_SCHEMA
            or manifest.get("schema_version") != CACHE_SCHEMA
            or complete.get("status") != "COMPLETE"
            or manifest.get("status") != "SEALED_LABEL_FREE_CACHE"
            or complete.get("cache_identity") != manifest.get("cache_identity")
            or complete.get("supervised_payloads_present") is not False
        ):
            raise ValueError("fast sanitized cache closure is invalid")
        identity_basis = manifest.get("identity_basis")
        if not isinstance(identity_basis, Mapping) or canonical_json_hash(identity_basis) != manifest.get("cache_identity"):
            raise ValueError("fast sanitized cache identity is not canonical")
        if expected_dataset is not None and manifest.get("dataset") != expected_dataset:
            raise ValueError("fast sanitized cache dataset mismatch")
        if (
            sha256_file(root / "SANITIZED_CACHE_MANIFEST.json") != complete.get("manifest_sha256")
            or sha256_file(root / "SANITIZED_CACHE_CHECKSUMS.sha256") != complete.get("checksums_sha256")
        ):
            raise ValueError("fast sanitized cache seal hashes are invalid")
        checksums = load_checksums(root / "SANITIZED_CACHE_CHECKSUMS.sha256")
        expected_checksum_members = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in {"SANITIZED_CACHE_CHECKSUMS.sha256", "SANITIZED_CACHE_COMPLETE.json"}
        }
        if set(checksums) != expected_checksum_members or any(
            sha256_file(root / relative) != digest for relative, digest in checksums.items()
        ):
            raise ValueError("fast sanitized cache checksum inventory is invalid")
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("fast sanitized cache file manifest is missing")
        expected_payload_members = expected_checksum_members - {"SANITIZED_CACHE_MANIFEST.json"}
        if set(map(str, files)) != expected_payload_members:
            raise ValueError("fast sanitized cache manifest member set is invalid")
        for relative, row in files.items():
            source = root / str(relative)
            if (
                not source.is_file()
                or source.is_symlink()
                or sha256_file(source) != row.get("raw_file_sha256")
                or source.stat().st_size != int(row.get("size_bytes", -1))
            ):
                raise ValueError(f"fast sanitized cache member changed: {relative}")

        self._root = root
        self._manifest = manifest
        self._dataset_schema = read_json(root / "schema" / "dataset_schema.json")
        self._relation_schema = read_json(root / "schema" / "relation_schema.json")
        self._meta_path_schema = read_json(root / "schema" / "meta_path_schema.json")
        self._path_library = read_json(root / "candidates" / "path_library.json")
        self._all_sets = read_json(root / "candidates" / "all_sets.json")
        self._fixed_sets = read_json(root / "candidates" / "fixed_cardinality_sets.json")
        self._data_identity = read_json(root / "metadata" / "data_identity.json")
        self._projection_identity = read_json(root / "metadata" / "projection_identity.json")
        if (
            self._dataset_schema.get("dataset") != manifest.get("dataset")
            or self._path_library.get("path_library_identity")
            != identity_basis.get("path_library_identity")
            or self._all_sets.get("all_sets_identity") != identity_basis.get("all_sets_identity")
            or self._data_identity.get("structural_data_identity")
            != identity_basis.get("structural_data_identity")
        ):
            raise ValueError("fast sanitized cache documents do not bind the manifest")

    @property
    def cache_root(self) -> Path:
        return self._root

    @property
    def dataset(self) -> str:
        return str(self._manifest["dataset"])

    @property
    def cache_identity(self) -> str:
        return str(self._manifest["cache_identity"])

    def _array(self, relative_path: str) -> np.ndarray:
        files = self._manifest["files"]
        if relative_path not in files:
            raise KeyError(relative_path)
        source = self._root / relative_path
        array = load_npy_safe(source)
        expected = files[relative_path]
        observed = array_summary(array)
        if (
            sha256_file(source) != expected.get("raw_file_sha256")
            or observed["semantic_sha256"] != expected.get("semantic_sha256")
            or observed["shape"] != expected.get("shape")
            or observed["dtype"] != expected.get("dtype")
            or observed["finite"] is not True
        ):
            raise ValueError(f"fast sanitized array changed: {relative_path}")
        return array

    def features(self) -> torch.Tensor:
        relative = str(self._dataset_schema["target_feature_array"])
        array = self._array(relative)
        if array.dtype != np.float32 or array.ndim != 2:
            raise ValueError("target features must be rank-2 float32")
        return torch.from_numpy(array.copy())

    def projected_views(self, path_ids: Sequence[str]) -> Mapping[str, torch.Tensor]:
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("duplicate fast-stage path requested")
        lookup = {str(row["path_id"]): row for row in self._path_library["paths"]}
        result: dict[str, torch.Tensor] = {}
        for path_id in path_ids:
            if path_id not in lookup:
                raise KeyError(path_id)
            row = lookup[path_id]
            array = self._array(str(row["array"]))
            if array.dtype != np.int64 or array.ndim != 2 or array.shape != (2, int(row["edge_count"])):
                raise ValueError(f"invalid fast-stage projected view: {path_id}")
            result[str(path_id)] = torch.from_numpy(array.copy())
        return result

    def raw_relations(self) -> Mapping[str, torch.Tensor]:
        return {
            str(row["relation_id"]): torch.from_numpy(self._array(str(row["array"])).copy())
            for row in self._relation_schema["relations"]
        }

    def schema(self) -> Mapping[str, Any]:
        return {
            "dataset": dict(self._dataset_schema),
            "relations": dict(self._relation_schema),
            "meta_paths": dict(self._meta_path_schema),
        }

    def candidate_bank(self) -> Mapping[str, Any]:
        return {
            "path_library": dict(self._path_library),
            "all_sets": dict(self._all_sets),
            "fixed_cardinality_sets": dict(self._fixed_sets),
        }

    def data_identity(self) -> Mapping[str, Any]:
        return {
            "cache_identity": self.cache_identity,
            "source": dict(self._data_identity),
            "projection": dict(self._projection_identity),
        }


def adapter_compliance() -> dict[str, Any]:
    required = ("features", "projected_views", "schema", "candidate_bank", "data_identity")
    forbidden = ("labels", "y", "num_classes", "class_count", "train_mask", "test_mask")
    return {
        "schema_version": "metachorus.v53.m6f-m7f.adapter-compliance.v1",
        "implementation": "FastSanitizedAdapter",
        "required_methods_present": {name: callable(getattr(FastSanitizedAdapter, name, None)) for name in required},
        "forbidden_methods_present": {name: hasattr(FastSanitizedAdapter, name) for name in forbidden},
        "complete": all(callable(getattr(FastSanitizedAdapter, name, None)) for name in required)
        and not any(hasattr(FastSanitizedAdapter, name) for name in forbidden),
    }


__all__ = ["FastSanitizedAdapter", "adapter_compliance"]
