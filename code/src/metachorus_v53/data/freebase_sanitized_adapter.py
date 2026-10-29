"""Strict label-free Freebase adapter over the deterministic NPY cache."""

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
from .array_identity import array_summary, load_npy_safe
from .label_free_adapter import CONTRACT
from .sanitized_cache_schema import assert_no_forbidden_fields, assert_safe_relative_path


FREEBASE_CACHE_SCHEMA = "metachorus.freebase-sanitized-cache.v1"


class FreebaseSanitizedAdapter:
    """Expose only target features, structural views, and structural identities."""

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

    def __init__(self, cache_root: str | Path) -> None:
        root = Path(cache_root).resolve()
        complete = read_json(root / "SANITIZED_CACHE_COMPLETE.json")
        manifest = read_json(root / "SANITIZED_CACHE_MANIFEST.json")
        if complete.get("schema_version") != FREEBASE_CACHE_SCHEMA:
            raise ValueError("unsupported Freebase sanitized-cache schema")
        if complete.get("status") != "COMPLETE":
            raise ValueError("Freebase sanitized cache is not complete")
        if manifest.get("schema_version") != FREEBASE_CACHE_SCHEMA:
            raise ValueError("unsupported Freebase sanitized-cache manifest")
        if complete.get("cache_identity") != manifest.get("cache_identity"):
            raise ValueError("Freebase sanitized cache identity mismatch")
        identity_basis = manifest.get("identity_basis")
        if not isinstance(identity_basis, Mapping):
            raise ValueError("Freebase cache identity basis is missing")
        if canonical_json_hash(identity_basis) != manifest.get("cache_identity"):
            raise ValueError("Freebase cache identity is not canonical")
        manifest_path = root / "SANITIZED_CACHE_MANIFEST.json"
        checksums_path = root / "SANITIZED_CACHE_CHECKSUMS.sha256"
        if (
            sha256_file(manifest_path) != complete.get("manifest_sha256")
            or sha256_file(checksums_path) != complete.get("checksums_sha256")
        ):
            raise ValueError("Freebase cache completion hashes are invalid")
        if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("Freebase sanitized cache may not contain symlinks")
        checksums = load_checksums(checksums_path)
        expected_members = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name
            not in {"SANITIZED_CACHE_CHECKSUMS.sha256", "SANITIZED_CACHE_COMPLETE.json"}
        }
        if set(checksums) != expected_members or any(
            sha256_file(root / relative) != digest
            for relative, digest in checksums.items()
        ):
            raise ValueError("Freebase cache checksum inventory is invalid")
        files = manifest.get("files")
        arrays = identity_basis.get("arrays")
        if (
            not isinstance(files, Mapping)
            or not isinstance(arrays, Mapping)
            or set(files) != set(arrays)
            or any(
                files[relative].get("semantic_sha256") != arrays[relative]
                for relative in files
            )
        ):
            raise ValueError("Freebase cache array identity basis is inconsistent")

        self._root = root
        self._manifest = manifest
        self._dataset_schema = read_json(root / "schema" / "dataset_schema.json")
        self._relation_schema = read_json(root / "schema" / "relation_schema.json")
        self._meta_path_schema = read_json(root / "schema" / "meta_path_schema.json")
        self._path_library = read_json(root / "candidates" / "path_library.json")
        self._all_sets = read_json(root / "candidates" / "all_sets.json")
        self._fixed_sets = read_json(
            root / "candidates" / "fixed_cardinality_sets.json"
        )
        self._data_identity = read_json(root / "metadata" / "data_identity.json")
        self._projection_identity = read_json(
            root / "metadata" / "projection_identity.json"
        )
        assert_no_forbidden_fields(
            (
                complete,
                manifest,
                self._dataset_schema,
                self._relation_schema,
                self._meta_path_schema,
                self._path_library,
                self._all_sets,
                self._fixed_sets,
                self._data_identity,
                self._projection_identity,
            )
        )
        if self._dataset_schema.get("dataset_id") != "freebase":
            raise ValueError("Freebase cache dataset identity is invalid")
        if self._dataset_schema.get("target_node_type") != "movie":
            raise ValueError("Freebase cache target node type is invalid")
        if int(self._path_library.get("effective_path_count", -1)) != 12:
            raise ValueError("Freebase cache must contain exactly 12 paths")
        if int(self._all_sets.get("set_count", -1)) != 220:
            raise ValueError("Freebase cache must bind exactly 220 logical sets")
        roles = {row.get("path_set_role") for row in self._fixed_sets.get("sets", [])}
        if roles != {"Manual", "Automatic"}:
            raise ValueError("Freebase cache must bind Manual and Automatic sets")
        fixed_by_role = {
            str(row["path_set_role"]): row for row in self._fixed_sets["sets"]
        }
        if (
            self._projection_identity.get("projection_identity")
            != identity_basis.get("projection_identity")
            or self._projection_identity.get("path_library_sha256")
            != identity_basis.get("path_library_sha256")
            or self._projection_identity.get("all_sets_sha256")
            != identity_basis.get("all_sets_sha256")
            or self._path_library.get("source_path_library_sha256")
            != identity_basis.get("path_library_sha256")
            or self._all_sets.get("all_sets_sha256")
            != identity_basis.get("all_sets_sha256")
            or fixed_by_role["Manual"].get("candidate_set_hash")
            != identity_basis.get("manual_set_sha256")
            or fixed_by_role["Automatic"].get("candidate_set_hash")
            != identity_basis.get("automatic_set_sha256")
            or self._data_identity.get("feature_semantic_sha256")
            != arrays.get("features/movie_x.npy")
        ):
            raise ValueError("Freebase cache metadata does not bind its identity basis")

    @property
    def cache_root(self) -> Path:
        return self._root

    def _array(self, relative_path: str) -> np.ndarray:
        assert_safe_relative_path(relative_path)
        files = self._manifest.get("files")
        if not isinstance(files, Mapping) or relative_path not in files:
            raise KeyError(relative_path)
        path = self._root / relative_path
        array = load_npy_safe(path)
        expected = files[relative_path]
        observed = array_summary(array)
        if (
            sha256_file(path) != expected.get("raw_file_sha256")
            or observed["semantic_sha256"] != expected.get("semantic_sha256")
            or observed["shape"] != expected.get("shape")
            or observed["dtype"] != expected.get("dtype")
            or observed["finite"] is not True
        ):
            raise ValueError(f"Freebase sanitized array changed: {relative_path}")
        return array

    def features(self) -> torch.Tensor:
        array = self._array("features/movie_x.npy")
        if array.dtype != np.float32 or array.ndim != 2:
            raise ValueError("Freebase movie features must be rank-2 float32")
        return torch.from_numpy(array.copy())

    def projected_views(
        self, path_ids: Sequence[str]
    ) -> Mapping[str, torch.Tensor]:
        lookup = {row["path_id"]: row for row in self._path_library["paths"]}
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("duplicate Freebase path ID requested")
        requested: dict[str, torch.Tensor] = {}
        for path_id in path_ids:
            if path_id not in lookup:
                raise KeyError(path_id)
            row = lookup[path_id]
            array = self._array(str(row["array"]))
            if (
                array.dtype != np.int64
                or array.ndim != 2
                or array.shape[0] != 2
                or int(array.shape[1]) != int(row["edge_count"])
            ):
                raise ValueError(f"invalid projected Freebase view: {path_id}")
            requested[str(path_id)] = torch.from_numpy(array.copy())
        return requested

    def raw_relations(self) -> Mapping[str, torch.Tensor]:
        return {
            row["relation_id"]: torch.from_numpy(
                self._array(str(row["array"])).copy()
            )
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
            "cache_identity": self._manifest["cache_identity"],
            "source": dict(self._data_identity),
            "projection": dict(self._projection_identity),
        }


def freebase_adapter_compliance() -> dict[str, Any]:
    required = list(CONTRACT["required_methods"])
    forbidden = list(CONTRACT["forbidden_methods"])
    return {
        "schema_version": "metachorus.dataset-adapter-compliance.v1",
        "adapter_id": "freebase",
        "implementation": "FreebaseSanitizedAdapter",
        "required_methods_present": {
            name: callable(getattr(FreebaseSanitizedAdapter, name, None))
            for name in required
        },
        "forbidden_methods_present": {
            name: hasattr(FreebaseSanitizedAdapter, name) for name in forbidden
        },
        "complete": all(
            callable(getattr(FreebaseSanitizedAdapter, name, None))
            for name in required
        )
        and not any(hasattr(FreebaseSanitizedAdapter, name) for name in forbidden),
    }


__all__ = [
    "FREEBASE_CACHE_SCHEMA",
    "FreebaseSanitizedAdapter",
    "freebase_adapter_compliance",
]
