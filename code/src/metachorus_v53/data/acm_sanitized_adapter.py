"""Label-free ACM adapter backed only by validated JSON and safe NPY files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from metachorus_v53.audit.common import read_json
from .array_identity import load_npy_safe
from .label_free_adapter import CONTRACT, LabelFreeDatasetAdapter


class ACMSanitizedAdapter:
    """Read-only adapter with no evaluation-information API."""

    __slots__ = ("_root", "_manifest", "_dataset_schema", "_path_library", "_sets")

    def __init__(self, cache_root: str | Path) -> None:
        root = Path(cache_root).resolve()
        complete = read_json(root / "SANITIZED_CACHE_COMPLETE.json")
        if complete.get("status") != "COMPLETE":
            raise ValueError("sanitized cache is not complete")
        self._root = root
        self._manifest = read_json(root / "SANITIZED_CACHE_MANIFEST.json")
        self._dataset_schema = read_json(root / "schema" / "dataset_schema.json")
        self._path_library = read_json(root / "candidates" / "path_library.json")
        self._sets = read_json(root / "candidates" / "fixed_cardinality_sets.json")

    def features(self) -> torch.Tensor:
        array = load_npy_safe(self._root / "features" / "paper_x.npy")
        return torch.from_numpy(array.copy())

    def projected_views(
        self, path_ids: Sequence[str]
    ) -> Mapping[str, torch.Tensor]:
        lookup = {row["path_id"]: row for row in self._path_library["paths"]}
        requested: dict[str, torch.Tensor] = {}
        for path_id in path_ids:
            if path_id not in lookup:
                raise KeyError(path_id)
            array = load_npy_safe(self._root / lookup[path_id]["array"])
            requested[path_id] = torch.from_numpy(array.copy())
        return requested

    def schema(self) -> Mapping[str, Any]:
        return {
            "dataset": dict(self._dataset_schema),
            "relations": read_json(self._root / "schema" / "relation_schema.json"),
            "meta_paths": read_json(self._root / "schema" / "meta_path_schema.json"),
        }

    def candidate_bank(self) -> Mapping[str, Any]:
        return {
            "path_library": dict(self._path_library),
            "fixed_cardinality_sets": dict(self._sets),
        }

    def data_identity(self) -> Mapping[str, Any]:
        return {
            "cache_identity": self._manifest["cache_identity"],
            "source": read_json(self._root / "metadata" / "data_identity.json"),
            "projection": read_json(
                self._root / "metadata" / "projection_identity.json"
            ),
        }


def adapter_compliance() -> dict[str, Any]:
    required = list(CONTRACT["required_methods"])
    forbidden = list(CONTRACT["forbidden_methods"])
    return {
        "schema_version": "metachorus.dataset-adapter-compliance.v1",
        "adapter_id": "acm",
        "implementation": "ACMSanitizedAdapter",
        "required_methods_present": {
            name: callable(getattr(ACMSanitizedAdapter, name, None)) for name in required
        },
        "forbidden_methods_present": {
            name: hasattr(ACMSanitizedAdapter, name) for name in forbidden
        },
        "complete": all(
            callable(getattr(ACMSanitizedAdapter, name, None)) for name in required
        )
        and not any(hasattr(ACMSanitizedAdapter, name) for name in forbidden),
    }


from .freebase_sanitized_adapter import FreebaseSanitizedAdapter


class DBLPSanitizedAdapter:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("DBLP sanitized adapter is outside M0-B")


class IMDBSanitizedAdapter:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("IMDb sanitized adapter is outside M0-B")


class AMinerSanitizedAdapter:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("AMiner sanitized adapter is outside M0-B")


__all__ = [
    "ACMSanitizedAdapter",
    "FreebaseSanitizedAdapter",
    "DBLPSanitizedAdapter",
    "IMDBSanitizedAdapter",
    "AMinerSanitizedAdapter",
    "adapter_compliance",
]
