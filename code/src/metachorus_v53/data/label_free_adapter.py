"""Common evaluation-information-free dataset adapter contract."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class LabelFreeDatasetAdapter(Protocol):
    def features(self) -> torch.Tensor:
        """Return target-node features without evaluation metadata."""

    def projected_views(self, path_ids: Sequence[str]) -> Mapping[str, torch.Tensor]:
        """Return exact projected edge-index arrays for declared paths."""

    def schema(self) -> Mapping[str, Any]:
        """Return dataset metadata restricted to structural fields."""

    def candidate_bank(self) -> Mapping[str, Any]:
        """Return the fixed path library and fixed-cardinality sets."""

    def data_identity(self) -> Mapping[str, Any]:
        """Return immutable source/cache identities without local paths."""


CONTRACT = {
    "schema_version": "metachorus.label-free-dataset-adapter-contract.v1",
    "required_methods": [
        "features",
        "projected_views",
        "schema",
        "candidate_bank",
        "data_identity",
    ],
    "forbidden_methods": [
        "labels",
        "num_classes",
        "class_count",
        "masks",
        "evaluate",
    ],
    "completed_adapters": ["acm", "freebase"],
    "not_implemented_adapters": ["dblp", "imdb", "aminer"],
    "downstream_evaluation_interface_present": False,
}
