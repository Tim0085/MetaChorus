"""Frozen cross-dataset adapter contract and compliance schema."""

from __future__ import annotations

from typing import Any

from .label_free_adapter import CONTRACT
from .sanitized_cache_schema import SCHEMA_VERSION


def contract_documents() -> dict[str, dict[str, Any]]:
    return {
        "LABEL_FREE_DATASET_ADAPTER_CONTRACT.json": dict(CONTRACT),
        "SANITIZED_CACHE_SCHEMA_V1.json": {
            "schema_version": SCHEMA_VERSION,
            "storage": "canonical_json_plus_npy_v2_allow_pickle_false",
            "evaluation_information_allowed": False,
            "implemented_datasets": ["acm"],
            "not_implemented_datasets": ["freebase", "dblp", "imdb", "aminer"],
        },
        "DATASET_ADAPTER_COMPLIANCE_SCHEMA.json": {
            "schema_version": "metachorus.dataset-adapter-compliance-schema.v1",
            "required_fields": [
                "adapter_id",
                "implementation",
                "required_methods_present",
                "forbidden_methods_present",
                "complete",
            ],
            "completion_rule": (
                "all required methods present and all forbidden methods absent"
            ),
            "unimplemented_is_not_complete": True,
        },
    }
