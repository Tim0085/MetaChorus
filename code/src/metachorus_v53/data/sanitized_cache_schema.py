"""Frozen v1 schema and recursive leakage-token checks."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = "metachorus.sanitized-cache.v1"
FORMAT_ID = "npy_dir_v1"
DATASET_ID = "acm"
TARGET_NODE_TYPE = "paper"
RELATIONS = {
    "paper_to_author": ("paper", "author"),
    "author_to_paper": ("author", "paper"),
    "paper_to_subject": ("paper", "subject"),
    "subject_to_paper": ("subject", "paper"),
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".joblib",
    ".ckpt",
}
FORBIDDEN_TERMS = {
    "y",
    "label",
    "labels",
    "classid",
    "classids",
    "classcount",
    "numclasses",
    "labelname",
    "labelnames",
    "trainmask",
    "valmask",
    "testmask",
    "microf1",
    "macrof1",
    "nmi",
    "ari",
    "svc",
    "kmeans",
    "bestepoch",
    "earlystopmetric",
}
ABSOLUTE_PATH_PATTERN = re.compile(r"(?i)(?:[A-Z]:[\\/]|^/|\\\\[^\\]+\\)")


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def string_tokens(value: str) -> set[str]:
    parts = re.split(r"[^A-Za-z0-9]+", value)
    tokens = {normalize_token(part) for part in parts if part}
    tokens.add(normalize_token(value))
    for component in re.split(r"[\\/]", value):
        stem = component.rsplit(".", 1)[0]
        tokens.add(normalize_token(stem))
    return {token for token in tokens if token}


def forbidden_terms_in_string(value: str) -> set[str]:
    return string_tokens(value) & FORBIDDEN_TERMS


def recursive_forbidden_scan(value: Any, *, location: str = "$") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            for term in sorted(forbidden_terms_in_string(key_text)):
                findings.append(
                    {"location": f"{location}.<key>", "term": term, "value": key_text}
                )
            findings.extend(
                recursive_forbidden_scan(child, location=f"{location}.{key_text}")
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(
                recursive_forbidden_scan(child, location=f"{location}[{index}]")
            )
    elif isinstance(value, str):
        for term in sorted(forbidden_terms_in_string(value)):
            findings.append({"location": location, "term": term, "value": value})
    return findings


def assert_safe_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    if any(part.startswith(".") for part in path.parts):
        raise ValueError(f"hidden path component is forbidden: {value}")
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden serialization suffix: {value}")
    if forbidden_terms_in_string(value):
        raise ValueError(f"forbidden information token in path: {value}")


def schema_documents() -> dict[str, dict[str, Any]]:
    relation_rows = [
        {
            "relation_id": relation_id,
            "source_node_type": endpoints[0],
            "destination_node_type": endpoints[1],
            "array": f"relations/{relation_id}.edge_index.npy",
        }
        for relation_id, endpoints in sorted(RELATIONS.items())
    ]
    return {
        "dataset_schema.json": {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_ID,
            "dataset_id": DATASET_ID,
            "target_node_type": TARGET_NODE_TYPE,
            "feature_array": "features/paper_x.npy",
            "node_count_fields": ["paper", "author", "subject"],
        },
        "relation_schema.json": {
            "schema_version": SCHEMA_VERSION,
            "relations": relation_rows,
        },
    }


def allowed_static_files() -> set[str]:
    return {
        "schema/dataset_schema.json",
        "schema/relation_schema.json",
        "schema/meta_path_schema.json",
        "features/paper_x.npy",
        *{
            f"relations/{name}.edge_index.npy"
            for name in RELATIONS
        },
        "candidates/path_library.json",
        "candidates/fixed_cardinality_sets.json",
        "metadata/data_identity.json",
        "metadata/projection_identity.json",
        "metadata/export_config.json",
        "SANITIZED_CACHE_SCHEMA.json",
        "SANITIZED_CACHE_MANIFEST.json",
        "SANITIZED_CACHE_CHECKSUMS.sha256",
        "SANITIZED_CACHE_COMPLETE.json",
    }


def assert_no_forbidden_fields(values: Iterable[Any]) -> None:
    findings: list[dict[str, str]] = []
    for index, value in enumerate(values):
        findings.extend(recursive_forbidden_scan(value, location=f"$[{index}]"))
    if findings:
        raise ValueError(f"forbidden information found: {findings[:5]}")
