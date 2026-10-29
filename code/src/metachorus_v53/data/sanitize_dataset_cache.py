"""Trusted ACM exporter: raw/PyG input in, deterministic label-free cache out."""

from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp
import torch

from metachorus.acm_adapter import (
    build_acm_candidate_bank,
    load_acm_projection,
)
from metachorus_v53.audit.common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_checksums,
    canonical_json_hash,
    read_json,
    sha256_file,
)
from metachorus_v53.m0b.context import (
    ACM_CANDIDATE_SPACE_IDENTITY,
    ACM_DATA_IDENTITY,
    ACM_PROCESSED_SHA256,
    ACM_PROJECTION_FILE_SHA256,
    ACM_PROJECTION_IDENTITY,
    ACM_RAW_SHA256,
    M0BContext,
)
from .array_identity import array_summary, write_npy_deterministic
from .label_cache import write_label_cache
from .sanitized_cache_schema import (
    DATASET_ID,
    FORMAT_ID,
    RELATIONS,
    SCHEMA_VERSION,
    TARGET_NODE_TYPE,
    schema_documents,
)
from .sanitized_cache_validator import validate_sanitized_cache


EXPORTER_ID = "metachorus.v53.m0b.acm-trusted-exporter.v1"
RELATION_KEYS = {
    "paper_to_author": ("paper", "to", "author"),
    "author_to_paper": ("author", "to", "paper"),
    "paper_to_subject": ("paper", "to", "subject"),
    "subject_to_paper": ("subject", "to", "paper"),
}


def _atomic_json(path: Path, value: Any) -> dict[str, Any]:
    atomic_write_json(path, value)
    return {
        "relative_path": "",
        "raw_file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_trusted_pyg(path: Path) -> Any:
    
    
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, (tuple, list)) and payload:
        return payload[0]
    return payload


def _extract_source(context: M0BContext) -> dict[str, Any]:
    if sha256_file(context.acm_processed) != ACM_PROCESSED_SHA256:
        raise ValueError("ACM processed input hash differs from M0-A")
    if sha256_file(context.acm_raw) != ACM_RAW_SHA256:
        raise ValueError("ACM raw input hash differs from M0-A")
    if sha256_file(context.acm_projection) != ACM_PROJECTION_FILE_SHA256:
        raise ValueError("ACM projection file hash differs from M0-A")
    data = _load_trusted_pyg(context.acm_processed)
    x = data["paper"].x.detach().cpu().contiguous()
    y = data["paper"].y.detach().cpu().contiguous()
    relations = {
        name: data[key].edge_index.detach().cpu().contiguous()
        for name, key in RELATION_KEYS.items()
    }
    loaded = load_acm_projection(
        context.acm_projection,
        context.acm_projection_manifest,
        expected_identity=ACM_PROJECTION_IDENTITY,
    )
    bank = build_acm_candidate_bank(loaded.acm)
    if bank.effective_count != 5 or bank.number_of_sets(2) != 10:
        raise ValueError("ACM candidate-space cardinality differs from M0-A")
    return {
        "features": x,
        "labels": y,
        "relations": relations,
        "loaded_projection": loaded,
        "bank": bank,
    }


def _csr_edge_index(graph: sp.csr_matrix) -> np.ndarray:
    graph = graph.tocsr(copy=True)
    graph.sum_duplicates()
    graph.eliminate_zeros()
    graph.sort_indices()
    rows = np.repeat(
        np.arange(graph.shape[0], dtype=np.int64),
        np.diff(graph.indptr).astype(np.int64, copy=False),
    )
    columns = graph.indices.astype(np.int64, copy=False)
    return np.ascontiguousarray(np.stack((rows, columns), axis=0))


def _safe_path_file_id(candidate_id: str, graph_hash: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    return f"acmp_{digest[:12]}_{graph_hash[:12]}"


def _candidate_documents(
    context: M0BContext,
    bank: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    path_rows: list[dict[str, Any]] = []
    file_ids: dict[str, str] = {}
    for ordinal, candidate in enumerate(bank.candidates):
        identifier = candidate.identifier
        graph_hash = str(candidate.canonical_graph_hash)
        file_id = _safe_path_file_id(identifier, graph_hash)
        file_ids[identifier] = file_id
        path_rows.append(
            {
                "ordinal": ordinal,
                "path_id": identifier,
                "filesystem_id": file_id,
                "relation_sequence": list(candidate.relations),
                "node_type_sequence": list(candidate.node_types),
                "semantic_projection_hash": graph_hash,
                "stored_edge_count": int(candidate.edge_count),
                "self_loop_count": int(
                    bank.graphs[str(candidate.canonical_graph_hash)]
                    .diagonal()
                    .astype(bool)
                    .sum()
                ),
                "array": f"projections/{file_id}.edge_index.npy",
            }
        )
    set_rows: list[dict[str, Any]] = []
    with context.candidate_space_csv().open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            set_rows.append(
                {
                    "set_id": row["set_id"],
                    "set_ordinal": int(row["set_ordinal"]),
                    "indices": [int(value) for value in row["indices"].split("|")],
                    "path_ids": row["candidate_ids"].split("|"),
                    "candidate_set_hash": row["candidate_set_hash"],
                }
            )
    path_library = {
        "schema_version": "metachorus.acm-sanitized-path-library.v1",
        "dataset_id": DATASET_ID,
        "projection_identity": ACM_PROJECTION_IDENTITY,
        "candidate_space_identity": ACM_CANDIDATE_SPACE_IDENTITY,
        "path_count": len(path_rows),
        "paths": path_rows,
    }
    fixed_sets = {
        "schema_version": "metachorus.acm-fixed-cardinality-sets.v1",
        "dataset_id": DATASET_ID,
        "view_budget": 2,
        "path_count": len(path_rows),
        "set_count": len(set_rows),
        "candidate_space_identity": ACM_CANDIDATE_SPACE_IDENTITY,
        "sets": set_rows,
    }
    return path_library, fixed_sets, file_ids


def _payload_row(path: Path, relative: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "relative_path": relative,
        "raw_file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if extra:
        row.update(dict(extra))
    return row


def _write_one_cache(
    context: M0BContext,
    *,
    output_parent: Path,
    extracted: Mapping[str, Any],
    label_identity: Mapping[str, Any],
) -> dict[str, Any]:
    bank = extracted["bank"]
    path_library, fixed_sets, file_ids = _candidate_documents(context, bank)
    cache_id_basis = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_ID,
        "dataset_id": DATASET_ID,
        "source_data_identity": ACM_DATA_IDENTITY,
        "source_processed_sha256": ACM_PROCESSED_SHA256,
        "source_raw_sha256": ACM_RAW_SHA256,
        "projection_identity": ACM_PROJECTION_IDENTITY,
        "candidate_space_identity": ACM_CANDIDATE_SPACE_IDENTITY,
        "exporter_id": EXPORTER_ID,
        "path_file_ids": file_ids,
    }
    deterministic_cache_id = canonical_json_hash(cache_id_basis)
    final_root = output_parent / f"acm_sanitized_{deterministic_cache_id[:24]}"
    if final_root.exists():
        raise FileExistsError(f"append-only sanitized cache exists: {final_root}")
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".acm_sanitized_staging_", dir=output_parent)
    )
    try:
        for name in (
            "schema",
            "features",
            "relations",
            "projections",
            "candidates",
            "metadata",
        ):
            (staging / name).mkdir()
        rows: list[dict[str, Any]] = []
        documents = schema_documents()
        documents["meta_path_schema.json"] = {
            "schema_version": SCHEMA_VERSION,
            "paths": path_library["paths"],
        }
        for name, value in sorted(documents.items()):
            path = staging / "schema" / name
            atomic_write_json(path, value)
            rows.append(_payload_row(path, f"schema/{name}"))
        feature_path = staging / "features" / "paper_x.npy"
        feature_record = write_npy_deterministic(
            feature_path, extracted["features"]
        )
        rows.append(_payload_row(feature_path, "features/paper_x.npy", feature_record))
        for name in RELATIONS:
            relative = f"relations/{name}.edge_index.npy"
            path = staging / relative
            record = write_npy_deterministic(path, extracted["relations"][name])
            rows.append(_payload_row(path, relative, record))
        for candidate in bank.candidates:
            relative = (
                f"projections/{file_ids[candidate.identifier]}.edge_index.npy"
            )
            path = staging / relative
            array = _csr_edge_index(
                bank.graphs[str(candidate.canonical_graph_hash)]
            )
            record = write_npy_deterministic(path, array)
            record["semantic_projection_hash"] = candidate.canonical_graph_hash
            rows.append(_payload_row(path, relative, record))
        candidate_documents = {
            "candidates/path_library.json": path_library,
            "candidates/fixed_cardinality_sets.json": fixed_sets,
            "metadata/data_identity.json": {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "source_data_identity": ACM_DATA_IDENTITY,
                "source_processed_sha256": ACM_PROCESSED_SHA256,
                "source_raw_sha256": ACM_RAW_SHA256,
                "node_counts": {
                    key: int(value)
                    for key, value in extracted[
                        "loaded_projection"
                    ].acm.num_nodes_by_type.items()
                },
                "feature_dimensions": list(extracted["features"].shape),
            },
            "metadata/projection_identity.json": {
                "schema_version": SCHEMA_VERSION,
                "projection_identity": ACM_PROJECTION_IDENTITY,
                "projection_file_sha256": ACM_PROJECTION_FILE_SHA256,
                "candidate_space_identity": ACM_CANDIDATE_SPACE_IDENTITY,
                "semantic_projection_hashes": {
                    candidate.identifier: candidate.canonical_graph_hash
                    for candidate in bank.candidates
                },
            },
            "metadata/export_config.json": {
                "schema_version": SCHEMA_VERSION,
                "format": FORMAT_ID,
                "exporter_id": EXPORTER_ID,
                "array_format": "npy_v2_allow_pickle_false",
                "edge_order_policy": {
                    "raw_relations": "source_exact",
                    "projected_views": "canonical_csr_row_major",
                },
                "uses_evaluation_information": False,
            },
            "SANITIZED_CACHE_SCHEMA.json": {
                "schema_version": SCHEMA_VERSION,
                "format": FORMAT_ID,
                "dataset_id": DATASET_ID,
                "target_node_type": TARGET_NODE_TYPE,
                "training_information_policy": "structural_and_feature_fields_only",
            },
        }
        for relative, value in sorted(candidate_documents.items()):
            path = staging / relative
            atomic_write_json(path, value)
            rows.append(_payload_row(path, relative))
        rows.sort(key=lambda row: row["relative_path"])
        cache_identity_basis = {
            "schema_version": SCHEMA_VERSION,
            "deterministic_cache_id": deterministic_cache_id,
            "files": rows,
        }
        cache_identity = canonical_json_hash(cache_identity_basis)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_ID,
            "dataset_id": DATASET_ID,
            "deterministic_cache_id": deterministic_cache_id,
            "cache_identity": cache_identity,
            "cache_identity_basis": cache_identity_basis,
            "files": rows,
            "file_count": len(rows),
            "uses_evaluation_information": False,
            "local_absolute_paths_present": False,
        }
        atomic_write_json(staging / "SANITIZED_CACHE_MANIFEST.json", manifest)
        atomic_write_text(
            staging / "SANITIZED_CACHE_CHECKSUMS.sha256",
            build_checksums(
                staging,
                exclude_names={
                    "SANITIZED_CACHE_CHECKSUMS.sha256",
                    "SANITIZED_CACHE_COMPLETE.json",
                },
            ),
        )
        complete = {
            "schema_version": "metachorus.sanitized-cache-complete.v1",
            "status": "COMPLETE",
            "cache_identity": cache_identity,
            "manifest_sha256": sha256_file(
                staging / "SANITIZED_CACHE_MANIFEST.json"
            ),
            "checksums_sha256": sha256_file(
                staging / "SANITIZED_CACHE_CHECKSUMS.sha256"
            ),
            "atomic_publish": True,
        }
        atomic_write_json(staging / "SANITIZED_CACHE_COMPLETE.json", complete)
        validation = validate_sanitized_cache(
            staging,
            label_array_identity=label_identity,
            require_complete=True,
        )
        if validation["status"] != "PASS":
            raise RuntimeError(f"sanitized staging validation failed: {validation}")
        os.replace(staging, final_root)
        return {
            "root": final_root,
            "deterministic_cache_id": deterministic_cache_id,
            "cache_identity": cache_identity,
            "manifest": manifest,
            "validation": validation,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _field_map_rows(path_library: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = [
        ("features/paper_x.npy", "data['paper'].x", "cpu_contiguous_npy_v2"),
        *[
            (
                f"relations/{name}.edge_index.npy",
                f"data[{RELATION_KEYS[name]!r}].edge_index",
                "cpu_contiguous_npy_v2_preserve_order",
            )
            for name in RELATIONS
        ],
        *[
            (
                row["array"],
                f"label_free_projection:{row['path_id']}",
                "canonical_csr_row_major_npy_v2",
            )
            for row in path_library["paths"]
        ],
        (
            "candidates/path_library.json",
            "M0A projection manifest + candidate bank",
            "canonical_json",
        ),
        (
            "candidates/fixed_cardinality_sets.json",
            "M0A ACM_CANDIDATE_SPACE.csv",
            "canonical_json",
        ),
    ]
    return [
        {
            "destination_field": destination,
            "source_field": source,
            "transformation": transform,
            "uses_labels": "false",
            "allowed_in_training_cache": "true",
        }
        for destination, source, transform in rows
    ]


def compare_exports(root_a: Path, root_b: Path) -> dict[str, Any]:
    def inventory(root: Path) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            row: dict[str, Any] = {
                "raw_file_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if path.suffix.casefold() == ".npy":
                row.update(array_summary(np.load(path, allow_pickle=False)))
            elif path.suffix.casefold() == ".json":
                row["canonical_json_sha256"] = canonical_json_hash(
                    read_json(path)
                )
            result[relative] = row
        return result

    a = inventory(root_a)
    b = inventory(root_b)
    raw_equal = set(a) == set(b) and all(
        a[path]["raw_file_sha256"] == b[path]["raw_file_sha256"] for path in a
    )
    semantic_equal = set(a) == set(b) and all(
        a[path].get("semantic_sha256") == b[path].get("semantic_sha256")
        and a[path].get("canonical_json_sha256")
        == b[path].get("canonical_json_sha256")
        for path in a
    )
    manifest_a = read_json(root_a / "SANITIZED_CACHE_MANIFEST.json")
    manifest_b = read_json(root_b / "SANITIZED_CACHE_MANIFEST.json")
    return {
        "schema_version": "metachorus.sanitized-export-reproducibility.v1",
        "status": "PASS"
        if raw_equal
        and semantic_equal
        and manifest_a["cache_identity"] == manifest_b["cache_identity"]
        else "FAIL",
        "directory_structure_identical": set(a) == set(b),
        "raw_byte_identity": raw_equal,
        "semantic_identity": semantic_equal,
        "cache_identity_identical": (
            manifest_a["cache_identity"] == manifest_b["cache_identity"]
        ),
        "cache_identity": manifest_a["cache_identity"],
        "file_count": len(a),
        "total_size_bytes_each": sum(row["size_bytes"] for row in a.values()),
        "mismatched_files": [
            path
            for path in sorted(set(a) | set(b))
            if a.get(path) != b.get(path)
        ],
    }


def export_acm_twice(context: M0BContext) -> dict[str, Any]:
    started = time.perf_counter()
    extracted_a = _extract_source(context)
    label_identity = array_summary(extracted_a["labels"])
    label = write_label_cache(
        context.output_root / "label_cache_local",
        extracted_a["labels"].numpy(),
        source_data_sha256=ACM_PROCESSED_SHA256,
        target_node_type=TARGET_NODE_TYPE,
    )
    export_a = _write_one_cache(
        context,
        output_parent=context.output_root / "sanitized_cache_local" / "export_A",
        extracted=extracted_a,
        label_identity=label_identity,
    )
    del extracted_a
    gc.collect()
    
    
    extracted_b = _extract_source(context)
    export_b = _write_one_cache(
        context,
        output_parent=context.output_root / "sanitized_cache_local" / "export_B",
        extracted=extracted_b,
        label_identity=label_identity,
    )
    del extracted_b
    gc.collect()
    reproducibility = compare_exports(export_a["root"], export_b["root"])
    if reproducibility["status"] != "PASS":
        raise RuntimeError(f"deterministic double export failed: {reproducibility}")
    trusted = context.output_root / "trusted_export"
    path_library = read_json(export_a["root"] / "candidates" / "path_library.json")
    field_rows = _field_map_rows(path_library)
    atomic_write_csv(
        trusted / "SANITIZATION_FIELD_MAP.csv",
        field_rows,
        fieldnames=[
            "destination_field",
            "source_field",
            "transformation",
            "uses_labels",
            "allowed_in_training_cache",
        ],
    )
    atomic_write_json(
        trusted / "SANITIZED_EXPORT_REPRODUCIBILITY.json", reproducibility
    )
    report = (
        "# Sanitization Report\n\n"
        f"- Format: {FORMAT_ID}\n"
        f"- Cache identity: {reproducibility['cache_identity']}\n"
        f"- Export A/B raw-byte identity: {reproducibility['raw_byte_identity']}\n"
        f"- Export A/B semantic identity: {reproducibility['semantic_identity']}\n"
        f"- Training fields using evaluation information: 0\n"
        f"- Separate local-only label cache: {label['label_cache_id']}\n"
        "- The trusted process alone deserialized the hash-pinned legacy PyG source. "
        "The training adapter reads only canonical JSON and allow_pickle=False NPY.\n"
    )
    atomic_write_text(trusted / "SANITIZATION_REPORT.md", report)
    result = {
        "schema_version": "metachorus.v53.m0b.acm-double-export.v1",
        "status": "PASS",
        "wall_time_seconds": time.perf_counter() - started,
        "export_a_root_local_only": str(export_a["root"]),
        "export_b_root_local_only": str(export_b["root"]),
        "label_cache_root_local_only": str(label["root"]),
        "cache_identity": export_a["cache_identity"],
        "label_cache_id": label["label_cache_id"],
        "label_array_identity": label_identity,
        "reproducibility": reproducibility,
        "field_map_rows": len(field_rows),
    }
    atomic_write_json(trusted / "TRUSTED_EXPORT_RECEIPT.json", result)
    return result
