"""Trusted deterministic exporter for the three fast-stage label-free caches."""

from __future__ import annotations

import hashlib
import itertools
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData

from metachorus.candidate_bank import (
    MetaPathCandidate,
    SchemaRelation,
    StructuralBudget,
    build_candidate_bank,
)
from metachorus.graph_canonicalizer import CanonicalizationConfig
from metachorus_v53.audit.common import (
    build_checksums,
    canonical_json_hash,
    load_checksums,
    read_json,
    sha256_file,
)
from metachorus_v53.data.array_identity import (
    array_semantic_sha256,
    array_summary,
    write_npy_deterministic,
)

from .common import write_json_once, write_text_once
from .constants import ALLOWED_HOPS, MAX_HOPS
from .dataset_specs import DATASET_SPECS, DatasetSpec


CACHE_SCHEMA = "metachorus.v53.m6f-m7f.fast-sanitized-cache.v1"
EXPORTER_ID = "metachorus.v53.m6f-m7f.trusted-heterodata-exporter.v1"
CANONICALIZATION = CanonicalizationConfig(
    symmetrize=False,
    add_self_loops=True,
    binary=True,
)


def _trusted_load_heterodata(path: Path) -> HeteroData:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value: Any = payload[0] if isinstance(payload, (tuple, list)) and payload else payload
    if isinstance(value, HeteroData):
        return value
    if isinstance(value, Mapping):
        return HeteroData.from_dict(dict(value))
    raise TypeError(f"unsupported hash-pinned PyG payload: {type(value)!r}")


def _relation_matrix(edge_index: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    edge = np.asarray(edge_index, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("relation edge index must have shape [2,E]")
    if edge.size and (
        edge[0].min() < 0
        or edge[1].min() < 0
        or edge[0].max() >= shape[0]
        or edge[1].max() >= shape[1]
    ):
        raise ValueError("relation edge index exceeds its frozen node counts")
    matrix = sp.coo_matrix(
        (np.ones(edge.shape[1], dtype=np.float32), (edge[0], edge[1])),
        shape=shape,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _binary_product(left: sp.csr_matrix, right: sp.csr_matrix) -> sp.csr_matrix:
    value = (left @ right).tocsr()
    value.sum_duplicates()
    value.data[:] = 1.0
    value.eliminate_zeros()
    value.sort_indices()
    return value


def _csr_edge_index(graph: sp.csr_matrix) -> np.ndarray:
    value = graph.tocsr(copy=True)
    value.sum_duplicates()
    value.eliminate_zeros()
    value.sort_indices()
    rows = np.repeat(
        np.arange(value.shape[0], dtype=np.int64),
        np.diff(value.indptr).astype(np.int64, copy=False),
    )
    columns = value.indices.astype(np.int64, copy=False)
    return np.ascontiguousarray(np.stack((rows, columns), axis=0))


def _safe_file_id(dataset: str, path_id: str, graph_hash: str) -> str:
    path_digest = hashlib.sha256(path_id.encode("utf-8")).hexdigest()
    return f"{dataset}p_{path_digest[:12]}_{graph_hash[:12]}"


def _candidate_set_hash(
    *,
    dataset: str,
    path_ids: Sequence[str],
    view_budget: int,
    path_library_identity: str,
) -> str:
    if len(path_ids) != view_budget or len(set(path_ids)) != view_budget:
        raise ValueError("candidate set violates its fixed cardinality")
    return canonical_json_hash(
        {
            "schema_version": "metachorus.v53.m6f-m7f.candidate-set.v1",
            "dataset": dataset,
            "path_ids": sorted(map(str, path_ids)),
            "view_budget": int(view_budget),
            "path_library_identity": path_library_identity,
        }
    )


def _extract_source(spec: DatasetSpec, source_root: Path) -> dict[str, Any]:
    processed = source_root / spec.processed_relative_path
    if (
        not processed.is_file()
        or processed.is_symlink()
        or processed.stat().st_size != spec.processed_size_bytes
        or sha256_file(processed) != spec.processed_sha256
    ):
        raise ValueError(f"{spec.dataset} processed source identity mismatch")
    data = _trusted_load_heterodata(processed)
    target = data[spec.target_node_type]
    feature = target.x.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    if tuple(feature.shape) != spec.feature_shape or not np.isfinite(feature).all():
        raise ValueError(f"{spec.dataset} target feature contract mismatch")
    relation_arrays: dict[str, np.ndarray] = {}
    relation_graphs: dict[str, sp.csr_matrix] = {}
    for relation in spec.relations:
        edge = data[relation.pyg_relation].edge_index.detach().cpu().contiguous().numpy()
        edge = np.ascontiguousarray(edge, dtype=np.int64)
        relation_arrays[relation.name] = edge
        relation_graphs[relation.name] = _relation_matrix(
            edge,
            (spec.node_counts[relation.source_type], spec.node_counts[relation.target_type]),
        )

    def graph_builder(candidate: MetaPathCandidate) -> sp.csr_matrix:
        product = relation_graphs[candidate.relations[0]]
        for relation_name in candidate.relations[1:]:
            product = _binary_product(product, relation_graphs[relation_name])
        return product

    bank = build_candidate_bank(
        relations=tuple(
            SchemaRelation(row.name, row.source_type, row.target_type)
            for row in spec.relations
        ),
        target_type=spec.target_node_type,
        max_hops=MAX_HOPS,
        allowed_hops=ALLOWED_HOPS,
        graph_builder=graph_builder,
        manual_paths=spec.manual_paths,
        canonicalization=CANONICALIZATION,
        structural_budget=StructuralBudget(),
        view_budget=spec.view_budget,
    )
    return {
        "feature": feature,
        "relation_arrays": relation_arrays,
        "bank": bank,
    }


def _documents(spec: DatasetSpec, extracted: Mapping[str, Any]) -> dict[str, Any]:
    bank = extracted["bank"]
    path_rows: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(bank.candidates):
        graph_hash = str(candidate.canonical_graph_hash)
        path_id = candidate.identifier
        path_rows.append(
            {
                "ordinal": ordinal,
                "path_id": path_id,
                "filesystem_id": _safe_file_id(spec.dataset, path_id, graph_hash),
                "relation_sequence": list(candidate.relations),
                "node_type_sequence": list(candidate.node_types),
                "semantic_projection_hash": graph_hash,
                "equivalent_path_ids": ["::".join(value) for value in candidate.equivalent_paths],
                "is_manual_representative": bool(candidate.is_manual),
                "edge_count": int(candidate.edge_count),
                "average_degree": float(candidate.average_degree),
                "target_coverage": float(candidate.target_coverage),
                "gpu_edge_index_bytes": int(candidate.edge_count) * 2 * 8,
                "array": f"projections/{_safe_file_id(spec.dataset, path_id, graph_hash)}.edge_index.npy",
            }
        )
    path_library_basis = {
        "schema_version": "metachorus.v53.m6f-m7f.fast-path-library.v1",
        "dataset": spec.dataset,
        "structural_data_identity": spec.structural_data_identity,
        "target_node_type": spec.target_node_type,
        "max_hops": MAX_HOPS,
        "allowed_hops": list(ALLOWED_HOPS),
        "canonicalization": {
            "symmetrize": False,
            "add_self_loops": True,
            "binary": True,
        },
        "schema_path_count": sum(1 for _ in _schema_paths(spec)),
        "effective_path_count": len(path_rows),
        "paths": path_rows,
    }
    path_library_identity = canonical_json_hash(path_library_basis)
    path_library = {**path_library_basis, "path_library_identity": path_library_identity}

    set_rows: list[dict[str, Any]] = []
    for ordinal, indices in enumerate(itertools.combinations(range(len(path_rows)), spec.view_budget)):
        path_ids = [path_rows[index]["path_id"] for index in indices]
        candidate_hash = _candidate_set_hash(
            dataset=spec.dataset,
            path_ids=path_ids,
            view_budget=spec.view_budget,
            path_library_identity=path_library_identity,
        )
        set_rows.append(
            {
                "candidate_set_ordinal": ordinal,
                "candidate_set_id": f"fast_{spec.dataset}_{ordinal:03d}_{candidate_hash[:16]}",
                "candidate_set_hash": candidate_hash,
                "path_indices": list(indices),
                "path_ids": path_ids,
            }
        )
    all_sets_basis = {
        "schema_version": "metachorus.v53.m6f-m7f.fast-all-sets.v1",
        "dataset": spec.dataset,
        "path_library_identity": path_library_identity,
        "view_budget": spec.view_budget,
        "set_count": len(set_rows),
        "sets": set_rows,
    }
    all_sets_identity = canonical_json_hash(all_sets_basis)
    all_sets = {**all_sets_basis, "all_sets_identity": all_sets_identity}

    manual_ids = []
    for path in spec.manual_paths:
        manual_ids.append(bank.by_relations(path).identifier)
    manual_matches = [
        row for row in set_rows if set(row["path_ids"]) == set(manual_ids)
    ]
    if len(manual_matches) != 1:
        raise ValueError(f"{spec.dataset} Manual set does not resolve uniquely")
    manual = {**manual_matches[0], "path_set_role": "MANUAL", "status": "EXACT_IDENTITY_RESOLVED"}
    fixed = {
        "schema_version": "metachorus.v53.m6f-m7f.fast-baseline-set-registry.v1",
        "dataset": spec.dataset,
        "path_library_identity": path_library_identity,
        "all_sets_identity": all_sets_identity,
        "view_budget": spec.view_budget,
        "sets": [manual],
        "current_automatic": {
            "status": "NOT_AVAILABLE_EXACT_IDENTITY",
            "reason": "no repository-closed candidate-set/source/config/projection identity found",
        },
    }
    return {
        "path_library": path_library,
        "all_sets": all_sets,
        "fixed_sets": fixed,
    }


def _schema_paths(spec: DatasetSpec) -> tuple[MetaPathCandidate, ...]:
    from metachorus.candidate_bank import enumerate_schema_closed_paths

    return enumerate_schema_closed_paths(
        tuple(SchemaRelation(row.name, row.source_type, row.target_type) for row in spec.relations),
        spec.target_node_type,
        MAX_HOPS,
        allowed_hops=ALLOWED_HOPS,
    )


def _write_cache(
    *,
    spec: DatasetSpec,
    output_parent: Path,
    extracted: Mapping[str, Any],
) -> dict[str, Any]:
    documents = _documents(spec, extracted)
    cache_identity_basis = {
        "schema_version": CACHE_SCHEMA,
        "dataset": spec.dataset,
        "structural_data_identity": spec.structural_data_identity,
        "path_library_identity": documents["path_library"]["path_library_identity"],
        "all_sets_identity": documents["all_sets"]["all_sets_identity"],
        "manual_set_hash": documents["fixed_sets"]["sets"][0]["candidate_set_hash"],
        "exporter_id": EXPORTER_ID,
    }
    cache_identity = canonical_json_hash(cache_identity_basis)
    final_root = output_parent / spec.dataset
    if final_root.exists():
        complete = read_json(final_root / "SANITIZED_CACHE_COMPLETE.json")
        manifest = read_json(final_root / "SANITIZED_CACHE_MANIFEST.json")
        observed_basis = manifest.get("identity_basis", {})
        manifest_path = final_root / "SANITIZED_CACHE_MANIFEST.json"
        checksums_path = final_root / "SANITIZED_CACHE_CHECKSUMS.sha256"
        checksums = load_checksums(checksums_path)
        expected_checksum_members = {
            path.relative_to(final_root).as_posix()
            for path in final_root.rglob("*")
            if path.is_file()
            and path.name
            not in {"SANITIZED_CACHE_CHECKSUMS.sha256", "SANITIZED_CACHE_COMPLETE.json"}
        }
        if (
            complete.get("status") != "COMPLETE"
            or complete.get("cache_identity") != manifest.get("cache_identity")
            or canonical_json_hash(observed_basis) != manifest.get("cache_identity")
            or sha256_file(manifest_path) != complete.get("manifest_sha256")
            or sha256_file(checksums_path) != complete.get("checksums_sha256")
            or set(checksums) != expected_checksum_members
            or any(sha256_file(final_root / relative) != digest for relative, digest in checksums.items())
            or observed_basis.get("structural_data_identity") != spec.structural_data_identity
            or observed_basis.get("path_library_identity")
            != documents["path_library"]["path_library_identity"]
            or observed_basis.get("all_sets_identity")
            != documents["all_sets"]["all_sets_identity"]
        ):
            raise FileExistsError(f"conflicting cache exists: {final_root}")
        return {
            "dataset": spec.dataset,
            "cache_root": str(final_root),
            "cache_identity": str(complete["cache_identity"]),
            "path_library_identity": documents["path_library"]["path_library_identity"],
            "all_sets_identity": documents["all_sets"]["all_sets_identity"],
            "path_count": len(documents["path_library"]["paths"]),
            "set_count": len(documents["all_sets"]["sets"]),
            "manual_set_hash": documents["fixed_sets"]["sets"][0]["candidate_set_hash"],
            "status": "REUSED_VERIFIED_CACHE",
        }
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.dataset}_fast_cache_", dir=output_parent))
    try:
        for name in ("schema", "features", "relations", "projections", "candidates", "metadata"):
            (staging / name).mkdir()

        dataset_schema = {
            "schema_version": CACHE_SCHEMA,
            "dataset": spec.dataset,
            "target_node_type": spec.target_node_type,
            "target_feature_array": f"features/{spec.target_node_type}_x.npy",
            "target_feature_shape": list(spec.feature_shape),
            "node_counts": dict(spec.node_counts),
            "view_budget": spec.view_budget,
        }
        relation_rows: list[dict[str, Any]] = []
        for relation in spec.relations:
            edge = np.asarray(extracted["relation_arrays"][relation.name], dtype=np.int64)
            relative = f"relations/{relation.name}.edge_index.npy"
            relation_rows.append(
                {
                    "relation_id": relation.name,
                    "source_node_type": relation.source_type,
                    "destination_node_type": relation.target_type,
                    "array": relative,
                    "shape": list(edge.shape),
                    "semantic_sha256": array_semantic_sha256(edge),
                }
            )
        relation_schema = {"schema_version": CACHE_SCHEMA, "dataset": spec.dataset, "relations": relation_rows}
        meta_path_schema = {
            "schema_version": CACHE_SCHEMA,
            "dataset": spec.dataset,
            "path_library_identity": documents["path_library"]["path_library_identity"],
            "paths": documents["path_library"]["paths"],
        }
        data_identity = {
            "schema_version": CACHE_SCHEMA,
            "dataset": spec.dataset,
            "structural_data_identity": spec.structural_data_identity,
            "structural_data_identity_basis": spec.structural_data_identity_basis,
            "feature_semantic_sha256": array_semantic_sha256(extracted["feature"]),
            "relation_semantic_sha256": {
                row["relation_id"]: row["semantic_sha256"] for row in relation_rows
            },
        }
        projection_identity = {
            "schema_version": CACHE_SCHEMA,
            "dataset": spec.dataset,
            "path_library_identity": documents["path_library"]["path_library_identity"],
            "all_sets_identity": documents["all_sets"]["all_sets_identity"],
            "semantic_projection_hashes": [
                row["semantic_projection_hash"] for row in documents["path_library"]["paths"]
            ],
        }
        export_config = {
            "schema_version": CACHE_SCHEMA,
            "exporter_id": EXPORTER_ID,
            "uses_supervised_information": False,
            "source_deserialization_boundary": "hash_pinned_trusted_export_only",
            "training_adapter_deserialization": "numeric_npy_allow_pickle_false",
        }
        json_documents = {
            "schema/dataset_schema.json": dataset_schema,
            "schema/relation_schema.json": relation_schema,
            "schema/meta_path_schema.json": meta_path_schema,
            "candidates/path_library.json": documents["path_library"],
            "candidates/all_sets.json": documents["all_sets"],
            "candidates/fixed_cardinality_sets.json": documents["fixed_sets"],
            "metadata/data_identity.json": data_identity,
            "metadata/projection_identity.json": projection_identity,
            "metadata/export_config.json": export_config,
            "SANITIZED_CACHE_SCHEMA.json": {
                "schema_version": CACHE_SCHEMA,
                "format": "deterministic_numeric_npy_directory_v1",
                "forbidden_payloads": ["supervised targets", "split assignments", "metric values"],
            },
        }
        for relative, value in json_documents.items():
            write_json_once(staging / relative, value)

        array_records: dict[str, dict[str, Any]] = {}
        feature_relative = f"features/{spec.target_node_type}_x.npy"
        array_records[feature_relative] = write_npy_deterministic(staging / feature_relative, extracted["feature"])
        for relation in spec.relations:
            relative = f"relations/{relation.name}.edge_index.npy"
            array_records[relative] = write_npy_deterministic(staging / relative, extracted["relation_arrays"][relation.name])
        graph_lookup = extracted["bank"].graphs
        for row in documents["path_library"]["paths"]:
            relative = str(row["array"])
            edge = _csr_edge_index(graph_lookup[str(row["semantic_projection_hash"])])
            if edge.shape != (2, int(row["edge_count"])):
                raise ValueError(f"projection edge contract changed: {row['path_id']}")
            array_records[relative] = write_npy_deterministic(staging / relative, edge)

        file_rows: dict[str, dict[str, Any]] = {}
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(staging).as_posix()
            row = {
                "raw_file_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if relative in array_records:
                row.update(array_records[relative])
            file_rows[relative] = row
        manifest_identity_basis = {
            **cache_identity_basis,
            "cache_identity": cache_identity,
            "arrays": {relative: record["semantic_sha256"] for relative, record in sorted(array_records.items())},
            "static_documents": {
                relative: record["raw_file_sha256"]
                for relative, record in sorted(file_rows.items())
                if relative not in array_records
            },
        }
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "status": "SEALED_LABEL_FREE_CACHE",
            "dataset": spec.dataset,
            "cache_identity": canonical_json_hash(manifest_identity_basis),
            "identity_basis": manifest_identity_basis,
            "files": file_rows,
        }
        
        
        cache_identity = manifest["cache_identity"]
        write_json_once(staging / "SANITIZED_CACHE_MANIFEST.json", manifest)
        checksums = build_checksums(
            staging,
            exclude_names={"SANITIZED_CACHE_CHECKSUMS.sha256", "SANITIZED_CACHE_COMPLETE.json"},
        )
        write_text_once(staging / "SANITIZED_CACHE_CHECKSUMS.sha256", checksums)
        complete = {
            "schema_version": CACHE_SCHEMA,
            "status": "COMPLETE",
            "dataset": spec.dataset,
            "cache_identity": cache_identity,
            "manifest_sha256": sha256_file(staging / "SANITIZED_CACHE_MANIFEST.json"),
            "checksums_sha256": sha256_file(staging / "SANITIZED_CACHE_CHECKSUMS.sha256"),
            "atomic_publish": True,
            "supervised_payloads_present": False,
        }
        write_json_once(staging / "SANITIZED_CACHE_COMPLETE.json", complete)
        os.replace(staging, final_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "dataset": spec.dataset,
        "cache_root": str(final_root),
        "cache_identity": cache_identity,
        "path_library_identity": documents["path_library"]["path_library_identity"],
        "all_sets_identity": documents["all_sets"]["all_sets_identity"],
        "path_count": len(documents["path_library"]["paths"]),
        "set_count": len(documents["all_sets"]["sets"]),
        "manual_set_hash": documents["fixed_sets"]["sets"][0]["candidate_set_hash"],
        "status": "COMPLETE",
    }


def build_sanitized_cache(
    *,
    dataset: str,
    source_root: str | Path,
    output_parent: str | Path,
) -> dict[str, Any]:
    if dataset not in DATASET_SPECS:
        raise KeyError(dataset)
    spec = DATASET_SPECS[dataset]
    source = Path(source_root).resolve()
    output = Path(output_parent).resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source root must be a regular directory")
    extracted = _extract_source(spec, source)
    return _write_cache(spec=spec, output_parent=output, extracted=extracted)


__all__ = [
    "CACHE_SCHEMA",
    "EXPORTER_ID",
    "CANONICALIZATION",
    "build_sanitized_cache",
]
