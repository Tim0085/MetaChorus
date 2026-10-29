"""Adapter for label-free ACM graph projections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp
import torch

from .candidate_bank import (
    CandidateBank,
    MetaPathCandidate,
    SchemaRelation,
    StructuralBudget,
    build_candidate_bank,
)
from .graph_canonicalizer import CanonicalizationConfig


ACM_PROJECTION_SCHEMA = "metachorus.acm-label-free-projection.v1"
ACM_PROJECTION_MANIFEST_SCHEMA = (
    "metachorus.acm-label-free-projection-manifest.v1"
)
ACM_SOURCE_LOGICAL_NAME = "data/acm/processed/data.pt"
ACM_NODE_TYPES = ("paper", "author", "subject")
ACM_RELATION_NAMES = (
    "paper_to_author",
    "author_to_paper",
    "paper_to_subject",
    "subject_to_paper",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_FIELD_NAMES = {"y", "label", "labels", "target", "targets"}


@dataclass(frozen=True)
class LabelFreeACM:
    """Only fields authorised to enter the unsupervised search process."""

    features: torch.Tensor
    relations: Mapping[str, sp.csr_matrix]
    num_nodes_by_type: Mapping[str, int]
    projection_identity: str
    source_identity: Mapping[str, Any] | None = None

    @property
    def num_papers(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class LoadedACMProjection:
    """Validated search object plus the immutable artifact attestation."""

    acm: LabelFreeACM
    attestation: Mapping[str, Any]


ACM_SCHEMA = (
    SchemaRelation("paper_to_author", "paper", "author"),
    SchemaRelation("author_to_paper", "author", "paper"),
    SchemaRelation("paper_to_subject", "paper", "subject"),
    SchemaRelation("subject_to_paper", "subject", "paper"),
)
ACM_MANUAL_PATHS = (
    ("paper_to_author", "author_to_paper"),
    ("paper_to_subject", "subject_to_paper"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    frozen = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(frozen.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(frozen.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(frozen.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping_with_exact_keys(
    value: Any, expected: set[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    observed = {str(key) for key in value}
    if observed != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )
    return value


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_source_identity(value: Any) -> dict[str, Any]:
    source = _mapping_with_exact_keys(
        value, {"logical_name", "sha256", "size_bytes"}, "source_identity"
    )
    if source.get("logical_name") != ACM_SOURCE_LOGICAL_NAME:
        raise ValueError("projection source is not the fixed ACM processed data.pt")
    return {
        "logical_name": ACM_SOURCE_LOGICAL_NAME,
        "sha256": _strict_sha256(source.get("sha256"), "source_identity.sha256"),
        "size_bytes": _strict_positive_int(
            source.get("size_bytes"), "source_identity.size_bytes"
        ),
    }


def _validate_shape(value: Any, *, rank: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != rank:
        raise ValueError(f"{name} must be a rank-{rank} JSON shape")
    result = tuple(_strict_positive_int(item, name) for item in value)
    if rank == 2 and result[0] != 2 and name.startswith("relations."):
        raise ValueError(f"{name} must have leading dimension 2")
    return result


def _validate_tensor_contract(
    value: Any, counts: Mapping[str, int]
) -> dict[str, Any]:
    contract = _mapping_with_exact_keys(
        value,
        {"features", "relations", "num_nodes_by_type"},
        "tensor_contract",
    )
    feature = _mapping_with_exact_keys(
        contract.get("features"), {"dtype", "shape", "sha256"}, "features contract"
    )
    if feature.get("dtype") != "torch.float32":
        raise ValueError("features contract dtype must be torch.float32")
    feature_shape = _validate_shape(
        feature.get("shape"), rank=2, name="features.shape"
    )
    if feature_shape[0] != counts["paper"]:
        raise ValueError("features contract row count differs from paper count")
    normalized_feature = {
        "dtype": "torch.float32",
        "shape": list(feature_shape),
        "sha256": _strict_sha256(feature.get("sha256"), "features.sha256"),
    }

    relations = _mapping_with_exact_keys(
        contract.get("relations"), set(ACM_RELATION_NAMES), "relations contract"
    )
    normalized_relations: dict[str, Any] = {}
    expected_counts = {
        "paper_to_author": (counts["paper"], counts["author"]),
        "author_to_paper": (counts["author"], counts["paper"]),
        "paper_to_subject": (counts["paper"], counts["subject"]),
        "subject_to_paper": (counts["subject"], counts["paper"]),
    }
    for name in ACM_RELATION_NAMES:
        relation = _mapping_with_exact_keys(
            relations.get(name),
            {"dtype", "shape", "sha256", "source_nodes", "target_nodes"},
            f"relations.{name}",
        )
        if relation.get("dtype") != "torch.int64":
            raise ValueError(f"relations.{name} dtype must be torch.int64")
        shape = _validate_shape(
            relation.get("shape"), rank=2, name=f"relations.{name}.shape"
        )
        source_nodes, target_nodes = expected_counts[name]
        if (
            relation.get("source_nodes") != source_nodes
            or relation.get("target_nodes") != target_nodes
        ):
            raise ValueError(f"relations.{name} node-count contract is invalid")
        normalized_relations[name] = {
            "dtype": "torch.int64",
            "shape": list(shape),
            "sha256": _strict_sha256(
                relation.get("sha256"), f"relations.{name}.sha256"
            ),
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
        }
    manifest_counts = _mapping_with_exact_keys(
        contract.get("num_nodes_by_type"),
        set(ACM_NODE_TYPES),
        "tensor_contract.num_nodes_by_type",
    )
    if dict(manifest_counts) != dict(counts):
        raise ValueError("tensor contract node counts differ from source counts")
    return {
        "features": normalized_feature,
        "relations": normalized_relations,
        "num_nodes_by_type": dict(counts),
    }


def _validate_manifest(
    manifest: Any,
    *,
    projection_path: Path,
    manifest_sha256: str,
    manifest_size_bytes: int,
    expected_identity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _mapping_with_exact_keys(
        manifest,
        {
            "schema_version",
            "projection",
            "projection_identity",
            "source_identity",
            "tensor_contract",
        },
        "projection manifest",
    )
    if root.get("schema_version") != ACM_PROJECTION_MANIFEST_SCHEMA:
        raise ValueError("unsupported ACM projection manifest schema")
    source = _validate_source_identity(root.get("source_identity"))
    counts_raw = _mapping_with_exact_keys(
        _mapping_with_exact_keys(
            root.get("tensor_contract"),
            {"features", "relations", "num_nodes_by_type"},
            "tensor_contract",
        ).get("num_nodes_by_type"),
        set(ACM_NODE_TYPES),
        "tensor_contract.num_nodes_by_type",
    )
    counts = {
        name: _strict_positive_int(counts_raw.get(name), f"node count {name}")
        for name in ACM_NODE_TYPES
    }
    tensor_contract = _validate_tensor_contract(root.get("tensor_contract"), counts)
    projection = _mapping_with_exact_keys(
        root.get("projection"),
        {"schema_version", "sha256", "size_bytes"},
        "projection file contract",
    )
    if projection.get("schema_version") != ACM_PROJECTION_SCHEMA:
        raise ValueError("projection file contract has the wrong schema")
    expected_size = _strict_positive_int(
        projection.get("size_bytes"), "projection.size_bytes"
    )
    expected_sha = _strict_sha256(projection.get("sha256"), "projection.sha256")
    if projection_path.stat().st_size != expected_size:
        raise ValueError("ACM projection file size differs from its manifest")
    if sha256_file(projection_path) != expected_sha:
        raise ValueError("ACM projection file SHA-256 differs from its manifest")
    identity_payload = {
        "schema_version": ACM_PROJECTION_MANIFEST_SCHEMA,
        "projection": {
            "schema_version": ACM_PROJECTION_SCHEMA,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        },
        "source_identity": source,
        "tensor_contract": tensor_contract,
    }
    identity = _strict_sha256(
        root.get("projection_identity"), "projection_identity"
    )
    if identity != canonical_hash(identity_payload):
        raise ValueError("ACM projection identity is not canonical")
    required_identity = _strict_sha256(expected_identity, "expected projection identity")
    if identity != required_identity:
        raise ValueError("ACM projection identity differs from the explicit expectation")
    normalized = {**identity_payload, "projection_identity": identity}
    attestation = {
        "projection_identity": identity,
        "projection_schema_version": ACM_PROJECTION_SCHEMA,
        "projection_sha256": expected_sha,
        "projection_size_bytes": expected_size,
        "manifest_schema_version": ACM_PROJECTION_MANIFEST_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": manifest_size_bytes,
        "source_identity": source,
    }
    return normalized, attestation


def _assert_no_forbidden_keys(value: Any, path: str = "projection") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_FIELD_NAMES:
                raise ValueError(f"label-bearing field is forbidden: {path}.{key}")
            _assert_no_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, f"{path}[{index}]")


def _validate_edge_index(
    value: Any,
    *,
    name: str,
    source_nodes: int,
    target_nodes: int,
    contract: Mapping[str, Any],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.dtype != torch.int64 or value.device.type != "cpu" or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 CPU int64 tensor")
    edge = value.detach().contiguous()
    if tuple(edge.shape) != tuple(contract["shape"]) or edge.shape[0] != 2:
        raise ValueError(f"{name} shape differs from its manifest")
    if edge.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one edge")
    if (
        int(edge[0].min()) < 0
        or int(edge[0].max()) >= source_nodes
        or int(edge[1].min()) < 0
        or int(edge[1].max()) >= target_nodes
    ):
        raise ValueError(f"{name} contains an out-of-range endpoint")
    canonical_keys = edge[0] * target_nodes + edge[1]
    if canonical_keys.numel() > 1 and not torch.all(
        canonical_keys[1:] > canonical_keys[:-1]
    ):
        raise ValueError(f"{name} must be strictly sorted and duplicate-free")
    if tensor_sha256(edge) != contract["sha256"]:
        raise ValueError(f"{name} tensor hash differs from its manifest")
    return edge


def _to_csr(edge_index: torch.Tensor, shape: tuple[int, int]) -> sp.csr_matrix:
    rows = edge_index[0].numpy()
    columns = edge_index[1].numpy()
    values = np.ones(rows.shape[0], dtype=np.float32)
    matrix = sp.csr_matrix((values, (rows, columns)), shape=shape)
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.sort_indices()
    return matrix


def load_acm_projection(
    projection_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_identity: str,
) -> LoadedACMProjection:
    """Load only a manifest-bound pure-tensor projection with ``weights_only``."""

    projection = Path(projection_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not projection.is_file():
        raise FileNotFoundError(f"ACM label-free projection not found: {projection}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"ACM projection manifest not found: {manifest_file}")
    try:
        manifest_bytes = manifest_file.read_bytes()
        manifest_value = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("ACM projection manifest is not valid UTF-8 JSON") from error
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_size_bytes = len(manifest_bytes)
    manifest, attestation = _validate_manifest(
        manifest_value,
        projection_path=projection,
        manifest_sha256=manifest_sha256,
        manifest_size_bytes=manifest_size_bytes,
        expected_identity=expected_identity,
    )
    verified_projection_sha = attestation["projection_sha256"]
    try:
        payload = torch.load(projection, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "this PyTorch build cannot enforce weights_only=True for the ACM projection"
        ) from error
    if sha256_file(projection) != verified_projection_sha:
        raise ValueError("ACM projection changed while it was being loaded")
    if (
        manifest_file.stat().st_size != manifest_size_bytes
        or sha256_file(manifest_file) != manifest_sha256
    ):
        raise ValueError("ACM projection manifest changed while it was being loaded")
    _assert_no_forbidden_keys(payload)
    root = _mapping_with_exact_keys(
        payload,
        {
            "schema_version",
            "source_identity",
            "features",
            "relations",
            "num_nodes_by_type",
            "tensor_sha256",
        },
        "projection payload",
    )
    if root.get("schema_version") != ACM_PROJECTION_SCHEMA:
        raise ValueError("unsupported ACM projection payload schema")
    if _validate_source_identity(root.get("source_identity")) != manifest[
        "source_identity"
    ]:
        raise ValueError("projection payload source identity differs from manifest")
    counts_raw = _mapping_with_exact_keys(
        root.get("num_nodes_by_type"), set(ACM_NODE_TYPES), "num_nodes_by_type"
    )
    counts = {
        name: _strict_positive_int(counts_raw.get(name), f"node count {name}")
        for name in ACM_NODE_TYPES
    }
    if counts != manifest["tensor_contract"]["num_nodes_by_type"]:
        raise ValueError("projection payload node counts differ from manifest")

    features = root.get("features")
    if (
        not isinstance(features, torch.Tensor)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or features.ndim != 2
    ):
        raise ValueError("projection features must be a rank-2 CPU float32 tensor")
    features = features.detach().contiguous()
    feature_contract = manifest["tensor_contract"]["features"]
    if tuple(features.shape) != tuple(feature_contract["shape"]):
        raise ValueError("projection feature shape differs from manifest")
    if features.shape[0] != counts["paper"] or not torch.isfinite(features).all():
        raise ValueError("projection features have invalid paper count or values")
    if tensor_sha256(features) != feature_contract["sha256"]:
        raise ValueError("projection feature tensor hash differs from manifest")

    relation_values = _mapping_with_exact_keys(
        root.get("relations"), set(ACM_RELATION_NAMES), "relations"
    )
    hash_values = _mapping_with_exact_keys(
        root.get("tensor_sha256"),
        {"features", *ACM_RELATION_NAMES},
        "tensor_sha256",
    )
    if hash_values.get("features") != feature_contract["sha256"]:
        raise ValueError("projection embedded feature hash differs from manifest")
    endpoints = {
        "paper_to_author": (counts["paper"], counts["author"]),
        "author_to_paper": (counts["author"], counts["paper"]),
        "paper_to_subject": (counts["paper"], counts["subject"]),
        "subject_to_paper": (counts["subject"], counts["paper"]),
    }
    edge_indices: dict[str, torch.Tensor] = {}
    relations: dict[str, sp.csr_matrix] = {}
    for name in ACM_RELATION_NAMES:
        source_nodes, target_nodes = endpoints[name]
        relation_contract = manifest["tensor_contract"]["relations"][name]
        if hash_values.get(name) != relation_contract["sha256"]:
            raise ValueError(f"projection embedded {name} hash differs from manifest")
        edge = _validate_edge_index(
            relation_values.get(name),
            name=name,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            contract=relation_contract,
        )
        edge_indices[name] = edge
        relations[name] = _to_csr(edge, (source_nodes, target_nodes))
    for forward, reverse in (
        ("paper_to_author", "author_to_paper"),
        ("paper_to_subject", "subject_to_paper"),
    ):
        transposed = relations[forward].transpose().tocsr()
        transposed.sort_indices()
        if (transposed != relations[reverse]).nnz != 0:
            raise ValueError(f"{forward}/{reverse} are not exact reverse topologies")

    identity = str(manifest["projection_identity"])
    acm = LabelFreeACM(
        features=features,
        relations=relations,
        num_nodes_by_type=counts,
        projection_identity=identity,
        source_identity=manifest["source_identity"],
    )
    return LoadedACMProjection(acm=acm, attestation=attestation)


def load_acm_label_free(*_args: Any, **_kwargs: Any) -> LabelFreeACM:
    """Fail closed for callers that still expect implicit ``data.pt`` loading."""

    raise RuntimeError(
        "implicit ACM data.pt loading is disabled; provide a prepared projection, "
        "manifest and expected identity"
    )


def _graph_builder(acm: LabelFreeACM):
    def build(candidate: MetaPathCandidate) -> sp.csr_matrix:
        product: sp.csr_matrix | None = None
        for name in candidate.relations:
            relation = acm.relations[name]
            product = relation.copy() if product is None else (product @ relation).tocsr()
            product.sum_duplicates()
            product.eliminate_zeros()
            product.data[:] = 1.0
        if product is None:
            raise ValueError("empty meta-path")
        return product

    return build


def build_acm_candidate_bank(
    acm: LabelFreeACM,
    *,
    max_hops: int = 4,
    view_budget: int = 2,
    max_edges_multiplier_vs_manual: float | None = 2.0,
    min_target_coverage: float = 0.01,
) -> CandidateBank:
    """Enumerate actual ACM meta-path graphs under baseline graph semantics."""

    return build_candidate_bank(
        relations=ACM_SCHEMA,
        target_type="paper",
        max_hops=max_hops,
        allowed_hops=tuple(hop for hop in (2, 4, 6) if hop <= max_hops),
        graph_builder=_graph_builder(acm),
        manual_paths=ACM_MANUAL_PATHS,
        canonicalization=CanonicalizationConfig(
            symmetrize=False, add_self_loops=True, binary=True
        ),
        structural_budget=StructuralBudget(
            manual_path_exempt=True,
            max_edges_multiplier_vs_manual=max_edges_multiplier_vs_manual,
            min_target_coverage=min_target_coverage,
        ),
        view_budget=view_budget,
    )


__all__ = [
    "ACM_MANUAL_PATHS",
    "ACM_NODE_TYPES",
    "ACM_PROJECTION_MANIFEST_SCHEMA",
    "ACM_PROJECTION_SCHEMA",
    "ACM_RELATION_NAMES",
    "ACM_SCHEMA",
    "ACM_SOURCE_LOGICAL_NAME",
    "LabelFreeACM",
    "LoadedACMProjection",
    "build_acm_candidate_bank",
    "canonical_hash",
    "load_acm_projection",
    "load_acm_label_free",
    "sha256_file",
    "tensor_sha256",
]
