"""Canonical sparse-graph normalization and graph-level deduplication.

Meta-path strings are not a valid deduplication key: two reverse-looking paths
may induce different target-node graphs, while syntactically different paths
may induce the same graph.  This module makes the graph used by training the
source of truth and hashes its canonical sparse representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Hashable, Iterable, Mapping

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class CanonicalizationConfig:
    """Parameters for sparse graph canonicalization."""

    symmetrize: bool = False
    add_self_loops: bool = False
    binary: bool = True
    value_decimals: int | None = None


@dataclass(frozen=True)
class CanonicalGraph:
    adjacency: sp.csr_matrix
    canonical_hash: str


@dataclass(frozen=True)
class EquivalentGraphGroup:
    """All path identifiers that induce exactly the same canonical graph."""

    canonical_hash: str
    representative: Hashable
    equivalent_paths: tuple[Hashable, ...]
    adjacency: sp.csr_matrix


def _as_csr(graph: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    if sp.issparse(graph):
        matrix = graph.tocsr(copy=True)
    else:
        array = np.asarray(graph)
        if array.ndim != 2:
            raise ValueError("graph must be a rank-2 matrix")
        matrix = sp.csr_matrix(array)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("a target-node meta-path graph must be square")
    return matrix


def canonicalize_sparse_graph(
    graph: sp.spmatrix | np.ndarray,
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> sp.csr_matrix:
    """Apply the declared training rules and return a stable CSR matrix."""

    matrix = _as_csr(graph).astype(np.float64, copy=False)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    if config.symmetrize:
        matrix = matrix.maximum(matrix.transpose()).tocsr()
    if config.add_self_loops:
        matrix = (matrix + sp.eye(matrix.shape[0], dtype=np.float64, format="csr")).tocsr()
    matrix.sum_duplicates()
    if config.binary:
        matrix.data[:] = 1.0
    elif config.value_decimals is not None:
        matrix.data = np.round(matrix.data, decimals=config.value_decimals)
    if not np.isfinite(matrix.data).all():
        raise ValueError("canonical graph contains NaN or infinity")
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def sparse_graph_hash(graph: sp.csr_matrix) -> str:
    """Hash shape, coordinates and values using platform-independent dtypes."""

    matrix = graph.tocoo(copy=True)
    order = np.lexsort((matrix.col, matrix.row))
    row = np.asarray(matrix.row[order], dtype="<i8")
    col = np.asarray(matrix.col[order], dtype="<i8")
    data = np.asarray(matrix.data[order], dtype="<f8")
    
    data[data == 0] = 0.0
    shape = np.asarray(matrix.shape, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"MetaChorusCanonicalGraph-v1\0")
    digest.update(shape.tobytes(order="C"))
    digest.update(row.tobytes(order="C"))
    digest.update(col.tobytes(order="C"))
    digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def canonical_graph(
    graph: sp.spmatrix | np.ndarray,
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> CanonicalGraph:
    adjacency = canonicalize_sparse_graph(graph, config)
    return CanonicalGraph(adjacency=adjacency, canonical_hash=sparse_graph_hash(adjacency))


def sparse_graphs_equal(left: sp.spmatrix, right: sp.spmatrix) -> bool:
    """Exact equality after canonical CSR cleanup."""

    a = canonicalize_sparse_graph(left, CanonicalizationConfig(binary=False))
    b = canonicalize_sparse_graph(right, CanonicalizationConfig(binary=False))
    if a.shape != b.shape:
        return False
    difference = (a != b).nnz
    return difference == 0


def deduplicate_canonical_graphs(
    graphs: Mapping[Hashable, sp.spmatrix | np.ndarray]
    | Iterable[tuple[Hashable, sp.spmatrix | np.ndarray]],
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> tuple[EquivalentGraphGroup, ...]:
    """Group canonical adjacencies, resolving hash collisions by comparison."""

    items = graphs.items() if isinstance(graphs, Mapping) else graphs
    buckets: dict[str, list[tuple[Hashable, sp.csr_matrix]]] = {}
    for path, graph in items:
        result = canonical_graph(graph, config)
        buckets.setdefault(result.canonical_hash, []).append((path, result.adjacency))

    groups: list[EquivalentGraphGroup] = []
    for digest in sorted(buckets):
        collision_groups: list[list[tuple[Hashable, sp.csr_matrix]]] = []
        for item in buckets[digest]:
            for collision_group in collision_groups:
                if sparse_graphs_equal(item[1], collision_group[0][1]):
                    collision_group.append(item)
                    break
            else:
                collision_groups.append([item])
        for collision_index, collision_group in enumerate(collision_groups):
            paths = tuple(item[0] for item in collision_group)
            group_hash = digest if collision_index == 0 else f"{digest}:{collision_index}"
            groups.append(
                EquivalentGraphGroup(
                    canonical_hash=group_hash,
                    representative=paths[0],
                    equivalent_paths=paths,
                    adjacency=collision_group[0][1],
                )
            )
    return tuple(groups)
