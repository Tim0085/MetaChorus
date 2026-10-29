"""Schema-closed MetaChorus candidate generation and structural budgeting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import comb
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from .graph_canonicalizer import (
    CanonicalizationConfig,
    canonical_graph,
    sparse_graphs_equal,
)


@dataclass(frozen=True, order=True)
class SchemaRelation:
    name: str
    source_type: str
    target_type: str


@dataclass(frozen=True)
class MetaPathCandidate:
    relations: tuple[str, ...]
    node_types: tuple[str, ...]
    is_manual: bool = False
    canonical_graph_hash: str | None = None
    equivalent_paths: tuple[tuple[str, ...], ...] = ()
    edge_count: int = 0
    average_degree: float = 0.0
    target_coverage: float = 0.0
    
    
    
    memory_estimate_mb: float = 0.0
    csr_memory_mb: float = 0.0
    gpu_memory_estimate_mb: float = 0.0
    build_time_seconds: float = 0.0

    @property
    def identifier(self) -> str:
        return "::".join(self.relations)

    @property
    def hops(self) -> int:
        return len(self.relations)


@dataclass(frozen=True)
class StructuralBudget:
    manual_path_exempt: bool = True
    max_edges_multiplier_vs_manual: float | None = None
    max_avg_degree: float | None = None
    max_gpu_memory_estimate_mb: float | None = None
    min_target_coverage: float = 0.0


@dataclass(frozen=True)
class CandidateBank:
    target_type: str
    candidates: tuple[MetaPathCandidate, ...]
    rejected: tuple[tuple[MetaPathCandidate, tuple[str, ...]], ...] = ()
    graphs: Mapping[str, sp.csr_matrix] = field(default_factory=dict, repr=False)

    @property
    def effective_count(self) -> int:
        return len(self.candidates)

    def validate_budget(self, view_budget: int) -> None:
        if view_budget < 1:
            raise ValueError("view_budget C must be positive")
        if self.effective_count < view_budget:
            raise ValueError(
                f"candidate construction error: P={self.effective_count} is smaller "
                f"than C={view_budget}"
            )

    def number_of_sets(self, view_budget: int) -> int:
        self.validate_budget(view_budget)
        return comb(self.effective_count, view_budget)

    def by_relations(self, relations: Sequence[str]) -> MetaPathCandidate:
        key = tuple(relations)
        for candidate in self.candidates:
            if candidate.relations == key or key in candidate.equivalent_paths:
                return candidate
        raise KeyError(key)


def _relation_map(relations: Iterable[SchemaRelation]) -> dict[str, SchemaRelation]:
    result: dict[str, SchemaRelation] = {}
    for relation in relations:
        if relation.name in result:
            raise ValueError(f"duplicate schema relation name: {relation.name}")
        result[relation.name] = relation
    if not result:
        raise ValueError("schema must contain at least one relation")
    return result


def validate_meta_path(
    relation_names: Sequence[str],
    schema: Mapping[str, SchemaRelation],
    target_type: str,
) -> tuple[str, ...]:
    """Validate relation continuity and target-type closure."""

    if not relation_names:
        raise ValueError("a meta-path must contain at least one relation")
    try:
        first = schema[relation_names[0]]
    except KeyError as exc:
        raise ValueError(f"unknown relation: {relation_names[0]}") from exc
    if first.source_type != target_type:
        raise ValueError("meta-path does not start at the target node type")
    node_types = [target_type]
    current = target_type
    for relation_name in relation_names:
        try:
            relation = schema[relation_name]
        except KeyError as exc:
            raise ValueError(f"unknown relation: {relation_name}") from exc
        if relation.source_type != current:
            raise ValueError(
                f"schema discontinuity before relation {relation_name}: "
                f"expected source {current}, got {relation.source_type}"
            )
        current = relation.target_type
        node_types.append(current)
    if current != target_type:
        raise ValueError("meta-path does not return to the target node type")
    return tuple(node_types)


def enumerate_schema_closed_paths(
    relations: Iterable[SchemaRelation],
    target_type: str,
    max_hops: int,
    *,
    allowed_hops: Iterable[int] | None = None,
) -> tuple[MetaPathCandidate, ...]:
    """Enumerate all schema-valid target-to-target paths up to ``max_hops``."""

    if max_hops not in (2, 4, 6):
        raise ValueError("max_hops must be one of 2, 4, or 6")
    relation_list = tuple(sorted(relations))
    _relation_map(relation_list)
    desired = set(allowed_hops if allowed_hops is not None else range(1, max_hops + 1))
    if any(hop < 1 or hop > max_hops for hop in desired):
        raise ValueError("allowed_hops must lie in [1, max_hops]")
    outgoing: dict[str, list[SchemaRelation]] = {}
    for relation in relation_list:
        outgoing.setdefault(relation.source_type, []).append(relation)

    results: list[MetaPathCandidate] = []

    def visit(current_type: str, names: tuple[str, ...], node_types: tuple[str, ...]) -> None:
        length = len(names)
        if length in desired and current_type == target_type:
            results.append(MetaPathCandidate(names, node_types))
        if length == max_hops:
            return
        for relation in outgoing.get(current_type, ()):
            visit(
                relation.target_type,
                names + (relation.name,),
                node_types + (relation.target_type,),
            )

    visit(target_type, (), (target_type,))
    return tuple(sorted(results, key=lambda item: (item.hops, item.relations)))


def _csr_memory_mb(graph: sp.csr_matrix) -> float:
    return float(graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes) / (1024.0**2)


def _gpu_edge_index_memory_mb(graph: sp.csr_matrix) -> float:
    """Estimate GPU edge-index storage, excluding activations and workspace."""

    return float(2 * graph.nnz * np.dtype(np.int64).itemsize) / (1024.0**2)


def _candidate_with_graph_stats(
    candidate: MetaPathCandidate,
    graph: sp.csr_matrix,
    digest: str,
    equivalent_paths: tuple[tuple[str, ...], ...],
    target_coverage_before_forced_self_loops: float,
    build_time_seconds: float,
) -> MetaPathCandidate:
    n = graph.shape[0]
    csr_memory = _csr_memory_mb(graph)
    gpu_memory = _gpu_edge_index_memory_mb(graph)
    return replace(
        candidate,
        canonical_graph_hash=digest,
        equivalent_paths=equivalent_paths,
        edge_count=int(graph.nnz),
        average_degree=float(graph.nnz / n) if n else 0.0,
        
        
        
        target_coverage=target_coverage_before_forced_self_loops,
        memory_estimate_mb=gpu_memory,
        csr_memory_mb=csr_memory,
        gpu_memory_estimate_mb=gpu_memory,
        build_time_seconds=float(build_time_seconds),
    )


def _budget_reasons(
    candidate: MetaPathCandidate,
    budget: StructuralBudget,
    manual_edge_reference: int,
) -> tuple[str, ...]:
    if candidate.is_manual and budget.manual_path_exempt:
        return ()
    reasons: list[str] = []
    if (
        budget.max_edges_multiplier_vs_manual is not None
        and manual_edge_reference > 0
        and candidate.edge_count
        > budget.max_edges_multiplier_vs_manual * manual_edge_reference
    ):
        reasons.append("max_edges_multiplier_vs_manual")
    if budget.max_avg_degree is not None and candidate.average_degree > budget.max_avg_degree:
        reasons.append("max_avg_degree")
    if (
        budget.max_gpu_memory_estimate_mb is not None
        and candidate.gpu_memory_estimate_mb > budget.max_gpu_memory_estimate_mb
    ):
        reasons.append("max_gpu_memory_estimate_mb")
    if candidate.target_coverage < budget.min_target_coverage:
        reasons.append("min_target_coverage")
    return tuple(reasons)


def build_candidate_bank(
    *,
    relations: Iterable[SchemaRelation],
    target_type: str,
    max_hops: int,
    graph_builder: Callable[[MetaPathCandidate], sp.spmatrix | np.ndarray],
    manual_paths: Iterable[Sequence[str]],
    allowed_hops: Iterable[int] | None = None,
    canonicalization: CanonicalizationConfig = CanonicalizationConfig(),
    structural_budget: StructuralBudget = StructuralBudget(),
    view_budget: int | None = None,
) -> CandidateBank:
    """Construct, graph-deduplicate, budget and validate the candidate bank."""

    schema = _relation_map(relations)
    generated = list(
        enumerate_schema_closed_paths(
            schema.values(), target_type, max_hops, allowed_hops=allowed_hops
        )
    )
    by_path = {candidate.relations: candidate for candidate in generated}
    manual_set: set[tuple[str, ...]] = set()
    for path in manual_paths:
        path_tuple = tuple(path)
        node_types = validate_meta_path(path_tuple, schema, target_type)
        manual_set.add(path_tuple)
        by_path[path_tuple] = MetaPathCandidate(path_tuple, node_types, is_manual=True)
    for path in manual_set:
        by_path[path] = replace(by_path[path], is_manual=True)

    
    graph_items: list[
        tuple[MetaPathCandidate, sp.csr_matrix, str, float, float]
    ] = []
    for candidate in sorted(by_path.values(), key=lambda item: (item.hops, item.relations)):
        build_started = time.perf_counter()
        raw_graph = graph_builder(candidate)
        canonical = canonical_graph(raw_graph, canonicalization)
        
        
        pre_loop_config = replace(canonicalization, add_self_loops=False)
        pre_loop_graph = canonical_graph(raw_graph, pre_loop_config).adjacency
        node_count = pre_loop_graph.shape[0]
        covered = int(np.count_nonzero(np.diff(pre_loop_graph.indptr)))
        pre_loop_coverage = float(covered / node_count) if node_count else 0.0
        build_elapsed = time.perf_counter() - build_started
        graph_items.append(
            (
                candidate,
                canonical.adjacency,
                canonical.canonical_hash,
                pre_loop_coverage,
                build_elapsed,
            )
        )

    digest_buckets: dict[
        str, list[list[tuple[MetaPathCandidate, sp.csr_matrix, float, float]]]
    ] = {}
    for candidate, graph, digest, pre_loop_coverage, build_elapsed in graph_items:
        collision_groups = digest_buckets.setdefault(digest, [])
        for group in collision_groups:
            if sparse_graphs_equal(graph, group[0][1]):
                group.append((candidate, graph, pre_loop_coverage, build_elapsed))
                break
        else:
            collision_groups.append(
                [(candidate, graph, pre_loop_coverage, build_elapsed)]
            )

    deduplicated: list[MetaPathCandidate] = []
    graph_lookup: dict[str, sp.csr_matrix] = {}
    for digest in sorted(digest_buckets):
        for collision_index, group in enumerate(digest_buckets[digest]):
            group_hash = digest if collision_index == 0 else f"{digest}:{collision_index}"
            manual_members = [item for item in group if item[0].is_manual]
            representative_item = (manual_members or group)[0]
            representative = representative_item[0]
            equivalents = tuple(item[0].relations for item in group)
            enriched = _candidate_with_graph_stats(
                replace(representative, is_manual=bool(manual_members)),
                group[0][1],
                group_hash,
                equivalents,
                representative_item[2],
                representative_item[3],
            )
            deduplicated.append(enriched)
            graph_lookup[group_hash] = group[0][1]

    manual_edges = [candidate.edge_count for candidate in deduplicated if candidate.is_manual]
    manual_reference = max(manual_edges, default=0)
    accepted: list[MetaPathCandidate] = []
    rejected: list[tuple[MetaPathCandidate, tuple[str, ...]]] = []
    for candidate in sorted(deduplicated, key=lambda item: (item.hops, item.relations)):
        reasons = _budget_reasons(candidate, structural_budget, manual_reference)
        if reasons:
            rejected.append((candidate, reasons))
        else:
            accepted.append(candidate)

    accepted_hashes = {candidate.canonical_graph_hash for candidate in accepted}
    bank = CandidateBank(
        target_type=target_type,
        candidates=tuple(accepted),
        rejected=tuple(rejected),
        graphs={key: value for key, value in graph_lookup.items() if key in accepted_hashes},
    )
    if view_budget is not None:
        bank.validate_budget(view_budget)
    return bank
