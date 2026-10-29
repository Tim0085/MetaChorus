"""Frozen public dataset/schema identities for DBLP, IMDb and AMiner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from metachorus_v53.audit.common import canonical_json_hash


@dataclass(frozen=True)
class RelationSpec:
    name: str
    source_type: str
    target_type: str
    pyg_relation: tuple[str, str, str]


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    loader: str
    source_relative_root: str
    processed_relative_path: str
    processed_sha256: str
    processed_size_bytes: int
    label_relative_path: str
    label_sha256: str
    label_count: int
    label_dtype: str
    class_count: int
    target_node_type: str
    feature_shape: tuple[int, int]
    node_counts: Mapping[str, int]
    relations: tuple[RelationSpec, ...]
    manual_paths: tuple[tuple[str, ...], ...]
    view_budget: int

    @property
    def dataset_identity_basis(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "loader": self.loader,
            "source_relative_root": self.source_relative_root,
            "processed_relative_path": self.processed_relative_path,
            "processed_sha256": self.processed_sha256,
            "processed_size_bytes": self.processed_size_bytes,
            "label_relative_path": self.label_relative_path,
            "label_sha256": self.label_sha256,
            "label_count": self.label_count,
            "label_dtype": self.label_dtype,
            "class_count": self.class_count,
            "target_node_type": self.target_node_type,
            "feature_shape": list(self.feature_shape),
            "node_counts": dict(sorted(self.node_counts.items())),
            "relations": [
                {
                    "name": row.name,
                    "source_type": row.source_type,
                    "target_type": row.target_type,
                    "pyg_relation": list(row.pyg_relation),
                }
                for row in self.relations
            ],
            "manual_paths": [list(path) for path in self.manual_paths],
            "view_budget": self.view_budget,
            "candidate_hops": [2, 4, 6],
        }

    @property
    def structural_data_identity_basis(self) -> dict[str, object]:
        """Identify structural inputs used for training."""

        value = dict(self.dataset_identity_basis)
        for key in ("label_relative_path", "label_sha256", "label_count", "label_dtype", "class_count"):
            value.pop(key)
        return value

    @property
    def structural_data_identity(self) -> str:
        return canonical_json_hash(self.structural_data_identity_basis)

    @property
    def dataset_identity(self) -> str:
        return canonical_json_hash(self.dataset_identity_basis)


DBLP = DatasetSpec(
    dataset="dblp",
    loader="datasets.get_dataset('dblp') -> torch_geometric.datasets.DBLP",
    source_relative_root="data/dblp",
    processed_relative_path="processed/data.pt",
    processed_sha256="6ffb372d74fbe475e96554c8fb932e91c470c590c5cccdc9a51b43b5eb11262e",
    processed_size_bytes=253_333_679,
    label_relative_path="raw/labels.npy",
    label_sha256="817efdc3ebb9e96067b8770dad5894b06ecabde2ac95d4b17443985f0439fda7",
    label_count=4_057,
    label_dtype="int64",
    class_count=4,
    target_node_type="author",
    feature_shape=(4_057, 334),
    node_counts={"author": 4_057, "paper": 14_328, "term": 7_723, "conference": 20},
    relations=(
        RelationSpec("author_to_paper", "author", "paper", ("author", "to", "paper")),
        RelationSpec("paper_to_author", "paper", "author", ("paper", "to", "author")),
        RelationSpec("paper_to_term", "paper", "term", ("paper", "to", "term")),
        RelationSpec("term_to_paper", "term", "paper", ("term", "to", "paper")),
        RelationSpec("paper_to_conference", "paper", "conference", ("paper", "to", "conference")),
        RelationSpec("conference_to_paper", "conference", "paper", ("conference", "to", "paper")),
    ),
    manual_paths=(
        ("author_to_paper", "paper_to_author"),
        ("author_to_paper", "paper_to_term", "term_to_paper", "paper_to_author"),
        ("author_to_paper", "paper_to_conference", "conference_to_paper", "paper_to_author"),
    ),
    view_budget=3,
)

IMDB = DatasetSpec(
    dataset="imdb",
    loader="datasets.get_dataset('imdb') -> torch_geometric.datasets.IMDB",
    source_relative_root="data/imdb",
    processed_relative_path="processed/data.pt",
    processed_sha256="7eadbcfcc2cb4e44ffea60862543b1a119dc5b25721e4594f3eaa5313407c61b",
    processed_size_bytes=143_056_501,
    label_relative_path="raw/labels.npy",
    label_sha256="34321e008b9d7fac93942e1ec52aa5190d8e85654147c2a21d4ce0991fff4225",
    label_count=4_278,
    label_dtype="int32",
    class_count=3,
    target_node_type="movie",
    feature_shape=(4_278, 3_066),
    node_counts={"movie": 4_278, "director": 2_081, "actor": 5_257},
    relations=(
        RelationSpec("movie_to_director", "movie", "director", ("movie", "to", "director")),
        RelationSpec("director_to_movie", "director", "movie", ("director", "to", "movie")),
        RelationSpec("movie_to_actor", "movie", "actor", ("movie", "to", "actor")),
        RelationSpec("actor_to_movie", "actor", "movie", ("actor", "to", "movie")),
    ),
    manual_paths=(
        ("movie_to_director", "director_to_movie"),
        ("movie_to_actor", "actor_to_movie"),
    ),
    view_budget=2,
)

AMINER = DatasetSpec(
    dataset="aminer",
    loader="datasets.get_dataset('aminer') -> datasets.AMiner",
    source_relative_root="data/aminer",
    processed_relative_path="processed/data.pt",
    processed_sha256="ff6306ec826c3dc81eef5b5443ab1b20f9737344ad1365cbdc06bebd3451c885",
    processed_size_bytes=31_049_244,
    label_relative_path="raw/labels.npy",
    label_sha256="51a0500d6f74ee8745fc2d0f85c38dcdba064878980b890d5a347f29d68b531d",
    label_count=6_564,
    label_dtype="int64",
    class_count=4,
    target_node_type="paper",
    feature_shape=(6_564, 128),
    node_counts={"paper": 6_564, "author": 13_329, "reference": 35_890},
    relations=(
        RelationSpec("paper_to_reference", "paper", "reference", ("paper", "to", "reference")),
        RelationSpec("reference_to_paper", "reference", "paper", ("reference", "to", "paper")),
        RelationSpec("paper_to_author", "paper", "author", ("paper", "to", "author")),
        RelationSpec("author_to_paper", "author", "paper", ("author", "to", "paper")),
    ),
    manual_paths=(
        ("paper_to_reference", "reference_to_paper"),
        ("paper_to_author", "author_to_paper"),
    ),
    view_budget=2,
)

DATASET_SPECS: dict[str, DatasetSpec] = {row.dataset: row for row in (DBLP, IMDB, AMINER)}

__all__ = ["RelationSpec", "DatasetSpec", "DATASET_SPECS", "DBLP", "IMDB", "AMINER"]
