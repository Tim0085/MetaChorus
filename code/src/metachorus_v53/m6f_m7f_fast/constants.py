"""Frozen identities, namespaces and gates for the M6F/M7F fast stage."""

from __future__ import annotations

import hashlib

from metachorus_v53.audit.common import canonical_json_bytes, canonical_json_hash
from metachorus_v53.m4a_h8.constants import (
    DEVELOPMENT_MODEL_SEEDS,
    H5_BANK_SEEDS,
    H5_CLASSIFICATION_SPLIT_SEEDS,
    H5_CLUSTER_SEARCH_SEEDS,
    S_EVAL_MODEL_SEEDS,
)

from .dataset_specs import DATASET_SPECS


TASKBOOK_FILENAME = 'protocol_m6f-m7f.md'
TASKBOOK_SHA256 = "b30d9268ec7b32ba6b0103762b5bf03e9ff94f6f0f1adcf1458f2856ff1c5c86"

PARENT_PACKAGE_FILENAME = "20260813_0703_metachorus_v53_m4a_h8_s_eval_locked_MIN_ce83ac06.zip"
PARENT_PACKAGE_SHA256 = "896d3a9970ec6897b76d9300f6c4e44fa8747dd7e9ff918a251bd3e694490235"
PARENT_PACKAGE_SIZE_BYTES = 68_241
PARENT_SOURCE_COMMIT = "ce83ac06cbc057eb03ac257527a0d5d526801ff7"
PARENT_SOURCE_TREE = "8a324542b1db804c8209fa41e22084e95acb7acf"
PARENT_TERMINAL_STATUS = "TC_LPMSO_H8_CONFIRMED_IMPROVEMENT"
H8_METHOD_FREEZE_IDENTITY = "e713bac0fe1113ae94cd53b65762843750089ae30186b42c939714d0a709d6e8"
H8_PATH_REGISTRY_IDENTITY = "1b8022fd824274a62b35aea482e5c55db6124a4fd22af5d627761c90446c16f8"

DATASET_NAMES = ("dblp", "imdb", "aminer")
DATASET_IDENTITIES = {name: DATASET_SPECS[name].dataset_identity for name in DATASET_NAMES}
DATASET_IDENTITY_BUNDLE = canonical_json_hash(DATASET_IDENTITIES)

SEED_ROLE_PUBLIC_ARTIFACTS = (
    "FAST_SEED_NAMESPACE_MANIFEST.json",
    "FAST_SEED_COLLISION_AUDIT.json",
    "FAST_LABEL_ROLE_MANIFEST.json",
    "FAST_LABEL_ROLE_SEAL.json",
    "FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json",
)

HISTORICAL_SEEDS = frozenset(
    {
        383130895,
        2009113329,
        448070258,
        1144425652,
        314799452,
        978656299,
        1576980942,
        311645484,
        794585848,
        504176662,
        118144321,
        993833131,
        *DEVELOPMENT_MODEL_SEEDS,
        *S_EVAL_MODEL_SEEDS,
        *H5_CLUSTER_SEARCH_SEEDS,
        *H5_CLASSIFICATION_SPLIT_SEEDS,
        *H5_BANK_SEEDS,
        *range(10),
    }
)


def _derive_seed(namespace: str, ordinal: int, used: set[int]) -> int:
    counter = 0
    while True:
        payload = {
            "schema_version": "metachorus.v53.m6f-m7f.seed-derivation.v1",
            "taskbook_sha256": TASKBOOK_SHA256,
            "h8_method_freeze_identity": H8_METHOD_FREEZE_IDENTITY,
            "dataset_identity_bundle": DATASET_IDENTITY_BUNDLE,
            "namespace": namespace,
            "ordinal": int(ordinal),
            "collision_counter": counter,
        }
        candidate = int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:4], "little")
        if candidate not in HISTORICAL_SEEDS and candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


_USED_DERIVED_SEEDS: set[int] = set()
SEARCH_MODEL_SEEDS = tuple(_derive_seed("COMMON_SEARCH_MODEL", i, _USED_DERIVED_SEEDS) for i in range(3))
EVAL_MODEL_SEEDS = tuple(_derive_seed("COMMON_EVAL_MODEL", i, _USED_DERIVED_SEEDS) for i in range(3))
DATASET_SPLIT_SEEDS = {
    name: _derive_seed(f"{name.upper()}_GLOBAL_POOL_SPLIT", 0, _USED_DERIVED_SEEDS)
    for name in DATASET_NAMES
}
CLASSIFICATION_SPLIT_SEEDS = {
    name: tuple(
        _derive_seed(f"{name.upper()}_WITHIN_POOL_CLASSIFICATION", i, _USED_DERIVED_SEEDS)
        for i in range(5)
    )
    for name in DATASET_NAMES
}
SEARCH_KMEANS_SEEDS = tuple(_derive_seed("COMMON_SEARCH_KMEANS", i, _USED_DERIVED_SEEDS) for i in range(20))
PRELIM_KMEANS_SEEDS = tuple(_derive_seed("COMMON_PRELIM_KMEANS", i, _USED_DERIVED_SEEDS) for i in range(20))

if len(_USED_DERIVED_SEEDS) != 3 + 3 + 3 + 15 + 20 + 20:
    raise RuntimeError("derived seed namespaces are not globally unique")
if set(SEARCH_MODEL_SEEDS).intersection(EVAL_MODEL_SEEDS):
    raise RuntimeError("search/evaluation model seeds overlap")

CLASSIFIER_CONTRACT = {
    "class": "sklearn.linear_model.LogisticRegression",
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 5000,
    "fit_intercept": True,
    "class_weight": None,
    "preprocessing": "none",
    "fit_role": "GLOBAL_SEARCH_POOL",
    "metric_role": "PRELIM_LOCKED_REPORT",
}
KMEANS_CONTRACT = {
    "class": "sklearn.cluster.KMeans",
    "algorithm": "lloyd",
    "n_init": 10,
    "fit_role": "ALL_TARGET_NODES_UNLABELED",
    "metric_role": "PRELIM_LOCKED_REPORT",
    "search_repeat_count": 20,
    "preliminary_repeat_count": 20,
}

MAX_HOPS = 6
ALLOWED_HOPS = (2, 4, 6)
CANARY_ENDPOINT = 20
SEARCH_ENDPOINTS = (300, 1000, 2000, 3000)
PRELIM_ENDPOINT = 3000
MAX_GPUS = 2
MAX_ATTEMPTS_PER_IDENTITY = 2
COMMON_ROOT_CAUSE_PAUSE_THRESHOLD = 2

RESOURCE_LIMITS = {
    "soft_gpu_hours": 20.0,
    "hard_gpu_hours": 28.0,
    "soft_critical_path_hours": 12.0,
    "hard_critical_path_hours": 16.0,
    "soft_remote_storage_bytes": 20 * 1024**3,
    "hard_remote_storage_bytes": 28 * 1024**3,
    "hard_failed_attempts": 10,
    "hard_unrecovered_identities": 2,
    "target_package_bytes": 20 * 1024**2,
    "hard_package_bytes": 50 * 1024**2,
}

SEED_SEAL_BASIS = {
    "taskbook_sha256": TASKBOOK_SHA256,
    "h8_method_freeze_identity": H8_METHOD_FREEZE_IDENTITY,
    "dataset_identities": DATASET_IDENTITIES,
    "search_model_seeds": list(SEARCH_MODEL_SEEDS),
    "evaluation_model_seeds": list(EVAL_MODEL_SEEDS),
    "dataset_split_seeds": DATASET_SPLIT_SEEDS,
    "classification_split_seeds": {key: list(value) for key, value in CLASSIFICATION_SPLIT_SEEDS.items()},
    "search_kmeans_seeds": list(SEARCH_KMEANS_SEEDS),
    "preliminary_kmeans_seeds": list(PRELIM_KMEANS_SEEDS),
    "historical_seed_count": len(HISTORICAL_SEEDS),
    "all_namespaces_pairwise_disjoint": True,
    "search_eval_model_seeds_disjoint": True,
    "derived_historical_collision_count": 0,
}
SEED_SEAL_IDENTITY = canonical_json_hash(SEED_SEAL_BASIS)

__all__ = [name for name in globals() if name.isupper()]
