"""Frozen identities and protocol limits for M7F-H4."""

from __future__ import annotations

from metachorus_v53.m4a_h8.constants import S_EVAL_MODEL_SEEDS
from metachorus_v53.m6f_m7f_fast.constants import PRELIM_KMEANS_SEEDS

TASKBOOK_FILENAME = 'protocol_m7f-h4.md'
TASKBOOK_SHA256 = "b9462351c235f5e637eacf7612db9d0fa04aa2583ba1d8470524bf7d841df141"

PARENT_PACKAGE_FILENAME = "20260816_0945_metachorus_v53_m7f_h3_aminer_fix_dblp_budgeted_MIN_7b6e17f2.zip"
PARENT_PACKAGE_SHA256 = "83190385d89c75001106a8fe71de788489cdb3639bcbadc43327308f9c1c1ed1"
PARENT_PACKAGE_SIZE_BYTES = 443_010
PARENT_SOURCE_COMMIT = "7b6e17f2419fcab84ec0e57d991c16d783cce4e6"
PARENT_SOURCE_TREE = "79f17e62bab012fca5a3a396e6e63e78cf03e336"
PARENT_TERMINAL_STATUS = "M7F_H3_COMPLETE"

IMDB_SOURCE_PACKAGE_FILENAME = "20260814_0608_metachorus_v53_dblp_imdb_aminer_3seed_fast_MIN_a8902e0d.zip"
IMDB_SOURCE_PACKAGE_SHA256 = "8f7c159c8eaaf63050e88a0499ae0ed23b2a58f8986161436778fb6426084e1f"
IMDB_SOURCE_PACKAGE_SIZE_BYTES = 55_750
IMDB_SOURCE_MEMBER = "FAST_PRELIMINARY_SUMMARY.csv"

METHOD_NAME = "RS-TC-LPMSO"
TRAINER_ID = "O0-CodeFaithful"
VIEW_PIPELINE = "S0"
METHOD_FREEZE_IDENTITY = "29cff283152463d3ccc719b4cfb3aeb8004a84cdcd8d476f6c59090f20698bc5"
FROZEN_PROTOCOL_RELATIVE_PATH = "src/metachorus_v53/m6f_m7f_fast/protocol.py"
FROZEN_PROTOCOL_SHA256 = "f980f93bce6a7c5444a88f9acd7e2e2f9eb44c050d8f9b558b27d2aa65319f71"
H8_STATISTICS_RELATIVE_PATH = "src/metachorus_v53/m4a_h8/protocol.py"
H8_STATISTICS_SHA256 = "c3e97ced5e4b16086b2e81cb3b6b9bf4479d41b85d4200f37cf5652989ecd1de"
H8_STATISTICS_FUNCTION = "paired_interval"
H8_BOOTSTRAP_NAMESPACE = "TC_LPMSO_H8_PAIRED_BOOTSTRAP"
H8_BOOTSTRAP_RESAMPLES = 10_000

DATASET = "dblp"
DATASET_IDENTITY = "30aad38c6d6dcad7220aa85e7cb95b9699ed3d2d97970498750d1a0a3810a84b"
CACHE_IDENTITY = "69b3ff44a9025373e1fecc16f8c59731b8f645d1c02363c1df7cab663863a6b7"
PATH_LIBRARY_IDENTITY = "4bfcb4cf63351c8b7752441c0d2a33827b62281da85f33d27b073886cdf1cc1e"
ALL_SETS_IDENTITY = "fe7f38c77c389eb4ab79b52584904df7da6720d731de6555370bb53c3723b80c"
TARGET_NODE_TYPE = "author"
TARGET_NODE_COUNT = 4_057
CLASS_COUNT = 4
LEGAL_PATH_COUNT = 12
SET_CARDINALITY = 3
LEGAL_SET_COUNT = 220

SELECTED_SET_ID = "fast_dblp_057_81d0a8a9e8b5b789"
SELECTED_SET_HASH = "81d0a8a9e8b5b789451c30ac69b16b8479f548ed72aad426e9f1220a2a1b1c91"
SELECTED_SET_ORDINAL = 57
SELECTED_PATH_IDS = (
    "author_to_paper::paper_to_author::author_to_paper::paper_to_author",
    "author_to_paper::paper_to_conference::conference_to_paper::paper_to_author",
    "author_to_paper::paper_to_author::author_to_paper::paper_to_conference::conference_to_paper::paper_to_author",
)
MANUAL_SET_ID = "fast_dblp_010_e7c3ad780037d0be"
MANUAL_SET_HASH = "e7c3ad780037d0be26082d7e7903e6bbff736a9b220bb9795627c00fac45aa06"
MANUAL_SET_ORDINAL = 10
MANUAL_PATH_IDS = (
    "author_to_paper::paper_to_author",
    "author_to_paper::paper_to_conference::conference_to_paper::paper_to_author",
    "author_to_paper::paper_to_term::term_to_paper::paper_to_author",
)
CURRENT_AUTOMATIC_STATUS = "NOT_AVAILABLE_EXACT_IDENTITY"

CONFIRMATION_MODEL_SEEDS = tuple(S_EVAL_MODEL_SEEDS)
HISTORICAL_SEARCH_MODEL_SEEDS = (3139726158, 3297290585, 2204985712)
HISTORICAL_PRELIMINARY_MODEL_SEEDS = (3276665818, 1910071238, 610349320)
CONFIRMATION_ROLES = ("DBLP_SELECTED", "DBLP_MANUAL")
CANARY_ENDPOINT = 20
CONFIRMATION_ENDPOINT = 3000
CONFIRMATION_CHECKPOINT_SCHEDULE = (1000, 2000, 3000)
EXPECTED_GPU_COUNT = 4
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 4090"
MAX_ATTEMPTS_PER_IDENTITY = 2
COMMON_ROOT_CAUSE_PAUSE_THRESHOLD = 2

FROZEN_SEED_NAMESPACE_IDENTITY = "ae8677f2675ebab870fa5ee7796cca3021a0635579fe0193e88fcfb3881addc6"
LABEL_ROLE_MANIFEST_IDENTITY = "e7b5df5fad8db1054c2c06a4cb02cc1f998fdfde66da16adf757b2ba87833a63"
LABEL_ROLE_SEAL_IDENTITY = "7ad7b854d0ada0187152fb9b0c0d9092756579fafdb56f42af4f276f91421751"
GLOBAL_SEARCH_POOL_COUNT = 811
PRELIM_LOCKED_REPORT_COUNT = 3_246
DBLP_SEARCH_MOUNT_FILENAME = "dblp_global_search_pool.npz"
DBLP_SEARCH_MOUNT_SIZE_BYTES = 8_907
DBLP_SEARCH_MOUNT_SHA256 = "171efc1002e13e9adbf280fc290bd09a917925623c06e70ae48e994877394ac5"
DBLP_LABEL_SOURCE_RELATIVE_PATH = "dblp/labels.npy"
DBLP_LABEL_SOURCE_SIZE_BYTES = 32_584
DBLP_LABEL_SOURCE_SHA256 = "817efdc3ebb9e96067b8770dad5894b06ecabde2ac95d4b17443985f0439fda7"
KMEANS_SEEDS = tuple(PRELIM_KMEANS_SEEDS)

RESOURCE_LIMITS = {
    "soft_gpu_hours": 10.0,
    "hard_gpu_hours": 18.0,
    "soft_four_gpu_critical_path_hours": 3.5,
    "hard_four_gpu_critical_path_hours": 6.0,
    "soft_remote_storage_bytes": 5 * 1024**3,
    "hard_remote_storage_bytes": 8 * 1024**3,
    "soft_failed_attempts": 3,
    "hard_failed_attempts": 6,
    "hard_unrecovered_identities": 2,
    "target_package_bytes": 20 * 1024**2,
    "hard_package_bytes": 50 * 1024**2,
}

SEED_ROLE_PUBLIC_ARTIFACTS = (
    "FAST_SEED_NAMESPACE_MANIFEST.json",
    "FAST_SEED_COLLISION_AUDIT.json",
    "FAST_LABEL_ROLE_MANIFEST.json",
    "FAST_LABEL_ROLE_SEAL.json",
    "FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json",
)

REQUIRED_TEST_FILES = (
    "test_parent_package_hash.py",
    "test_method_freeze_identity.py",
    "test_protocol_sha.py",
    "test_selected_set_exact.py",
    "test_manual_set_exact.py",
    "test_confirmation_seed_list_exact.py",
    "test_confirmation_seeds_unused_in_dblp.py",
    "test_matrix_exact20.py",
    "test_pair_same_seed_same_gpu_domain.py",
    "test_training_from_zero.py",
    "test_training_no_labels.py",
    "test_locked_role_contract.py",
    "test_kmeans_seed_manifest_exact.py",
    "test_metrics_formula.py",
    "test_h8_statistics_implementation_exact.py",
    "test_paired_delta_formula.py",
    "test_confirmed_improvement_gate.py",
    "test_confirmed_noninferior_gate.py",
    "test_manual_fallback_rule.py",
    "test_no_second_choice.py",
    "test_imdb_main_table_source.py",
    "test_aminer_fallback_preserved.py",
    "test_final_five_dataset_table.py",
    "test_minimal_package_excludes_large_files.py",
    "test_checksums.py",
)

MINIMAL_MEMBER_NAMES = (
    "M7F_H4_PARENT_IDENTITY.json",
    "M7F_H4_METHOD_FREEZE.json",
    "M7F_H4_DBLP_GPU_ATTESTATION.json",
    "M7F_H4_DBLP_PATH_SET_RESOLUTION.csv",
    "FIVE_DATASET_STATUS_TABLE_PRE_DBLP_CONFIRMATION.csv",
    "DBLP_H4_CONFIRMATION_SEED_USAGE_AUDIT.csv",
    "DBLP_H4_CONFIRMATION_MATRIX.csv",
    "DBLP_H4_CONFIRMATION_MATRIX_SEAL.json",
    "DBLP_H4_CANARY_RESULTS.json",
    "DBLP_H4_RUN_REGISTRY_MINIMAL.csv",
    "DBLP_H4_RESOURCE_SUMMARY.csv",
    "DBLP_H4_ERROR_SUMMARY.csv",
    "DBLP_H4_LOCKED_CLASSIFICATION_METRICS.csv",
    "DBLP_H4_LOCKED_CLUSTERING_METRICS_SUMMARY.csv",
    "DBLP_H4_LOCKED_METRICS_BY_MODEL_SEED.csv",
    "DBLP_H4_PAIRWISE_DELTAS.csv",
    "DBLP_H4_PAIRED_INTERVALS.csv",
    "DBLP_H4_STATISTICS_IDENTITY.json",
    "DBLP_H4_CONFIRMATION_DECISION.json",
    "DBLP_H4_CONFIRMATION_DECISION.md",
    "DBLP_H4_FINAL_SET_REGISTRY.csv",
    "DBLP_H4_FINAL_METHOD_FREEZE.json",
    "FIVE_DATASET_FINAL_STATUS_TABLE.csv",
    "FIVE_DATASET_FINAL_STATUS_TABLE.md",
    "REMOTE_M7F_H4_ARTIFACT_RETENTION_MANIFEST.csv",
    "M7F_H4_GIT_AND_CONFIG_IDENTITY.json",
    "TEST_RECEIPT_M7F_H4.json",
    "TEST_WARNING_REGISTRY_M7F_H4.csv",
    "M7F_H4_VALIDATION.json",
    "M7F_H4_COMPLETE.json",
)

__all__ = [name for name in globals() if name.isupper()]
