"""Frozen identities and protocol limits for M7F-H3."""

from __future__ import annotations

from metachorus_v53.m6f_m7f_fast.constants import (
    CLASSIFICATION_SPLIT_SEEDS,
    EVAL_MODEL_SEEDS,
    PRELIM_KMEANS_SEEDS,
    SEARCH_KMEANS_SEEDS,
    SEARCH_MODEL_SEEDS,
)

TASKBOOK_FILENAME = 'protocol_m7f-h3.md'
TASKBOOK_SHA256 = "1acea3f45fa5bad13ee2bcf031d37cf9ee25a6461c122159c44671ac2211d26d"

PARENT_AMINER_PACKAGE_FILENAME = "20260815_0944_metachorus_v53_m7f_h2_aminer_4gpu_full_search_MIN_c438245e.zip"
PARENT_AMINER_PACKAGE_SHA256 = "877bae95729c9c2d2debc03d33871c5ce867ddc98fe1b6e7118bc27beccd4e03"
PARENT_AMINER_PACKAGE_SIZE_BYTES = 576_572
PARENT_AMINER_SOURCE_COMMIT = "c438245e87d74a7f33ec5751c38a16d9eab7b9a4"
PARENT_AMINER_SOURCE_TREE = "a1c7c7da389aba4bbbf3c6233ccaeafc585ca95e"

PARENT_H1_PACKAGE_FILENAME = "20260814_1900_metachorus_v53_m7f_h1_imdb_trajectory_budget_MIN_d2941bd8.zip"
PARENT_H1_PACKAGE_SHA256 = "d4720d5c13851ec1af012a845bb432e946a22e349dfb12020ec8892064cf190e"
PARENT_H1_PACKAGE_SIZE_BYTES = 158_288


PARENT_PACKAGE_FILENAME = PARENT_AMINER_PACKAGE_FILENAME
PARENT_PACKAGE_SHA256 = PARENT_AMINER_PACKAGE_SHA256
PARENT_PACKAGE_SIZE_BYTES = PARENT_AMINER_PACKAGE_SIZE_BYTES
PARENT_SOURCE_COMMIT = PARENT_AMINER_SOURCE_COMMIT
PARENT_SOURCE_TREE = PARENT_AMINER_SOURCE_TREE
PARENT_TERMINAL_STATUS = "AMINER_H2_COMPLETE"

FROZEN_PROTOCOL_RELATIVE_PATH = "src/metachorus_v53/m6f_m7f_fast/protocol.py"
FROZEN_PROTOCOL_SHA256 = "f980f93bce6a7c5444a88f9acd7e2e2f9eb44c050d8f9b558b27d2aa65319f71"

AMINER_MANUAL_SET_ID = "fast_aminer_000_ed3a05ce6bb8f245"
AMINER_MANUAL_SET_HASH = "ed3a05ce6bb8f24554c9fd3cde6eee27802ade3aae0b29523e8459ff2ad8410b"
AMINER_UNSAFE_SET_ID = "fast_aminer_052_b8a985e5df519ab4"
AMINER_UNSAFE_SET_HASH = "b8a985e5df519ab46fc67905238559fa4aefed68498f2c9fb7673b2554af5f23"

DATASET = "dblp"
DATASET_IDENTITY = "30aad38c6d6dcad7220aa85e7cb95b9699ed3d2d97970498750d1a0a3810a84b"
CACHE_IDENTITY = "69b3ff44a9025373e1fecc16f8c59731b8f645d1c02363c1df7cab663863a6b7"
PATH_LIBRARY_IDENTITY = "4bfcb4cf63351c8b7752441c0d2a33827b62281da85f33d27b073886cdf1cc1e"
ALL_SETS_IDENTITY = "fe7f38c77c389eb4ab79b52584904df7da6720d731de6555370bb53c3723b80c"
TARGET_NODE_TYPE = "author"
TARGET_NODE_COUNT = 4_057
FEATURE_SHAPE = (4_057, 128)
CLASS_COUNT = 4
LEGAL_PATH_COUNT = 12
SET_CARDINALITY = 3
LEGAL_SET_COUNT = 220

MANUAL_SET_ID = "fast_dblp_010_e7c3ad780037d0be"
MANUAL_SET_HASH = "e7c3ad780037d0be26082d7e7903e6bbff736a9b220bb9795627c00fac45aa06"
SHORTEST_CANARY_SET_ID = "fast_dblp_001_3f4a5d98d4475bb7"
SHORTEST_CANARY_SET_HASH = "3f4a5d98d4475bb78b1dd7e2a126e02f71dff71a709ac2b56553d12f4876bed9"
LONGEST_CANARY_SET_ID = "fast_dblp_184_03a9e0982f22528e"
LONGEST_CANARY_SET_HASH = "03a9e0982f22528e8e9442f0db96f25b7bc228d9738ab9077b253ac3af5aa38f"
CURRENT_AUTOMATIC_STATUS = "NOT_AVAILABLE_EXACT_IDENTITY"

FROZEN_SEED_NAMESPACE_IDENTITY = "ae8677f2675ebab870fa5ee7796cca3021a0635579fe0193e88fcfb3881addc6"
LABEL_ROLE_MANIFEST_IDENTITY = "e7b5df5fad8db1054c2c06a4cb02cc1f998fdfde66da16adf757b2ba87833a63"
R1_MODEL_SEEDS = (SEARCH_MODEL_SEEDS[0],)
HIGH_FIDELITY_MODEL_SEEDS = SEARCH_MODEL_SEEDS
EVALUATION_MODEL_SEEDS = EVAL_MODEL_SEEDS
DBLP_CLASSIFICATION_SPLIT_SEEDS = CLASSIFICATION_SPLIT_SEEDS[DATASET]
GLOBAL_SEARCH_POOL_COUNT = 811
PRELIM_LOCKED_REPORT_COUNT = 3_246
DBLP_SEARCH_MOUNT_FILENAME = "dblp_global_search_pool.npz"
DBLP_SEARCH_MOUNT_SIZE_BYTES = 8_907
DBLP_SEARCH_MOUNT_SHA256 = "171efc1002e13e9adbf280fc290bd09a917925623c06e70ae48e994877394ac5"
DBLP_LABEL_SOURCE_RELATIVE_PATH = "dblp/labels.npy"
DBLP_LABEL_SOURCE_SIZE_BYTES = 32_584
DBLP_LABEL_SOURCE_SHA256 = "817efdc3ebb9e96067b8770dad5894b06ecabde2ac95d4b17443985f0439fda7"

CANARY_ENDPOINT = 20
SEARCH_ENDPOINTS = (300, 1000, 2000, 3000)
PRELIM_ENDPOINT = 3000
R1_POOL_SIZE = 24
R3_SHORTLIST_SIZE = 8
EXPECTED_GPU_COUNT = 4
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 4090"
MAX_ATTEMPTS_PER_IDENTITY = 2
COMMON_ROOT_CAUSE_PAUSE_THRESHOLD = 2

RESOURCE_LIMITS = {
    "soft_gpu_hours": 40.0,
    "hard_gpu_hours": 60.0,
    "soft_four_gpu_critical_path_hours": 11.0,
    "hard_four_gpu_critical_path_hours": 16.0,
    "soft_remote_storage_bytes": 10 * 1024**3,
    "hard_remote_storage_bytes": 14 * 1024**3,
    "soft_failed_attempts": 4,
    "hard_failed_attempts": 8,
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

TRAINING_CORE_FILES = (
    "src/metachorus_v53/training/o0_codefaithful_runner.py",
    "src/metachorus_v53/training/o0_trace_capture.py",
    "src/metachorus_v53/training/o0_checkpoint_io.py",
    "src/metachorus_v53/m6f_m7f_fast/training.py",
    "src/metachorus_v53/m6f_m7f_fast/evaluation.py",
    FROZEN_PROTOCOL_RELATIVE_PATH,
)

REQUIRED_TEST_FILES = (
    "test_parent_aminer_package_hash.py",
    "test_aminer_unsafe_history_preserved.py",
    "test_aminer_final_fallback_manual.py",
    "test_aminer_ten_seed_not_eligible.py",
    "test_resource_incremental_accounting.py",
    "test_protocol_source_hash.py",
    "test_robust_rank_scope_exact.py",
    "test_dblp_manual_cardinality_three.py",
    "test_dblp_220_sets_resolve.py",
    "test_four_4090_attestation.py",
    "test_four_canaries.py",
    "test_r1_exact220_one_seed.py",
    "test_r1_pool_exact24.py",
    "test_r2_r3_three_seed_identities.py",
    "test_r3_shortlist_exact8.py",
    "test_r4_exact24_continuations.py",
    "test_eval_seed_list_exact.py",
    "test_training_no_labels.py",
    "test_locked_report_not_used_in_search.py",
    "test_no_current_automatic_synthesis.py",
    "test_no_second_choice.py",
    "test_minimal_package_excludes_large_files.py",
    "test_checksums.py",
)

MINIMAL_MEMBER_NAMES = (
    "AMINER_H2_FINAL_FALLBACK_REGISTRY.csv",
    "AMINER_H2_FINAL_FALLBACK_DECISION.json",
    "AMINER_H2_FINAL_FALLBACK_DECISION.md",
    "AMINER_10SEED_APPROVAL_STATUS_CORRECTED.json",
    "AMINER_10SEED_EXPANSION_MATRIX_CORRECTED.csv",
    "AMINER_H2_RESOURCE_SUMMARY_CORRECTED.csv",
    "AMINER_H2_RESOURCE_ACCOUNTING_NOTE.md",
    "FIVE_DATASET_STATUS_TABLE_CORRECTED.csv",
    "FIVE_DATASET_STATUS_TABLE_CORRECTED.md",
    "M7F_H3_PROTOCOL_SCOPE_AUDIT.json",
    "M7F_H3_PROTOCOL_SCOPE_AUDIT.md",
    "M7F_H3_PARENT_IDENTITY.json",
    "M7F_H3_METHOD_FREEZE.json",
    "M7F_H3_DBLP_GPU_ATTESTATION.json",
    "M7F_H3_DBLP_GPU_ATTESTATION.csv",
    "M7F_H3_DBLP_PATH_SET_RESOLUTION.csv",
    "M7F_H3_DBLP_CANARY_RESULTS.json",
    "M7F_H3_DBLP_CANARY_RUN_REGISTRY.csv",
    "DBLP_H3_R1_METRICS.csv",
    "DBLP_H3_R1_POOL_COMPONENTS.csv",
    "DBLP_H3_R1_POOL.csv",
    "DBLP_H3_R1_POOL_SEAL.json",
    "DBLP_H3_R3_OBJECTIVES.csv",
    "DBLP_H3_R3_SHORTLIST.csv",
    "DBLP_H3_R3_SHORTLIST_SEAL.json",
    "DBLP_H3_R4_METRICS.csv",
    "DBLP_H3_R4_MANUAL_SAFETY.csv",
    "DBLP_H3_R4_PARETO.csv",
    "DBLP_H3_SELECTED_SET.json",
    "DBLP_H3_SELECTION_DECISION.md",
    "DBLP_H3_SELECTION_SEAL.json",
    "DBLP_H3_PRELIMINARY_METRICS_BY_SEED.csv",
    "DBLP_H3_PRELIMINARY_SUMMARY.csv",
    "DBLP_H3_SELECTED_VS_MANUAL.csv",
    "DBLP_H3_FINAL_DECISION.json",
    "DBLP_H3_FINAL_SET_REGISTRY.csv",
    "FIVE_DATASET_STATUS_TABLE_AFTER_DBLP.csv",
    "FIVE_DATASET_STATUS_TABLE_AFTER_DBLP.md",
    "DBLP_10SEED_CONFIRMATION_PLAN_PENDING.md",
    "DBLP_10SEED_CONFIRMATION_MATRIX_PENDING.csv",
    "DBLP_10SEED_APPROVAL_STATUS.json",
    "M7F_H3_RUN_REGISTRY_MINIMAL.csv",
    "M7F_H3_RESOURCE_SUMMARY.csv",
    "M7F_H3_ERROR_SUMMARY.csv",
    "REMOTE_M7F_H3_ARTIFACT_RETENTION_MANIFEST.csv",
    "M7F_H3_GIT_AND_CONFIG_IDENTITY.json",
    "TEST_RECEIPT_M7F_H3.json",
    "TEST_WARNING_REGISTRY_M7F_H3.csv",
    "M7F_H3_VALIDATION.json",
    "M7F_H3_COMPLETE.json",
)

__all__ = [name for name in globals() if name.isupper()]
