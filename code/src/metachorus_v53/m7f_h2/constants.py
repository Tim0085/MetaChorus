"""Frozen identities and protocol limits for M7F-H2 AMiner."""

from __future__ import annotations

from metachorus_v53.m6f_m7f_fast.constants import (
    CLASSIFICATION_SPLIT_SEEDS,
    EVAL_MODEL_SEEDS,
    PRELIM_KMEANS_SEEDS,
    SEARCH_KMEANS_SEEDS,
    SEARCH_MODEL_SEEDS,
)

TASKBOOK_FILENAME = 'protocol_m7f-h2.md'
TASKBOOK_SHA256 = "8c0f131f8011f4979167a06982b517e00150de7c61c434ec8311a003498a943b"

PARENT_PACKAGE_FILENAME = "20260814_1900_metachorus_v53_m7f_h1_imdb_trajectory_budget_MIN_d2941bd8.zip"
PARENT_PACKAGE_SHA256 = "d4720d5c13851ec1af012a845bb432e946a22e349dfb12020ec8892064cf190e"
PARENT_PACKAGE_SIZE_BYTES = 158_288
PARENT_SOURCE_COMMIT = "d2941bd85836d4cb3704f54b564d3b00c3d36db9"
PARENT_SOURCE_TREE = "21283599db699c0a92d011a85e0130519229c721"
PARENT_TERMINAL_STATUS = "M7F_H1_PARTIAL"

DATASET = "aminer"
DATASET_IDENTITY = "53ae58ac944865b9c031899acc7f5e19a252e1aa67674da0b641f1c6d1ba06f3"
STRUCTURAL_DATA_IDENTITY = "2681a474b7db35cb9afcb727ff876da31c006dcd89bdbde6f8066e8da576d28d"
PROCESSED_DATA_SHA256 = "ff6306ec826c3dc81eef5b5443ab1b20f9737344ad1365cbdc06bebd3451c885"
CACHE_IDENTITY = "adf7444feff4d381c8f8eec8c6e3dff541547b375e0b055cb21d0256448ac2d5"
PATH_LIBRARY_IDENTITY = "431ae3b1bfe1409b61d5b3db26e3c89785204555798409cb3c2f39311d372ec1"
ALL_SETS_IDENTITY = "c94fb4b1d00e24a472e071c2939d953306d4d8ec22ac353bdf1302f4d008196e"
TARGET_NODE_TYPE = "paper"
TARGET_NODE_COUNT = 6_564
FEATURE_SHAPE = (6_564, 128)
CLASS_COUNT = 4
LEGAL_PATH_COUNT = 14
SET_CARDINALITY = 2
LEGAL_SET_COUNT = 91

MANUAL_SET_ID = "fast_aminer_000_ed3a05ce6bb8f245"
MANUAL_SET_HASH = "ed3a05ce6bb8f24554c9fd3cde6eee27802ade3aae0b29523e8459ff2ad8410b"
SHORTEST_CANARY_SET_ID = "fast_aminer_002_01a5c303479c8337"
SHORTEST_CANARY_SET_HASH = "01a5c303479c83372edfd6f7f34af53529f67264603fccefedef6e777f5870a8"
LONGEST_CANARY_SET_ID = "fast_aminer_086_161f282b83a23ffb"
LONGEST_CANARY_SET_HASH = "161f282b83a23ffb9af85dd5e2f5b2800ed74111dbcd1f8bdaacd06f7278378d"
CURRENT_AUTOMATIC_STATUS = "NOT_AVAILABLE_EXACT_IDENTITY"

FROZEN_SEED_NAMESPACE_IDENTITY = "ae8677f2675ebab870fa5ee7796cca3021a0635579fe0193e88fcfb3881addc6"
LABEL_ROLE_MANIFEST_IDENTITY = "e7b5df5fad8db1054c2c06a4cb02cc1f998fdfde66da16adf757b2ba87833a63"
EVALUATION_MODEL_SEEDS = EVAL_MODEL_SEEDS
AMINER_GLOBAL_SEARCH_POOL_SEED = 2470668472
AMINER_CLASSIFICATION_SPLIT_SEEDS = CLASSIFICATION_SPLIT_SEEDS[DATASET]
GLOBAL_SEARCH_POOL_COUNT = 1_312
PRELIM_LOCKED_REPORT_COUNT = 5_252
AMINER_SEARCH_MOUNT_FILENAME = "aminer_global_search_pool.npz"
AMINER_SEARCH_MOUNT_SIZE_BYTES = 13_252
AMINER_SEARCH_MOUNT_SHA256 = "a558434b2695e1ce50886b87c299ff281cffda06c71b803105060e16c09506ae"
AMINER_LABEL_SOURCE_RELATIVE_PATH = "aminer/labels.npy"
AMINER_LABEL_SOURCE_SIZE_BYTES = 52_640
AMINER_LABEL_SOURCE_SHA256 = "51a0500d6f74ee8745fc2d0f85c38dcdba064878980b890d5a347f29d68b531d"

CANARY_ENDPOINT = 20
SEARCH_ENDPOINTS = (300, 1000, 2000, 3000)
PRELIM_ENDPOINT = 3000
EXPECTED_GPU_COUNT = 4
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 4090"
MAX_ATTEMPTS_PER_IDENTITY = 2
COMMON_ROOT_CAUSE_PAUSE_THRESHOLD = 2

RESOURCE_LIMITS = {
    "soft_gpu_hours": 40.0,
    "hard_gpu_hours": 45.0,
    "soft_four_gpu_critical_path_hours": 11.0,
    "hard_four_gpu_critical_path_hours": 14.0,
    "soft_remote_storage_bytes": 8 * 1024**3,
    "hard_remote_storage_bytes": 10 * 1024**3,
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
    "src/metachorus_v53/m6f_m7f_fast/protocol.py",
)

REQUIRED_TEST_FILES = (
    "test_parent_identity.py",
    "test_4gpu_attestation.py",
    "test_91_sets_exact.py",
    "test_fixed_cardinality_two.py",
    "test_search_seed_list_exact.py",
    "test_eval_seed_list_exact.py",
    "test_all_sets_resolve.py",
    "test_four_canaries.py",
    "test_one_process_one_gpu.py",
    "test_r1_273_runs.py",
    "test_high_recall_frozen_implementation.py",
    "test_strict_continuation.py",
    "test_training_no_labels.py",
    "test_selection_seal_before_eval.py",
    "test_manual_reuse_exact.py",
    "test_selected_vs_manual_pairing.py",
    "test_no_second_choice.py",
    "test_minimal_package_excludes_large_files.py",
    "test_checksums.py",
)

MINIMAL_MEMBER_NAMES = (
    "AMINER_H2_PARENT_IDENTITY.json",
    "AMINER_H2_METHOD_FREEZE.json",
    "AMINER_H2_4GPU_ATTESTATION.json",
    "AMINER_H2_4GPU_ATTESTATION.csv",
    "AMINER_H2_ALL_SET_RESOLUTION.csv",
    "AMINER_H2_CANARY_RESULT.json",
    "AMINER_H2_R1_SUMMARY.csv",
    "AMINER_H2_R1_HIGH_RECALL_POOL.csv",
    "AMINER_H2_R2_METRICS_BY_SEED.csv",
    "AMINER_H2_R3_METRICS_BY_SEED.csv",
    "AMINER_H2_R3_ROBUST_SHORTLIST.csv",
    "AMINER_H2_R4_METRICS_BY_SEED.csv",
    "AMINER_H2_R4_MANUAL_SAFETY.csv",
    "AMINER_H2_R4_ROBUST_PARETO.csv",
    "AMINER_H2_SELECTED_SET.json",
    "AMINER_H2_SELECTION_DECISION.md",
    "AMINER_H2_SELECTION_SEAL.json",
    "AMINER_H2_MANUAL_REUSE_AUDIT.csv",
    "AMINER_H2_PRELIM_METRICS_BY_SEED.csv",
    "AMINER_H2_PRELIM_SUMMARY.csv",
    "AMINER_H2_SELECTED_VS_MANUAL.csv",
    "AMINER_H2_FINAL_DECISION.json",
    "AMINER_H2_FINAL_DECISION.md",
    "AMINER_H2_FINAL_SET_REGISTRY.csv",
    "AMINER_H2_RUN_REGISTRY_MINIMAL.csv",
    "AMINER_H2_RESOURCE_SUMMARY.csv",
    "AMINER_H2_ERROR_SUMMARY.csv",
    "REMOTE_AMINER_H2_ARTIFACT_RETENTION_MANIFEST.csv",
    "UPDATED_FIVE_DATASET_STATUS_TABLE.csv",
    "UPDATED_FIVE_DATASET_STATUS_TABLE.md",
    "AMINER_H2_GIT_AND_CONFIG_IDENTITY.json",
    "TEST_RECEIPT_AMINER_H2.json",
    "TEST_WARNING_REGISTRY_AMINER_H2.csv",
    "AMINER_10SEED_EXPANSION_PLAN_PENDING.md",
    "AMINER_10SEED_EXPANSION_MATRIX_PENDING.csv",
    "AMINER_10SEED_APPROVAL_STATUS.json",
    "AMINER_H2_VALIDATION.json",
    "AMINER_H2_COMPLETE.json",
)

__all__ = [name for name in globals() if name.isupper()]
