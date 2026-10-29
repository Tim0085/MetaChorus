"""Frozen identities and protocol limits for M8-F1."""

from __future__ import annotations

from metachorus_v53.m4a_h8.constants import S_EVAL_MODEL_SEEDS
from metachorus_v53.m6f_m7f_fast.constants import PRELIM_KMEANS_SEEDS

TASKBOOK_FILENAME = (
    'protocol_m8-f1.md'
)
TASKBOOK_SHA256 = "58af178422e6d64953df7720df7414bd130bc9d01b3c529f6201bd11475b4947"

PARENT_PACKAGE_FILENAME = (
    "20260816_2013_metachorus_v53_m7f_h4_dblp_10seed_confirm_MIN_fac2db4c.zip"
)
PARENT_PACKAGE_SHA256 = "e4411afa89416910746f529f521ebe186c7dda8da166cde9905564c84a9b1dbe"
PARENT_PACKAGE_SIZE_BYTES = 94_586
PARENT_SOURCE_COMMIT = "fac2db4ca6958a77d262170adf0dabaa7d4b8ba5"
PARENT_SOURCE_TREE = "78d592127aed5dee28949454e22ed57a07eb6465"
PARENT_TERMINAL_STATUS = "M7F_H4_COMPLETE"
PARENT_FIVE_TABLE_IDENTITY = "e58bdc56f176a3927f3da3aed9c6fff321edbe487dda6568ccb3570e58e271fb"

SUPPORT_PACKAGES = {
    "h8_acm_freebase": {
        "filename": "20260813_0703_metachorus_v53_m4a_h8_s_eval_locked_MIN_ce83ac06.zip",
        "sha256": "896d3a9970ec6897b76d9300f6c4e44fa8747dd7e9ff918a251bd3e694490235",
        "size_bytes": 68_241,
    },
    "h1_imdb_aminer_prior3": {
        "filename": "20260814_1900_metachorus_v53_m7f_h1_imdb_trajectory_budget_MIN_d2941bd8.zip",
        "sha256": "d4720d5c13851ec1af012a845bb432e946a22e349dfb12020ec8892064cf190e",
        "size_bytes": 158_288,
    },
    "h2_aminer_rejected": {
        "filename": "20260815_0944_metachorus_v53_m7f_h2_aminer_4gpu_full_search_MIN_c438245e.zip",
        "sha256": "877bae95729c9c2d2debc03d33871c5ce867ddc98fe1b6e7118bc27beccd4e03",
        "size_bytes": 576_572,
    },
}

METHOD_NAME = "RS-TC-LPMSO"
TRAINER_ID = "O0-CodeFaithful"
VIEW_PIPELINE = "S0"
METHOD_FREEZE_IDENTITY = "29cff283152463d3ccc719b4cfb3aeb8004a84cdcd8d476f6c59090f20698bc5"
FROZEN_PROTOCOL_RELATIVE_PATH = "src/metachorus_v53/m6f_m7f_fast/protocol.py"
FROZEN_PROTOCOL_SHA256 = "f980f93bce6a7c5444a88f9acd7e2e2f9eb44c050d8f9b558b27d2aa65319f71"

DATASETS = ("imdb", "aminer")
DATASET_SPECS = {
    "imdb": {
        "dataset_identity": "2647cb3a5af7e7f8e56abec700698479296d5519c53e3c3a3e63ea9e06d5cd41",
        "cache_identity": "f6d6d08699dd69ff9968cbf6ff4433ea362a629109ecf73aba941cfb2857aaca",
        "path_library_identity": "83cec3bba394a926126aa9059bfb8ba526ddd5e986ab1f17714b32e59e51c238",
        "all_sets_identity": "957639007f4368878086762cc0c01838ec23f442fca6f78d66fc0d3eb4da3d47",
        "target_node_type": "movie",
        "target_node_count": 4_278,
        "class_count": 3,
        "set_cardinality": 2,
        "legal_set_count": 45,
        "manual_set_id": "fast_imdb_000_1461aea830c29417",
        "manual_set_hash": "1461aea830c29417d7ef79320e43d337e61efb99865a0e29bbc497c4b19ccb1c",
        "manual_set_ordinal": 0,
        "manual_path_ids": (
            "movie_to_actor::actor_to_movie",
            "movie_to_director::director_to_movie",
        ),
        "rejected_set_id": "fast_imdb_014_a91a43294eadd6bc",
        "rejected_set_hash": "a91a43294eadd6bc44b9f18af021be977f7edd7a3698a4eafd0a70b0b83c2280",
        "global_search_pool_count": 855,
        "global_search_pool_indices_sha256": "4b5d45b5e5ce1fc3e28d5015b5e3c0d510cfb66e3c76325bf5ec9074ab753489",
        "prelim_locked_report_count": 3_423,
        "prelim_locked_report_indices_sha256": "d06267b35d08305d540f1d416614575e6ba78e3b16af85243a992fe9d51dca0a",
        "search_mount_filename": "imdb_global_search_pool.npz",
        "search_mount_size_bytes": 9_332,
        "search_mount_sha256": "0dbe6cd8ba7ff936056ee2e31b3a34aef4f95036aa1497a8d093147015179ef8",
        "label_source_relative_path": "imdb/labels.npy",
        "label_source_size_bytes": 17_240,
        "label_source_sha256": "34321e008b9d7fac93942e1ec52aa5190d8e85654147c2a21d4ce0991fff4225",
    },
    "aminer": {
        "dataset_identity": "53ae58ac944865b9c031899acc7f5e19a252e1aa67674da0b641f1c6d1ba06f3",
        "cache_identity": "adf7444feff4d381c8f8eec8c6e3dff541547b375e0b055cb21d0256448ac2d5",
        "path_library_identity": "431ae3b1bfe1409b61d5b3db26e3c89785204555798409cb3c2f39311d372ec1",
        "all_sets_identity": "c94fb4b1d00e24a472e071c2939d953306d4d8ec22ac353bdf1302f4d008196e",
        "target_node_type": "paper",
        "target_node_count": 6_564,
        "class_count": 4,
        "set_cardinality": 2,
        "legal_set_count": 91,
        "manual_set_id": "fast_aminer_000_ed3a05ce6bb8f245",
        "manual_set_hash": "ed3a05ce6bb8f24554c9fd3cde6eee27802ade3aae0b29523e8459ff2ad8410b",
        "manual_set_ordinal": 0,
        "manual_path_ids": (
            "paper_to_author::author_to_paper",
            "paper_to_reference::reference_to_paper",
        ),
        "rejected_set_id": "fast_aminer_052_b8a985e5df519ab4",
        "rejected_set_hash": "b8a985e5df519ab46fc67905238559fa4aefed68498f2c9fb7673b2554af5f23",
        "global_search_pool_count": 1_312,
        "global_search_pool_indices_sha256": "1899d546c0fcb724a03d5e657538cf8f2e77a5f0f23c0019705721499eae7654",
        "prelim_locked_report_count": 5_252,
        "prelim_locked_report_indices_sha256": "16c865be8cd924487d1cff64681cbbcb408895beebb5f18d239729b6ab7880eb",
        "search_mount_filename": "aminer_global_search_pool.npz",
        "search_mount_size_bytes": 13_252,
        "search_mount_sha256": "a558434b2695e1ce50886b87c299ff281cffda06c71b803105060e16c09506ae",
        "label_source_relative_path": "aminer/labels.npy",
        "label_source_size_bytes": 52_640,
        "label_source_sha256": "51a0500d6f74ee8745fc2d0f85c38dcdba064878980b890d5a347f29d68b531d",
    },
}

CONFIRMATION_MODEL_SEEDS = tuple(S_EVAL_MODEL_SEEDS)
CANARY_ENDPOINT = 20
FINAL10_ENDPOINT = 3_000
FINAL10_CHECKPOINT_SCHEDULE = (1_000, 2_000, 3_000)
EXPECTED_GPU_COUNT = 4
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 4090"
MAX_ATTEMPTS_PER_IDENTITY = 2
COMMON_ROOT_CAUSE_PAUSE_THRESHOLD = 2

FROZEN_SEED_NAMESPACE_IDENTITY = "ae8677f2675ebab870fa5ee7796cca3021a0635579fe0193e88fcfb3881addc6"
LABEL_ROLE_MANIFEST_IDENTITY = "e7b5df5fad8db1054c2c06a4cb02cc1f998fdfde66da16adf757b2ba87833a63"
LABEL_ROLE_SEAL_IDENTITY = "7ad7b854d0ada0187152fb9b0c0d9092756579fafdb56f42af4f276f91421751"
KMEANS_SEEDS = tuple(PRELIM_KMEANS_SEEDS)

FINAL_METHOD_ROLES = {
    "acm": "MANUAL",
    "freebase": "CONFIRMED_NEW_SET",
    "imdb": "MANUAL_FALLBACK",
    "dblp": "BUDGETED_SELECTED_SET_CONFIRMED",
    "aminer": "MANUAL_FALLBACK",
}

RESOURCE_LIMITS = {
    "soft_gpu_hours": 4.0,
    "hard_gpu_hours": 7.0,
    "soft_four_gpu_critical_path_hours": 1.5,
    "hard_four_gpu_critical_path_hours": 3.0,
    "soft_remote_storage_bytes": 4 * 1024**3,
    "hard_remote_storage_bytes": 7 * 1024**3,
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
    "test_imdb_manual_exact.py",
    "test_aminer_manual_exact.py",
    "test_confirmation_seed_list_exact.py",
    "test_seeds_unused_per_dataset.py",
    "test_matrix_exact20.py",
    "test_four_gpu_attestation.py",
    "test_training_from_zero.py",
    "test_training_no_labels.py",
    "test_imdb_label_roles.py",
    "test_aminer_label_roles.py",
    "test_preliminary_kmeans_seed_list_exact.py",
    "test_metrics_formula.py",
    "test_no_prior3_merge.py",
    "test_final_roles_immutable.py",
    "test_five_dataset_all_seed_count_10.py",
    "test_main_table_mean_sd.py",
    "test_minimal_package_excludes_large_files.py",
    "test_checksums.py",
)

MINIMAL_MEMBER_NAMES = (
    "M8_F1_PARENT_IDENTITY.json",
    "M8_F1_METHOD_FREEZE.json",
    "M8_F1_GPU_ATTESTATION.json",
    "M8_F1_PATH_SET_RESOLUTION.csv",
    "FINAL10_COMPLETION_SEED_USAGE_AUDIT.csv",
    "FINAL10_COMPLETION_MATRIX.csv",
    "FINAL10_COMPLETION_MATRIX_SEAL.json",
    "M8_F1_CANARY_RESULTS.json",
    "M8_F1_RUN_REGISTRY_MINIMAL.csv",
    "M8_F1_RESOURCE_SUMMARY.csv",
    "M8_F1_ERROR_SUMMARY.csv",
    "FINAL10_LOCKED_CLASSIFICATION_METRICS.csv",
    "FINAL10_LOCKED_CLUSTERING_METRICS_SUMMARY.csv",
    "FINAL10_METRICS_BY_MODEL_SEED.csv",
    "FINAL10_SUMMARY.csv",
    "FINAL10_INTERVALS.csv",
    "FINAL10_VS_PRIOR3_DESCRIPTIVE_AUDIT.csv",
    "REJECTED_SEARCH_SET_ABLATION_TABLE.csv",
    "REJECTED_SEARCH_SET_ABLATION_TABLE.md",
    "FIVE_DATASET_FINAL_10SEED_STATUS_TABLE.csv",
    "FIVE_DATASET_FINAL_10SEED_STATUS_TABLE.md",
    "FIVE_DATASET_PAPER_MAIN_TABLE.csv",
    "FIVE_DATASET_PAPER_MAIN_TABLE.md",
    "FIVE_DATASET_SUPPLEMENTARY_COMPOSITE_TABLE.csv",
    "FINAL_PATH_SET_REGISTRY_5DATASETS.csv",
    "FINAL_EVIDENCE_LEVEL_TABLE.csv",
    "FINAL_CLAIM_BOUNDARY.md",
    "M8_F1_FINAL_METHOD_FREEZE.json",
    "M8_F1_FINAL_EXPERIMENT_COMPLETION.json",
    "REMOTE_M8_F1_ARTIFACT_RETENTION_MANIFEST.csv",
    "M8_F1_GIT_AND_CONFIG_IDENTITY.json",
    "TEST_RECEIPT_M8_F1.json",
    "TEST_WARNING_REGISTRY_M8_F1.csv",
)

__all__ = [name for name in globals() if name.isupper()]
