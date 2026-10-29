"""Frozen identities and gates for H8 locked S_eval confirmation."""

from __future__ import annotations

TASKBOOK_FILENAME = 'protocol_m4a-h8.md'
TASKBOOK_SHA256 = "b0f5651eff3c8eeb01d216a07bb3fa916b5d789b1c9c25857624ec3c2b7d63fd"

PARENT_PACKAGE_FILENAME = (
    "20260812_1915_metachorus_v53_m4a_h7_robust_shortlist_r4_0c98cbef.zip"
)
PARENT_PACKAGE_SHA256 = "204ea4f2ab3123644a7448e0d2742417f78b2083898a56e3747465ca5b1d1346"
PARENT_PACKAGE_SIZE_BYTES = 38_450_323
PARENT_SOURCE_COMMIT = "0c98cbefec357485d85578ac1998a1ce5d7ca075"
PARENT_SOURCE_TREE = "f6d3cb9e6246a944571486f25bb25ce17ea5ac95"
PARENT_TERMINAL_STATUS = "TC_LPMSO_H7_NEW_SET_SELECTED"
PARENT_REGISTRY_IDENTITY = "c9d34b405b8a4e05e8cca5976110545d18e7120828ce96817bfd86ef9337a5ea"
PARENT_SELECTION_IDENTITY = "01de200bd74ca1fb6855542374c4f05f8a27bf24c672e7cc354094a968f84b98"

ROLE_MANIFEST_SHA256 = "d2c740ef11818d7a2b51dd5f1b95d50ef3ffdadf54805032f632f0e8e5493004"
ROLE_SEAL_SHA256 = "7074834d03d47105e7c9c6cfbcf79e6d7ab983e0a51e9cc207bd25ccf76f9ae2"
ROLE_SEAL_IDENTITY = "0ecc656964d050afd79b0c6d02504747811bbcd1b39941fd09c39dc18108cd35"
EVALUATOR_CONTRACT_SHA256 = "e66ac71ea5f818406b34ed90ad6fc61a974c58e25118bd229041e4851f318a17"
EVALUATOR_SEAL_SHA256 = "b6acc788883dbdf0a3117c7f982d5c8c1b9b126508dedd4ba26057308816954e"

FROZEN_CACHE_IDENTITIES = {
    "acm": "c7101a6ffca7975e1ae83acbf234df7ac59428c499c1c1f4da95741913e05d19",
    "freebase": "5c7ef95ae2fa68bab116d36aa81599709ee347889bf5da036230a86e04901c1f",
}
FREEBASE_CANDIDATE_BANK_SHA256 = "5b643dcae2f90ad5a7588a3ebb65b2036ec58e109dc56f6c743094e8c56c74e5"
FREEBASE_PATH_LIBRARY_SHA256 = "66ad85bffed63d8b2093ef468b4453a9c1bb73a2ad7de4a0437481c4baf9d143"
FREEBASE_H7_SELECTED_PROJECTION_HASHES = (
    "16118f46f333e3809c6b88d4c5d651e8ae3ef157d287a4bfb4fa22f6918ee073",
    "1debb04338dd1dc59fddeddd9ab3e716dd3aaacdd0fc77e312fb41150931dac6",
    "4f9feffc3af2fe4bba58bac930610b6c811115dd749254a96644ca47c14d5398",
)
FROZEN_PROJECTION_IDENTITIES = {
    "acm": "a74880f6993ed06cef472b3a8561e2f29ff32d77919c4bbbd716040a25a55774",
    "freebase": "cc0c66778a30d75df64765f205d5f28adfaf9f014f4c34e7bea8e4e2ba9799ce",
}


FROZEN_CLASS_COUNTS = {"acm": 3, "freebase": 3}

S_EVAL_MODEL_SEEDS = (
    1803939233,
    1780645700,
    1630573285,
    446298291,
    890281712,
    1939954495,
    1925711465,
    716773178,
    1430108399,
    1364270667,
)
DEVELOPMENT_MODEL_SEEDS = (1985123271, 508898167, 1950017805)
H5_CLUSTER_SEARCH_SEEDS = (
    2475652238,
    937043980,
    3874401161,
    177710587,
    2059122163,
    1664460139,
    1440784297,
    3136888937,
    4113260635,
    2135956355,
    3652717359,
    3998682334,
    816195535,
    2169778027,
    71998743,
    2406372044,
    409533668,
    3817936429,
    3908076876,
    3200116211,
)
H5_CLASSIFICATION_SPLIT_SEEDS = (
    2177582004,
    1767887828,
    3245677495,
    3131071785,
    3665927310,
    2680317059,
    2347186457,
    93052103,
    1911673430,
    2371119831,
)
H5_BANK_SEEDS = (3564682821, 33007483, 3919009015)

FROZEN_PATH_SETS = (
    {
        "dataset": "acm",
        "role": "H7_SELECTED",
        "candidate_set_id": "set_000__00-01",
        "candidate_set_hash": "8291c32669df0184a9ca1af7161850218f19347130b10a52f18e7885db2ff0fd",
        "candidate_set_ordinal": 0,
        "path_ids": (
            "paper_to_author::author_to_paper",
            "paper_to_subject::subject_to_paper",
        ),
        "is_manual": True,
        "is_current_automatic": False,
        "is_h7_selected": True,
    },
    {
        "dataset": "acm",
        "role": "CURRENT_AUTOMATIC",
        "candidate_set_id": "set_002__00-03",
        "candidate_set_hash": "fb76bd2a3315421c76757bb1e5a9c215666ffa9e9289839f26e8958af6f84c67",
        "candidate_set_ordinal": 2,
        "path_ids": (
            "paper_to_author::author_to_paper",
            "paper_to_author::author_to_paper::paper_to_subject::subject_to_paper",
        ),
        "is_manual": False,
        "is_current_automatic": True,
        "is_h7_selected": False,
    },
    {
        "dataset": "freebase",
        "role": "MANUAL",
        "candidate_set_id": "fbs_165bb53a4b3f67ae5d3310ea",
        "candidate_set_hash": "165bb53a4b3f67ae5d3310ea13397c05c954a51410a3ddcf632569ce74358e13",
        "candidate_set_ordinal": 0,
        "path_ids": (
            "fbp_1debb04338dd1dc59fddeddd",
            "fbp_3519c22f254da970c0120b14",
            "fbp_38b886ff4b3395533130c873",
        ),
        "is_manual": True,
        "is_current_automatic": False,
        "is_h7_selected": False,
    },
    {
        "dataset": "freebase",
        "role": "CURRENT_AUTOMATIC",
        "candidate_set_id": "fbs_8a6e01ba8154f9df2ddf25ea",
        "candidate_set_hash": "8a6e01ba8154f9df2ddf25eac73510459e2aef156f197743309b9d0b43182b8d",
        "candidate_set_ordinal": 54,
        "path_ids": (
            "fbp_072154a04aacb056540d2333",
            "fbp_1debb04338dd1dc59fddeddd",
            "fbp_63984193e0babd7476bc19dd",
        ),
        "is_manual": False,
        "is_current_automatic": True,
        "is_h7_selected": False,
    },
    {
        "dataset": "freebase",
        "role": "H7_SELECTED",
        "candidate_set_id": "fbs_d76f77491499bc567e544ca9",
        "candidate_set_hash": "d76f77491499bc567e544ca9e0d088eca75ab5a53df5aecdd8f957bdd8930442",
        "candidate_set_ordinal": 36,
        "path_ids": (
            "fbp_16118f46f333e3809c6b88d4",
            "fbp_1debb04338dd1dc59fddeddd",
            "fbp_4f9feffc3af2fe4bba58bac9",
        ),
        "is_manual": False,
        "is_current_automatic": False,
        "is_h7_selected": True,
    },
)

CANARY_ENDPOINT = 20
SCIENCE_ENDPOINT = 3000
CANARY_CHECKPOINT_SCHEDULE = (20,)
SCIENCE_CHECKPOINT_SCHEDULE = (1000, 2000, 3000)
EXPECTED_CANARY_RUNS = 2
EXPECTED_SCIENCE_RUNS = 50
EXPECTED_ACM_RUNS = 20
EXPECTED_FREEBASE_RUNS = 30
MAX_GPUS = 2
ENGINEERING_RETRIES_PER_IDENTITY = 1
COMMON_ROOT_CAUSE_PAUSE_THRESHOLD = 2
ROLLING_INTERMEDIATE_LIMIT = 1

CLASSIFIER_CONTRACT = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 5000,
    "fit_intercept": True,
    "class_weight": None,
    "feature_preprocessing": "none",
}
KMEANS_CONTRACT = {
    "n_init": 10,
    "algorithm": "lloyd",
    "repeat_count": 20,
    "fit_role": "ALL_TARGET_NODES_UNLABELED",
    "metric_role": "H5_LOCKED_REPORT",
}
BOOTSTRAP_RESAMPLES = 10_000
IMPROVEMENT_OTHER_TASK_MARGIN = -0.01
NONINFERIOR_MARGINS = {"F_cls": -0.01, "F_clu": -0.01, "F_joint": -0.005}
PER_SEED_TASK_DROP_LIMIT = -0.02

RESOURCE_LIMITS = {
    "soft_gpu_hours": 8.0,
    "hard_gpu_hours": 12.0,
    "soft_critical_path_hours": 5.0,
    "hard_critical_path_hours": 7.0,
    "soft_remote_storage_bytes": 10 * 1024**3,
    "hard_remote_storage_bytes": 15 * 1024**3,
    "soft_failed_attempts": 4,
    "hard_failed_attempts": 8,
    "target_package_bytes": 20 * 1024**2,
    "hard_package_bytes": 50 * 1024**2,
}

H8_TERMINAL_STATES = (
    "TC_LPMSO_H8_CONFIRMED_IMPROVEMENT",
    "TC_LPMSO_H8_CONFIRMED_NONINFERIOR",
    "TC_LPMSO_H8_FALLBACK_CURRENT_AUTOMATIC",
    "TC_LPMSO_H8_FALLBACK_MANUAL",
    "TC_LPMSO_H8_FAILED_PROTOCOL",
    "PARTIAL_RESOURCE_PAUSE",
    "IMMEDIATE_COMMON_ROOT_CAUSE_PAUSE",
)
