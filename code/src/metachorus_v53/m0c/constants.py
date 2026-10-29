"""Frozen identities and scientific boundaries for M0-C."""

from __future__ import annotations

M0C_TASKBOOK_SHA256 = (
    "9e0178fe1cc84fed535309522b8e2fcef1e26583b0f51e63e347aa811f6242ee"
)
M0B_TASKBOOK_SHA256 = (
    "86727239dbf680fce0db8d25d8a73526934eeed6fd145970b018beafcbf670cd"
)
M0B_FINAL_COMMIT = "5ed4f589505fcd4802531aea6bead6fd1ef0f711"
M0B_FINAL_TREE = "b4db56b68962a4c8ac65101219efcd5f6551fa10"
M0B_PARENT_COMMIT = "5fddd235cfcad8b3136c41d7b492ccbcde4aa285"
M0B_COMPLETE_SHA256 = (
    "3b2c283b648686ce9b2c8243b6e680fb39e64ef6a2c8dff238ff1d61da0a7c83"
)
M0B_VALIDATION_SHA256 = (
    "ea153a1a13eec470d9e8a1a40a548acaeec8760590bd8c38cd37d8fc1d706790"
)
M0B_REFINED_PACKAGE_SHA256 = (
    "bc80a34d75eb3ab4eae160a062beb2093da368e8753d417e1a1e996a378b14d6"
)
SANITIZED_CACHE_IDENTITY = (
    "c7101a6ffca7975e1ae83acbf234df7ac59428c499c1c1f4da95741913e05d19"
)

PAP = "paper_to_author::author_to_paper"
PSP = "paper_to_subject::subject_to_paper"
PAPSP = (
    "paper_to_author::author_to_paper::paper_to_subject::subject_to_paper"
)

PATH_SET_ROLES = {
    "Manual": (PAP, PSP),
    "Automatic": (PAP, PAPSP),
}

O0_DEFAULTS = {
    "dataset": "acm",
    "num_layers": 1,
    "hid_dim": 64,
    "lr": 1e-3,
    "tau": 0.2,
    "edge_drop_rate": 0.2,
    "feature_drop_rate": 0.2,
    "alpha": 0.5,
    "weight_decay": 0.0,
    "moving_average_decay": 0.0,
    "combine": "concat",
}

PROTECTED_SCIENCE_FILES = (
    "model.py",
    "main.py",
    "datasets.py",
    "utils.py",
)

FORBIDDEN_RUNNER_FIELDS = {
    "label",
    "labels",
    "label_cache",
    "label_path",
    "num_classes",
    "class_count",
    "svc",
    "kmeans",
    "early_stop",
    "early_stop_metric",
    "train_target",
    "evaluation",
    "eval_metric",
}

SCIENTIFIC_METRIC_TOKENS = (
    "micro_f1",
    "macro_f1",
    "nmi",
    "ari",
)
