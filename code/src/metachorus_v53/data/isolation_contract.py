"""Declarative mount contracts for training and evaluation sandboxes."""

TRAIN_SANDBOX_CONTRACT = {
    "schema_version": "metachorus.m0b.train-sandbox-contract.v1",
    "network": "none",
    "root_filesystem": "read_only",
    "code_mount": "read_only",
    "sanitized_cache_mount": "read_only",
    "output_mount": "read_write",
    "raw_data_mounted": False,
    "evaluation_information_cache_mounted": False,
    "host_project_root_mounted": False,
    "environment": {"LABEL_ACCESS": "FORBIDDEN"},
}

EVALUATION_SANDBOX_CONTRACT = {
    "schema_version": "metachorus.m0b.evaluation-sandbox-contract.v1",
    "network": "none",
    "training_artifacts_mount": "read_only",
    "evaluation_information_cache_mount": "read_only",
    "evaluation_output_mount": "read_write",
    "training_writeback_allowed": False,
}
