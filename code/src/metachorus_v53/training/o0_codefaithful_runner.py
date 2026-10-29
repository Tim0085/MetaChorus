"""Training routines for jointly learned relation views."""

from __future__ import annotations

import ast
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import re
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from metachorus_v53.audit.common import (
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
    utc_now,
)
from metachorus_v53.data.acm_sanitized_adapter import ACMSanitizedAdapter
from metachorus_v53.data.freebase_sanitized_adapter import FreebaseSanitizedAdapter
from metachorus_v53.m6f_m7f_fast.adapter import FastSanitizedAdapter
from metachorus_v53.m1a.constants import CHECKPOINT_SCHEDULE as M1_CHECKPOINT_SCHEDULE
from metachorus_v53.m0c.constants import (
    FORBIDDEN_RUNNER_FIELDS,
    O0_DEFAULTS,
    SANITIZED_CACHE_IDENTITY,
)
from .o0_checkpoint_io import (
    load_checkpoint,
    nested_state_sha256,
    rng_state_sha256,
)
from .o0_trace_capture import O0Sidecar, TraceWriter
from .resource_monitor import current_resources, percentile_summary


@dataclass(frozen=True)
class O0RunConfig:
    dataset: str
    path_set_role: str
    path_set_id: str
    path_ids: tuple[str, ...]
    candidate_set_hash: str
    model_seed: int
    max_updates: int
    capture_mode: str
    checkpoint_schedule: tuple[int, ...]
    cache_root: str
    output_root: str
    device: str
    source_commit: str
    source_tree: str
    cache_identity: str
    augmentation_namespace: str
    run_role: str = "ENGINEERING_RESOURCE_PROBE"
    eligible_for_scientific_result: bool = False
    eligible_for_selector_development: bool = False
    trainer_id: str = "O0-CodeFaithful"
    combine: str = "concat"
    num_layers: int = 1
    hid_dim: int = 64
    lr: float = 1e-3
    tau: float = 0.2
    edge_drop_rate: float = 0.2
    feature_drop_rate: float = 0.2
    alpha: float = 0.5
    weight_decay: float = 0.0
    moving_average_decay: float = 0.0
    physical_run_id: str | None = None
    candidate_set_ordinal: int | None = None
    source_data_identity: str | None = None
    projection_identity: str | None = None
    score_execution_contract_hash: str | None = None
    diagnostic_source_identity_hash: str | None = None
    diagnostic_config_hash: str | None = None
    bank_contract_hash: str | None = None
    candidate_bank_hash: str | None = None
    plan_seal_hash: str | None = None
    precision_runtime_contract_hash: str | None = None
    
    
    
    
    representation_schedule: tuple[int, ...] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "O0RunConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown runner config fields: {sorted(unknown)}")
        forbidden = {
            key
            for key in value
            if any(
                token in key.casefold()
                for token in FORBIDDEN_RUNNER_FIELDS
            )
        }
        if forbidden:
            raise ValueError(
                f"forbidden evaluation fields in runner config: {sorted(forbidden)}"
            )
        normalized = dict(value)
        normalized["path_ids"] = tuple(normalized["path_ids"])
        normalized["checkpoint_schedule"] = tuple(
            int(item) for item in normalized["checkpoint_schedule"]
        )
        if normalized.get("representation_schedule") is not None:
            normalized["representation_schedule"] = tuple(
                int(item) for item in normalized["representation_schedule"]
            )
        return cls(**normalized)

    def resolved_representation_schedule(self) -> tuple[int, ...]:
        return (
            tuple(self.checkpoint_schedule)
            if self.representation_schedule is None
            else tuple(self.representation_schedule)
        )

    def validate(self) -> None:
        if self.dataset not in {"acm", "freebase", "dblp", "imdb", "aminer"}:
            raise ValueError("O0 runner only permits frozen MetaChorus datasets")
        if self.trainer_id != "O0-CodeFaithful":
            raise ValueError("trainer_id must be O0-CodeFaithful")
        if self.run_role not in {
            "ENGINEERING_RESOURCE_PROBE",
            "ENGINEERING_PARITY",
            "ENGINEERING_PRACTICAL_PARITY_R2",
            "ENGINEERING_DIAGNOSTIC_AUX_R2",
            "M1A_SELECTOR_DEVELOPMENT",
            "M1B0_ENGINEERING_SMOKE",
            "M1B_SELECTOR_DEVELOPMENT",
            "M1B_H2_FIXED_BUDGET_VERIFICATION",
            "M1B_H3_CHECKPOINT_CONFIRMATION",
            "M2A_S1_SMOKE",
            "M2A_S1_SCIENTIFIC",
            "M2B_S2_EQUIVALENCE",
            "M2B_S2_SCIENTIFIC",
            "M2C_PCPA_SCREEN",
            "M2C_PCPA_AUDIT",
            "M2C_PCPA_CONFIRMATION",
            "M4A_H1_R1",
            "M4A_H4_SEARCH",
            "M4A_H4_RESERVED_AUDIT",
            "M4A_H5_SEARCH",
            "M4A_H5_THIRD_SEED_AUDIT",
            "M4A_H6_ENGINEERING_CANARY",
            "M4A_H6_THIRD_SEED_R3_AUDIT",
            "M4A_H6_R4_SEARCH",
            "M4A_H7_ENGINEERING_CANARY",
            "M4A_H7_R4_ROBUST_SHORTLIST",
            "M4A_H8_ENGINEERING_CANARY",
            "M4A_H8_S_EVAL_CONFIRMATION",
            "M6F_FAST_ENGINEERING_CANARY",
            "M6F_FAST_SEARCH",
            "M7F_FAST_PRELIMINARY",
            "M7F_H1_IMDB_TRAJECTORY",
            "M7F_H1_MANUAL_BASELINE",
            "M7F_H1_BUDGET_CANARY",
            "M7F_H4_DBLP_CANARY",
            "M7F_H4_DBLP_CONFIRMATION",
            "M8_F1_CANARY",
            "M8_F1_FINAL10",
            "M8_F3R_RERUN_2G_CANARY",
            "M8_F3R_RERUN_2G_EVALUATION",
        }:
            raise ValueError("invalid O0 run role")
        if self.run_role == "M1A_SELECTOR_DEVELOPMENT":
            if self.eligible_for_scientific_result:
                raise ValueError("M1A selector-development runs are not scientific results")
            if not self.eligible_for_selector_development:
                raise ValueError(
                    "M1A selector-development runs must declare selector eligibility"
                )
        elif self.run_role == "M1B_SELECTOR_DEVELOPMENT":
            if self.dataset != "freebase":
                raise ValueError("M1B selector-development runs require Freebase")
            if not self.eligible_for_scientific_result:
                raise ValueError("M1B formal runs must be scientific results")
            if not self.eligible_for_selector_development:
                raise ValueError("M1B formal runs must develop the selector")
            if (
                self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != tuple(M1_CHECKPOINT_SCHEDULE)
                or self.capture_mode != "on"
            ):
                raise ValueError("M1B formal run must use exact-3000 and the frozen 8 checkpoints")
        elif self.run_role == "M1B_H2_FIXED_BUDGET_VERIFICATION":
            if self.dataset not in {"acm", "freebase"}:
                raise ValueError("M1B-H2 verification requires ACM or Freebase")
            if not self.eligible_for_scientific_result:
                raise ValueError("M1B-H2 verification runs are scientific development evidence")
            if self.eligible_for_selector_development:
                raise ValueError("M1B-H2 Fixed-End verification cannot develop a selector")
            if not self.physical_run_id or not self.physical_run_id.startswith("M1B-H2-"):
                raise ValueError("M1B-H2 verification requires a canonical physical_run_id")
            if (
                self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != tuple(M1_CHECKPOINT_SCHEDULE)
                or self.capture_mode != "on"
            ):
                raise ValueError(
                    "M1B-H2 verification must use exact-3000 and the frozen 8 checkpoints"
                )
        elif self.run_role == "M1B_H3_CHECKPOINT_CONFIRMATION":
            if self.dataset not in {"acm", "freebase"}:
                raise ValueError("M1B-H3 confirmation requires ACM or Freebase")
            if not self.eligible_for_scientific_result:
                raise ValueError("M1B-H3 confirmation runs are scientific evidence")
            if self.eligible_for_selector_development:
                raise ValueError("M1B-H3 confirmation cannot develop the frozen policy")
            expected_id = re.fullmatch(
                r"M1B-H3-(ACM|FREEBASE)-S\d{2}-(MANUAL|AUTOMATIC)",
                self.physical_run_id or "",
            )
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or expected_id.group(2).casefold() != self.path_set_role.casefold()
            ):
                raise ValueError("M1B-H3 confirmation requires a canonical physical_run_id")
            if (
                self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != tuple(M1_CHECKPOINT_SCHEDULE)
                or self.capture_mode != "on"
            ):
                raise ValueError(
                    "M1B-H3 confirmation must use exact-3000 and the frozen 8 checkpoints"
                )
        elif self.run_role in {"M2A_S1_SCIENTIFIC", "M2B_S2_SCIENTIFIC"}:
            stage = "M2A-S1" if self.run_role == "M2A_S1_SCIENTIFIC" else "M2B-S2"
            expected_id = re.fullmatch(
                rf"{stage}-(ACM|FREEBASE)-S\d{{2}}-(MANUAL|AUTOMATIC)",
                self.physical_run_id or "",
            )
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or expected_id.group(2).casefold() != self.path_set_role.casefold()
            ):
                raise ValueError(f"{stage} requires a canonical physical_run_id")
            if not self.eligible_for_scientific_result:
                raise ValueError(f"{stage} full runs are scientific evidence")
            if self.eligible_for_selector_development:
                raise ValueError(f"{stage} cannot develop a selector")
            if (
                self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != tuple(M1_CHECKPOINT_SCHEDULE)
                or self.capture_mode != "on"
            ):
                raise ValueError(
                    f"{stage} must use exact-3000 and the frozen 8 checkpoints"
                )
        elif self.run_role == "M4A_H1_R1":
            expected_id = re.fullmatch(
                r"M4A-H1-R1-(ACM|FREEBASE)-C(\d{3})-S(00|01)",
                self.physical_run_id or "",
            )
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or self.candidate_set_ordinal is None
                or int(expected_id.group(2)) != self.candidate_set_ordinal
                or self.path_set_role != "Candidate"
            ):
                raise ValueError("M4A-H1 R1 requires a canonical candidate run identity")
            if not self.eligible_for_scientific_result:
                raise ValueError("M4A-H1 R1 runs are scientific evidence")
            if self.eligible_for_selector_development:
                raise ValueError("M4A-H1 R1 cannot develop a checkpoint selector")
            if (
                self.max_updates != 300
                or tuple(self.checkpoint_schedule) != (100, 300)
                or self.capture_mode != "on"
            ):
                raise ValueError(
                    "M4A-H1 R1 must run update 0->300 and save updates 100/300"
                )
            frozen_hash_fields = {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "score_execution_contract_hash": self.score_execution_contract_hash,
                "diagnostic_source_identity_hash": self.diagnostic_source_identity_hash,
                "diagnostic_config_hash": self.diagnostic_config_hash,
                "bank_contract_hash": self.bank_contract_hash,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
                "precision_runtime_contract_hash": self.precision_runtime_contract_hash,
            }
            for field_name, value in frozen_hash_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M4A-H1 R1 requires a SHA-256 {field_name}")
        elif self.run_role in {"M4A_H4_SEARCH", "M4A_H4_RESERVED_AUDIT"}:
            expected_id = re.fullmatch(
                r"M4A-H4-(ACM|FREEBASE)-C(\d{3})-(S00|S01|RA)",
                self.physical_run_id or "",
            )
            expected_token = "RA" if self.run_role == "M4A_H4_RESERVED_AUDIT" else None
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or self.candidate_set_ordinal is None
                or int(expected_id.group(2)) != self.candidate_set_ordinal
                or self.path_set_role != "Candidate"
                or (expected_token is None and expected_id.group(3) not in {"S00", "S01"})
                or (expected_token is not None and expected_id.group(3) != expected_token)
            ):
                raise ValueError("M4A-H4 requires a canonical candidate run identity")
            if not self.eligible_for_scientific_result:
                raise ValueError("M4A-H4 runs are scientific development evidence")
            if not self.eligible_for_selector_development:
                raise ValueError("M4A-H4 runs must declare selector-development eligibility")
            if (
                self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != (100, 300, 1000, 2000, 3000)
                or self.capture_mode != "on"
            ):
                raise ValueError(
                    "M4A-H4 requires max_update=3000 and the frozen 5-boundary schedule"
                )
            frozen_hash_fields = {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "score_execution_contract_hash": self.score_execution_contract_hash,
                "diagnostic_source_identity_hash": self.diagnostic_source_identity_hash,
                "diagnostic_config_hash": self.diagnostic_config_hash,
                "bank_contract_hash": self.bank_contract_hash,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
                "precision_runtime_contract_hash": self.precision_runtime_contract_hash,
            }
            for field_name, value in frozen_hash_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M4A-H4 requires a SHA-256 {field_name}")
        elif self.run_role in {"M4A_H5_SEARCH", "M4A_H5_THIRD_SEED_AUDIT"}:
            expected_id = re.fullmatch(
                r"M4A-H5-(ACM|FREEBASE)-C(\d{3})-S(00|01|02)",
                self.physical_run_id or "",
            )
            expected_token = (
                "02" if self.run_role == "M4A_H5_THIRD_SEED_AUDIT" else None
            )
            expected_seed_by_token = {
                "00": 1985123271,
                "01": 508898167,
                "02": 1950017805,
            }
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or self.candidate_set_ordinal is None
                or int(expected_id.group(2)) != self.candidate_set_ordinal
                or self.path_set_role != "Candidate"
                or (expected_token is None and expected_id.group(3) not in {"00", "01"})
                or (expected_token is not None and expected_id.group(3) != expected_token)
                or expected_seed_by_token.get(expected_id.group(3)) != self.model_seed
            ):
                raise ValueError("M4A-H5 requires a canonical candidate/seed run identity")
            if not self.eligible_for_scientific_result:
                raise ValueError("M4A-H5 runs are scientific development evidence")
            if not self.eligible_for_selector_development:
                raise ValueError("M4A-H5 runs must declare selector-development eligibility")
            if (
                self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != (100, 300, 1000, 2000, 3000)
                or self.capture_mode != "on"
            ):
                raise ValueError(
                    "M4A-H5 requires max_update=3000 and the frozen 5-boundary schedule"
                )
            frozen_hash_fields = {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "score_execution_contract_hash": self.score_execution_contract_hash,
                "diagnostic_source_identity_hash": self.diagnostic_source_identity_hash,
                "diagnostic_config_hash": self.diagnostic_config_hash,
                "bank_contract_hash": self.bank_contract_hash,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
                "precision_runtime_contract_hash": self.precision_runtime_contract_hash,
            }
            for field_name, value in frozen_hash_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M4A-H5 requires a SHA-256 {field_name}")
        elif self.run_role in {
            "M4A_H6_ENGINEERING_CANARY",
            "M4A_H6_THIRD_SEED_R3_AUDIT",
            "M4A_H6_R4_SEARCH",
        }:
            patterns = {
                "M4A_H6_ENGINEERING_CANARY": (
                    r"M4A-H6-CANARY-(ACM|FREEBASE)-C(\d{3})-S(02)",
                    {"02": 1950017805},
                ),
                "M4A_H6_THIRD_SEED_R3_AUDIT": (
                    r"M4A-H6-R3-(ACM|FREEBASE)-C(\d{3})-S(02)",
                    {"02": 1950017805},
                ),
                "M4A_H6_R4_SEARCH": (
                    r"M4A-H6-R4-(ACM|FREEBASE)-C(\d{3})-S(00|01|02)",
                    {"00": 1985123271, "01": 508898167, "02": 1950017805},
                ),
            }
            pattern, expected_seed_by_token = patterns[self.run_role]
            expected_id = re.fullmatch(pattern, self.physical_run_id or "")
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or self.candidate_set_ordinal is None
                or int(expected_id.group(2)) != self.candidate_set_ordinal
                or self.path_set_role != "Candidate"
                or expected_seed_by_token.get(expected_id.group(3)) != self.model_seed
            ):
                raise ValueError("M4A-H6 requires a canonical stage/candidate/seed identity")
            if self.run_role == "M4A_H6_ENGINEERING_CANARY":
                if self.eligible_for_scientific_result or self.eligible_for_selector_development:
                    raise ValueError("M4A-H6 canaries are engineering-only")
                if (
                    self.max_updates != 20
                    or tuple(self.checkpoint_schedule) != (20,)
                    or self.capture_mode != "on"
                ):
                    raise ValueError("M4A-H6 canaries must run exactly 20 captured updates")
            else:
                if not self.eligible_for_scientific_result:
                    raise ValueError("M4A-H6 R3/R4 runs are scientific development evidence")
                if not self.eligible_for_selector_development:
                    raise ValueError("M4A-H6 R3/R4 runs must declare development eligibility")
                if (
                    self.max_updates != 3000
                    or tuple(self.checkpoint_schedule) != (100, 300, 1000, 2000, 3000)
                    or self.capture_mode != "on"
                ):
                    raise ValueError(
                        "M4A-H6 R3/R4 requires max_update=3000 and the frozen schedule"
                    )
            frozen_hash_fields = {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "score_execution_contract_hash": self.score_execution_contract_hash,
                "diagnostic_source_identity_hash": self.diagnostic_source_identity_hash,
                "diagnostic_config_hash": self.diagnostic_config_hash,
                "bank_contract_hash": self.bank_contract_hash,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
                "precision_runtime_contract_hash": self.precision_runtime_contract_hash,
            }
            for field_name, value in frozen_hash_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M4A-H6 requires a SHA-256 {field_name}")
        elif self.run_role in {
            "M4A_H7_ENGINEERING_CANARY",
            "M4A_H7_R4_ROBUST_SHORTLIST",
        }:
            patterns = {
                "M4A_H7_ENGINEERING_CANARY": (
                    r"M4A-H7-CANARY-(ACM|FREEBASE)-C(\d{3})-S(02)",
                    {"02": 1950017805},
                ),
                "M4A_H7_R4_ROBUST_SHORTLIST": (
                    r"M4A-H7-R4-(ACM|FREEBASE)-C(\d{3})-S(00|01|02)",
                    {"00": 1985123271, "01": 508898167, "02": 1950017805},
                ),
            }
            pattern, expected_seed_by_token = patterns[self.run_role]
            expected_id = re.fullmatch(pattern, self.physical_run_id or "")
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or self.candidate_set_ordinal is None
                or int(expected_id.group(2)) != self.candidate_set_ordinal
                or self.path_set_role != "Candidate"
                or expected_seed_by_token.get(expected_id.group(3)) != self.model_seed
            ):
                raise ValueError("M4A-H7 requires a canonical stage/candidate/seed identity")
            if self.run_role == "M4A_H7_ENGINEERING_CANARY":
                if self.eligible_for_scientific_result or self.eligible_for_selector_development:
                    raise ValueError("M4A-H7 canaries are engineering-only")
                if (
                    self.max_updates != 2020
                    or tuple(self.checkpoint_schedule) != (100, 300, 1000, 2000, 2020)
                    or self.capture_mode != "on"
                ):
                    raise ValueError(
                        "M4A-H7 canaries must strictly continue 2000->2020 with capture"
                    )
            else:
                if not self.eligible_for_scientific_result:
                    raise ValueError("M4A-H7 R4 runs are scientific development evidence")
                if not self.eligible_for_selector_development:
                    raise ValueError("M4A-H7 R4 runs must declare development eligibility")
                if (
                    self.max_updates != 3000
                    or tuple(self.checkpoint_schedule) != (100, 300, 1000, 2000, 3000)
                    or self.capture_mode != "on"
                ):
                    raise ValueError(
                        "M4A-H7 R4 requires max_update=3000 and the frozen schedule"
                    )
            frozen_hash_fields = {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "score_execution_contract_hash": self.score_execution_contract_hash,
                "diagnostic_source_identity_hash": self.diagnostic_source_identity_hash,
                "diagnostic_config_hash": self.diagnostic_config_hash,
                "bank_contract_hash": self.bank_contract_hash,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
                "precision_runtime_contract_hash": self.precision_runtime_contract_hash,
            }
            for field_name, value in frozen_hash_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M4A-H7 requires a SHA-256 {field_name}")
        elif self.run_role in {
            "M4A_H8_ENGINEERING_CANARY",
            "M4A_H8_S_EVAL_CONFIRMATION",
        }:
            future_seeds = (
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
            expected_seed_by_token = {
                f"{index:02d}": seed for index, seed in enumerate(future_seeds)
            }
            if self.run_role == "M4A_H8_ENGINEERING_CANARY":
                pattern = r"M4A-H8-CANARY-(ACM|FREEBASE)-C(\d{3})-F(00)"
            else:
                pattern = r"M4A-H8-S-EVAL-(ACM|FREEBASE)-C(\d{3})-F(0[0-9])"
            expected_id = re.fullmatch(pattern, self.physical_run_id or "")
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or self.candidate_set_ordinal is None
                or int(expected_id.group(2)) != self.candidate_set_ordinal
                or self.path_set_role != "Candidate"
                or expected_seed_by_token.get(expected_id.group(3)) != self.model_seed
            ):
                raise ValueError("M4A-H8 requires a canonical stage/candidate/future-seed identity")
            canary = self.run_role == "M4A_H8_ENGINEERING_CANARY"
            if canary:
                if self.eligible_for_scientific_result or self.eligible_for_selector_development:
                    raise ValueError("M4A-H8 canaries are engineering-only")
                if (
                    self.max_updates != 20
                    or tuple(self.checkpoint_schedule) != (20,)
                    or self.capture_mode != "on"
                ):
                    raise ValueError("M4A-H8 canaries must run exactly 20 captured updates")
            else:
                if not self.eligible_for_scientific_result:
                    raise ValueError("M4A-H8 S_eval runs are scientific confirmation evidence")
                if self.eligible_for_selector_development:
                    raise ValueError("M4A-H8 cannot further develop or retune the selector")
                if (
                    self.max_updates != 3000
                    or tuple(self.checkpoint_schedule) != (1000, 2000, 3000)
                    or self.capture_mode != "on"
                ):
                    raise ValueError(
                        "M4A-H8 S_eval requires exact-3000 and rolling 1000/2000/3000 captures"
                    )
            frozen_hash_fields = {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "score_execution_contract_hash": self.score_execution_contract_hash,
                "diagnostic_source_identity_hash": self.diagnostic_source_identity_hash,
                "diagnostic_config_hash": self.diagnostic_config_hash,
                "bank_contract_hash": self.bank_contract_hash,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
                "precision_runtime_contract_hash": self.precision_runtime_contract_hash,
            }
            for field_name, value in frozen_hash_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M4A-H8 requires a SHA-256 {field_name}")
        elif self.run_role in {
            "M6F_FAST_ENGINEERING_CANARY",
            "M6F_FAST_SEARCH",
            "M7F_FAST_PRELIMINARY",
        }:
            from metachorus_v53.m6f_m7f_fast.constants import (
                EVAL_MODEL_SEEDS,
                SEARCH_MODEL_SEEDS,
            )

            dataset_token = {"dblp": "DBLP", "imdb": "IMDB", "aminer": "AMINER"}.get(self.dataset)
            if dataset_token is None or self.candidate_set_ordinal is None or self.path_set_role != "Candidate":
                raise ValueError("M6F/M7F fast runs require a frozen dataset/candidate identity")
            role_contract = {
                "M6F_FAST_ENGINEERING_CANARY": (
                    rf"M6F-FAST-CANARY-{dataset_token}-C(\d{{3}})-Q(0[0-2])",
                    SEARCH_MODEL_SEEDS,
                    20,
                    (20,),
                    False,
                    False,
                ),
                "M6F_FAST_SEARCH": (
                    rf"M6F-FAST-{dataset_token}-C(\d{{3}})-Q(0[0-2])",
                    SEARCH_MODEL_SEEDS,
                    3000,
                    (300, 1000, 2000, 3000),
                    True,
                    True,
                ),
                "M7F_FAST_PRELIMINARY": (
                    rf"M7F-FAST-{dataset_token}-C(\d{{3}})-E(0[0-2])",
                    EVAL_MODEL_SEEDS,
                    3000,
                    (1000, 2000, 3000),
                    True,
                    False,
                ),
            }[self.run_role]
            pattern, seeds, expected_updates, expected_schedule, science, selector = role_contract
            match = re.fullmatch(pattern, self.physical_run_id or "")
            expected_seed_by_token = {f"{index:02d}": seed for index, seed in enumerate(seeds)}
            if (
                match is None
                or int(match.group(1)) != self.candidate_set_ordinal
                or expected_seed_by_token.get(match.group(2)) != self.model_seed
                or self.max_updates != expected_updates
                or tuple(self.checkpoint_schedule) != expected_schedule
                or self.capture_mode != "on"
                or self.eligible_for_scientific_result is not science
                or self.eligible_for_selector_development is not selector
            ):
                raise ValueError("M6F/M7F fast run identity, seed, schedule, or eligibility mismatch")
            frozen_hash_fields = {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
            }
            for field_name, value in frozen_hash_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M6F/M7F fast run requires a SHA-256 {field_name}")
        elif self.run_role in {
            "M7F_H1_IMDB_TRAJECTORY",
            "M7F_H1_MANUAL_BASELINE",
            "M7F_H1_BUDGET_CANARY",
        }:
            from metachorus_v53.m7f_h1.constants import (
                BUDGET_CANARY_CHECKPOINT_SCHEDULE,
                BUDGET_CANARY_REPRESENTATION_SCHEDULE,
                BUDGET_CANARY_UPDATES,
                EVALUATION_MODEL_SEEDS,
                MANUAL_CHECKPOINT_SCHEDULE,
                MANUAL_REPRESENTATION_SCHEDULE,
                SEARCH_MODEL_SEEDS,
                TRAJECTORY_CHECKPOINT_SCHEDULE,
                TRAJECTORY_REPRESENTATION_SCHEDULE,
                TRAJECTORY_UPDATES,
            )

            dataset_token = {
                "dblp": "DBLP",
                "imdb": "IMDB",
                "aminer": "AMINER",
            }.get(self.dataset)
            if dataset_token is None or self.candidate_set_ordinal is None:
                raise ValueError("M7F-H1 requires a frozen dataset/candidate identity")
            if self.run_role == "M7F_H1_IMDB_TRAJECTORY":
                pattern = r"M7F-H1-IMDB-TRAJECTORY-C(\d{3})-E(0[0-2])"
                seeds = EVALUATION_MODEL_SEEDS
                expected_dataset = "imdb"
                expected_updates = TRAJECTORY_UPDATES
                expected_checkpoints = TRAJECTORY_CHECKPOINT_SCHEDULE
                expected_representations = TRAJECTORY_REPRESENTATION_SCHEDULE
                valid_roles = {"FINAL_FROZEN_METHOD", "MANUAL"}
                science, selector = True, True
            elif self.run_role == "M7F_H1_MANUAL_BASELINE":
                pattern = rf"M7F-H1-MANUAL-{dataset_token}-C(\d{{3}})-E(0[0-2])"
                seeds = EVALUATION_MODEL_SEEDS
                expected_dataset = self.dataset
                expected_updates = TRAJECTORY_UPDATES
                expected_checkpoints = MANUAL_CHECKPOINT_SCHEDULE
                expected_representations = MANUAL_REPRESENTATION_SCHEDULE
                valid_roles = {"MANUAL"}
                science, selector = True, False
            else:
                pattern = rf"M7F-H1-BUDGET-{dataset_token}-(MANUAL|SHORTEST|LONGEST)-C(\d{{3}})-Q(0[0-1])"
                seeds = SEARCH_MODEL_SEEDS[:2]
                expected_dataset = self.dataset
                expected_updates = BUDGET_CANARY_UPDATES
                expected_checkpoints = BUDGET_CANARY_CHECKPOINT_SCHEDULE
                expected_representations = BUDGET_CANARY_REPRESENTATION_SCHEDULE
                valid_roles = {"MANUAL", "BUDGET_SHORTEST", "BUDGET_LONGEST"}
                science, selector = False, False
            match = re.fullmatch(pattern, self.physical_run_id or "")
            seed_token = match.group(match.lastindex or 0) if match is not None else None
            expected_seed_by_token = {
                f"{index:02d}": seed for index, seed in enumerate(seeds)
            }
            if (
                match is None
                or expected_dataset != self.dataset
                or int(match.group(match.lastindex - 1 if self.run_role == "M7F_H1_BUDGET_CANARY" else 1))
                != self.candidate_set_ordinal
                or expected_seed_by_token.get(seed_token) != self.model_seed
                or self.path_set_role not in valid_roles
                or self.max_updates != expected_updates
                or tuple(self.checkpoint_schedule) != tuple(expected_checkpoints)
                or self.resolved_representation_schedule()
                != tuple(expected_representations)
                or self.capture_mode != "on"
                or self.eligible_for_scientific_result is not science
                or self.eligible_for_selector_development is not selector
            ):
                raise ValueError(
                    "M7F-H1 run identity, seed, schedules, role, or eligibility mismatch"
                )
            for field_name, value in {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
            }.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M7F-H1 requires a SHA-256 {field_name}")
        elif self.run_role in {
            "M7F_H4_DBLP_CANARY",
            "M7F_H4_DBLP_CONFIRMATION",
        }:
            from metachorus_v53.m7f_h4.constants import CONFIRMATION_MODEL_SEEDS

            canary = self.run_role == "M7F_H4_DBLP_CANARY"
            pattern = (
                r"M7F-H4-CANARY-DBLP-(SELECTED|MANUAL)-C(\d{3})-F(0[01])"
                if canary
                else r"M7F-H4-CONFIRM-DBLP-(SELECTED|MANUAL)-C(\d{3})-F(0[0-9])"
            )
            match = re.fullmatch(pattern, self.physical_run_id or "")
            expected_seed_by_token = {
                f"{index:02d}": seed for index, seed in enumerate(CONFIRMATION_MODEL_SEEDS)
            }
            expected_ordinal = {"SELECTED": 57, "MANUAL": 10}
            expected_role = {"SELECTED": "DBLP_SELECTED", "MANUAL": "DBLP_MANUAL"}
            arm = match.group(1) if match is not None else ""
            if (
                match is None
                or self.dataset != "dblp"
                or self.candidate_set_ordinal is None
                or int(match.group(2)) != self.candidate_set_ordinal
                or expected_ordinal.get(arm) != self.candidate_set_ordinal
                or self.path_set_role != expected_role.get(arm)
                or expected_seed_by_token.get(match.group(3)) != self.model_seed
                or self.capture_mode != "on"
            ):
                raise ValueError("M7F-H4 requires a canonical DBLP arm/candidate/seed identity")
            if canary:
                if (
                    self.eligible_for_scientific_result
                    or self.eligible_for_selector_development
                    or self.max_updates != 20
                    or tuple(self.checkpoint_schedule) != (20,)
                ):
                    raise ValueError("M7F-H4 canaries are exact 20-update engineering-only runs")
            elif (
                not self.eligible_for_scientific_result
                or self.eligible_for_selector_development
                or self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != (1000, 2000, 3000)
            ):
                raise ValueError("M7F-H4 confirmation requires fresh exact-3000 scientific runs")
            for field_name, value in {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
            }.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M7F-H4 requires a SHA-256 {field_name}")
        elif self.run_role in {"M8_F1_CANARY", "M8_F1_FINAL10"}:
            from metachorus_v53.m8_f1.constants import (
                CONFIRMATION_MODEL_SEEDS,
                DATASET_SPECS,
            )

            canary = self.run_role == "M8_F1_CANARY"
            phase = "CANARY" if canary else "FINAL10"
            seed_pattern = r"0[01]" if canary else r"0[0-9]"
            match = re.fullmatch(
                rf"M8-F1-{phase}-(IMDB|AMINER)-MANUAL-C000-F({seed_pattern})",
                self.physical_run_id or "",
            )
            expected_seed_by_token = {
                f"{index:02d}": seed for index, seed in enumerate(CONFIRMATION_MODEL_SEEDS)
            }
            dataset = match.group(1).casefold() if match is not None else ""
            if (
                match is None
                or dataset not in DATASET_SPECS
                or self.dataset != dataset
                or self.candidate_set_ordinal != 0
                or self.path_set_role != "MANUAL"
                or self.candidate_set_hash != DATASET_SPECS[dataset]["manual_set_hash"]
                or tuple(self.path_ids) != tuple(DATASET_SPECS[dataset]["manual_path_ids"])
                or expected_seed_by_token.get(match.group(2)) != self.model_seed
                or self.capture_mode != "on"
            ):
                raise ValueError("M8-F1 requires an exact IMDb/AMiner Manual identity")
            if canary:
                if (
                    self.eligible_for_scientific_result
                    or self.eligible_for_selector_development
                    or self.max_updates != 20
                    or tuple(self.checkpoint_schedule) != (20,)
                    or tuple(self.resolved_representation_schedule()) != (20,)
                ):
                    raise ValueError("M8-F1 canaries are exact 20-update engineering-only runs")
            elif (
                not self.eligible_for_scientific_result
                or self.eligible_for_selector_development
                or self.max_updates != 3000
                or tuple(self.checkpoint_schedule) != (1000, 2000, 3000)
                or tuple(self.resolved_representation_schedule()) != (3000,)
            ):
                raise ValueError("M8-F1 FINAL10 requires fresh exact-3000 scientific runs")
            for field_name, value in {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
            }.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"M8-F1 requires a SHA-256 {field_name}")
        elif self.run_role in {
            "M8_F3R_RERUN_2G_CANARY",
            "M8_F3R_RERUN_2G_EVALUATION",
        }:
            from metachorus_v53.m8_f3r_rerun_2g.constants import (
                CANARY_ENDPOINT_UPDATE,
                ENDPOINT_UPDATE,
                EVALUATION_MODEL_SEEDS,
            )

            canary = self.run_role == "M8_F3R_RERUN_2G_CANARY"
            if canary:
                match = re.fullmatch(
                    r"M8-F3R-2G-CANARY-(FREEBASE|DBLP)-([0-9a-f]{12})-C0[0-3]",
                    self.physical_run_id or "",
                )
            else:
                match = re.fullmatch(
                    r"M8-F3R-2G-(FREEBASE|DBLP)-([0-9a-f]{12})-E(0[0-2])",
                    self.physical_run_id or "",
                )
            dataset = match.group(1).casefold() if match is not None else ""
            if (
                match is None
                or self.dataset != dataset
                or match.group(2) != self.candidate_set_hash[:12]
                or self.capture_mode != "on"
                or self.eligible_for_selector_development
                or self.path_set_role not in {
                    "MANUAL",
                    "FULL_FROZEN",
                    "ABLATION_UNIQUE_SET",
                }
            ):
                raise ValueError("M8-F3R-RERUN-2G run identity is invalid")
            if canary:
                if (
                    self.model_seed != EVALUATION_MODEL_SEEDS[0]
                    or self.eligible_for_scientific_result
                    or self.max_updates != CANARY_ENDPOINT_UPDATE
                    or tuple(self.checkpoint_schedule) != (CANARY_ENDPOINT_UPDATE,)
                    or tuple(self.resolved_representation_schedule())
                    != (CANARY_ENDPOINT_UPDATE,)
                ):
                    raise ValueError(
                        "M8-F3R-RERUN-2G canaries are exact-20 engineering runs"
                    )
            else:
                seed_token = match.group(3)
                expected_seed = EVALUATION_MODEL_SEEDS[int(seed_token)]
                if (
                    self.model_seed != expected_seed
                    or not self.eligible_for_scientific_result
                    or self.max_updates != ENDPOINT_UPDATE
                    or tuple(self.checkpoint_schedule) != (1000, 2000, ENDPOINT_UPDATE)
                    or tuple(self.resolved_representation_schedule())
                    != (ENDPOINT_UPDATE,)
                ):
                    raise ValueError(
                        "M8-F3R-RERUN-2G evaluation requires fresh exact-3000 science"
                    )
            for field_name, value in {
                "source_data_identity": self.source_data_identity,
                "projection_identity": self.projection_identity,
                "candidate_bank_hash": self.candidate_bank_hash,
                "plan_seal_hash": self.plan_seal_hash,
            }.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(
                        f"M8-F3R-RERUN-2G requires a SHA-256 {field_name}"
                    )
        elif self.run_role in {
            "M2C_PCPA_SCREEN",
            "M2C_PCPA_AUDIT",
            "M2C_PCPA_CONFIRMATION",
        }:
            phase = {
                "M2C_PCPA_SCREEN": "SCREEN",
                "M2C_PCPA_AUDIT": "AUDIT",
                "M2C_PCPA_CONFIRMATION": "CONFIRM",
            }[self.run_role]
            expected_id = re.fullmatch(
                rf"M2C-PCPA-{phase}-(ACM|FREEBASE)-S\d{{2}}-"
                rf"(MANUAL|AUTOMATIC)-(S0|P1|P2|P3|P4)",
                self.physical_run_id or "",
            )
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or expected_id.group(2).casefold() != self.path_set_role.casefold()
            ):
                raise ValueError("M2C PCPA requires a canonical physical_run_id")
            if not self.eligible_for_scientific_result:
                raise ValueError("M2C PCPA runs are scientific evidence")
            if self.eligible_for_selector_development:
                raise ValueError("M2C PCPA cannot develop the meta-path selector")
            expected_updates = 1500 if phase == "SCREEN" else 3000
            expected_schedule = (
                (300, 500, 1000, 1500)
                if phase == "SCREEN"
                else tuple(M1_CHECKPOINT_SCHEDULE)
            )
            if (
                self.max_updates != expected_updates
                or tuple(self.checkpoint_schedule) != expected_schedule
                or self.capture_mode != "on"
            ):
                raise ValueError(
                    "M2C PCPA run length/checkpoints must match the frozen phase contract"
                )
        elif self.run_role == "M2A_S1_SMOKE":
            expected_id = re.fullmatch(
                r"M2A-S1-SMOKE-(ACM|FREEBASE)-MANUAL",
                self.physical_run_id or "",
            )
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or self.path_set_role.casefold() != "manual"
            ):
                raise ValueError("M2A-S1 smoke requires a canonical Manual run ID")
            if self.eligible_for_scientific_result or self.eligible_for_selector_development:
                raise ValueError("M2A-S1 smoke is ineligible for scientific reporting")
            if (
                self.max_updates != 20
                or tuple(self.checkpoint_schedule) != (20,)
                or self.capture_mode != "on"
            ):
                raise ValueError("M2A-S1 smoke must run exactly 20 updates")
        elif self.run_role == "M2B_S2_EQUIVALENCE":
            expected_id = re.fullmatch(
                r"M2B-S2-EQUIV-(ACM|FREEBASE)-(MANUAL|AUTOMATIC)-(S1|S2)",
                self.physical_run_id or "",
            )
            if (
                expected_id is None
                or expected_id.group(1).casefold() != self.dataset
                or expected_id.group(2).casefold() != self.path_set_role.casefold()
            ):
                raise ValueError("M2B-S2 equivalence requires a canonical run ID")
            if self.eligible_for_scientific_result or self.eligible_for_selector_development:
                raise ValueError("M2B-S2 equivalence is engineering-only")
            if (
                self.max_updates != 20
                or tuple(self.checkpoint_schedule) != (20,)
                or self.capture_mode != "on"
            ):
                raise ValueError("M2B-S2 equivalence must run exactly 20 updates")
        elif self.run_role == "M1B0_ENGINEERING_SMOKE":
            if self.dataset != "freebase":
                raise ValueError("M1B-0 smoke requires Freebase")
            if self.eligible_for_scientific_result or self.eligible_for_selector_development:
                raise ValueError("M1B-0 smoke is ineligible for science or selector development")
            if self.max_updates != 20 or self.capture_mode != "on":
                raise ValueError("M1B-0 smoke must run exactly 20 updates with capture")
        elif self.eligible_for_selector_development:
            raise ValueError("M0-C engineering runs cannot develop the selector")
        elif self.eligible_for_scientific_result:
            raise ValueError("engineering O0 runs cannot be scientific results")
        if self.dataset == "acm" and self.cache_identity != SANITIZED_CACHE_IDENTITY:
            raise ValueError("sanitized cache identity mismatch")
        if self.dataset == "freebase" and (
            len(self.cache_identity) != 64
            or any(character not in "0123456789abcdef" for character in self.cache_identity)
        ):
            raise ValueError("Freebase sanitized cache identity must be a SHA-256")
        if self.dataset in {"dblp", "imdb", "aminer"} and (
            len(self.cache_identity) != 64
            or any(character not in "0123456789abcdef" for character in self.cache_identity)
        ):
            raise ValueError("fast sanitized cache identity must be a SHA-256")
        if self.device != "cuda:0":
            raise ValueError(
                "frozen model.py constructs adjacency on cuda:0; exactly "
                "one visible GPU must be exposed as cuda:0"
            )
        expected_views = {
            "acm": 2,
            "freebase": 3,
            "dblp": 3,
            "imdb": 2,
            "aminer": 2,
        }[self.dataset]
        if len(self.path_ids) != expected_views:
            raise ValueError(
                f"{self.dataset} path set must contain exactly {expected_views} views"
            )
        if self.capture_mode not in {"off", "on"}:
            raise ValueError("capture_mode must be off or on")
        representation_schedule = self.resolved_representation_schedule()
        if self.capture_mode == "off" and (
            self.checkpoint_schedule or representation_schedule
        ):
            raise ValueError("capture-off run cannot have checkpoints or representations")
        for name, schedule in (
            ("checkpoint", tuple(self.checkpoint_schedule)),
            ("representation", representation_schedule),
        ):
            if tuple(schedule) != tuple(sorted(schedule)) or len(schedule) != len(set(schedule)):
                raise ValueError(f"{name} schedule must be strictly increasing")
            if any(update <= 0 or update > self.max_updates for update in schedule):
                raise ValueError(f"{name} schedule exceeds update budget")
        observed_defaults = {key: getattr(self, key) for key in O0_DEFAULTS}
        expected_defaults = {**O0_DEFAULTS, "dataset": self.dataset}
        if observed_defaults != expected_defaults:
            raise ValueError(
                f"O0 defaults changed: {observed_defaults!r}"
            )

    def identity(self) -> str:
        return canonical_json_hash(asdict(self))


def load_run_config(path: str | Path) -> O0RunConfig:
    config = O0RunConfig.from_mapping(read_json(path))
    config.validate()
    return config


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_frozen_seed_function(path: Path) -> Any:
    """Load only the frozen seed helper, not unrelated utility imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "set_random_seed"
    ]
    if len(matches) != 1 or isinstance(matches[0], ast.AsyncFunctionDef):
        raise RuntimeError("frozen utils must define one set_random_seed")
    function = matches[0]
    if function.decorator_list:
        raise RuntimeError("frozen set_random_seed cannot be decorated")
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "random": random,
        "np": np,
        "torch": torch,
    }
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["set_random_seed"]


class BranchLossTracer(AbstractContextManager["BranchLossTracer"]):
    """Observe local branch tensors at return without recomputation or RNG."""

    def __init__(
        self,
        contrast_code: Any,
        *,
        retain_autograd_graph: bool = False,
    ):
        self.contrast_code = contrast_code
        self.retain_autograd_graph = bool(retain_autograd_graph)
        self.rows: list[dict[str, Any]] = []
        self.payload_incomplete = False
        self.incomplete_events: list[dict[str, Any]] = []
        self.original_exception: dict[str, Any] | None = None
        self._previous = None

    def _trace(self, frame: Any, event: str, argument: Any) -> Any:
        if frame.f_code is not self.contrast_code:
            return None
        if event == "exception":
            exception_type, exception, _ = argument
            self.original_exception = {
                "type": exception_type.__name__,
                "message": str(exception),
                "error_type": exception_type.__name__,
                "error": str(exception),
            }
            return self._trace
        if event == "return":
            local = frame.f_locals
            required = (
                "recon_structure_loss",
                "l_cn",
                "l_cv",
                "l_gl",
                "log_weights",
            )
            missing = [name for name in required if name not in local]
            if missing:
                
                
                
                self.payload_incomplete = True
                self.incomplete_events.append(
                    {
                        "event": event,
                        "missing_locals": missing,
                        "return_argument_is_none": argument is None,
                    }
                )
                return self._trace
            values = {
                "recon_structure_loss": local["recon_structure_loss"],
                "l_cn": local["l_cn"],
                "l_cv": local["l_cv"],
                "l_gl": local["l_gl"],
                "weights": local["log_weights"],
            }
            self.rows.append(
                values
                if self.retain_autograd_graph
                else {
                    name: value.detach().cpu()
                    for name, value in values.items()
                }
            )
        return self._trace

    def __enter__(self) -> "BranchLossTracer":
        self._previous = sys.gettrace()
        if self._previous is not None:
            raise RuntimeError(
                "O0 branch tracing requires no pre-existing sys.settrace hook"
            )
        sys.settrace(self._trace)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        exception_traceback: Any,
    ) -> None:
        sys.settrace(self._previous)
        if exception_type is not None and exception is not None:
            self.original_exception = {
                "type": exception_type.__name__,
                "message": str(exception),
                "error_type": exception_type.__name__,
                "error": str(exception),
                "traceback": "".join(
                    traceback.format_exception(
                        exception_type,
                        exception,
                        exception_traceback,
                    )
                ),
            }

    def diagnostic_metadata(self) -> dict[str, Any]:
        return {
            "tracer_payload_incomplete": self.payload_incomplete,
            "incomplete_events": list(self.incomplete_events),
            "original_exception": self.original_exception,
            "captured_contrast_count": len(self.rows),
            "retains_autograd_graph": self.retain_autograd_graph,
        }

    def summary(self, expected_contrasts: int) -> dict[str, Any]:
        if len(self.rows) != expected_contrasts:
            raise RuntimeError(
                f"observed {len(self.rows)} contrasts, expected "
                f"{expected_contrasts}"
            )
        names = ("recon_structure_loss", "l_cn", "l_cv", "l_gl")
        values = {
            name: float(
                torch.stack([row[name] for row in self.rows])
                .mean()
                .detach()
                .cpu()
            )
            for name in names
        }
        weights = (
            torch.stack([row["weights"] for row in self.rows])
            .mean(dim=0)
            .detach()
            .cpu()
            .tolist()
        )
        return values | {"weights": [float(value) for value in weights]}

    def live_branch_means(self) -> tuple[torch.Tensor, ...]:
        """Return the four live mean branch losses for gradient surgery."""

        if not self.retain_autograd_graph:
            raise RuntimeError("branch tracer did not retain the autograd graph")
        if not self.rows:
            raise RuntimeError("branch tracer captured no contrast calls")
        names = ("recon_structure_loss", "l_cn", "l_cv", "l_gl")
        return tuple(
            torch.stack([row[name] for row in self.rows]).mean()
            for name in names
        )

    def live_mean_weights(self) -> torch.Tensor:
        """Return live softmax weights averaged over contrast calls."""

        if not self.retain_autograd_graph:
            raise RuntimeError("branch tracer did not retain the autograd graph")
        if not self.rows:
            raise RuntimeError("branch tracer captured no contrast calls")
        return torch.stack([row["weights"] for row in self.rows]).mean(dim=0)


def _gradient_norm_and_vector(
    model: torch.nn.Module,
    *,
    keep_vector: bool,
) -> tuple[float, torch.Tensor | None]:
    tensors = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not tensors:
        raise RuntimeError("model produced no gradients")
    squared = torch.stack(
        [torch.sum(tensor.float() ** 2) for tensor in tensors]
    ).sum()
    norm = float(torch.sqrt(squared).detach().cpu())
    vector = (
        torch.cat(
            [tensor.reshape(-1).detach().cpu().float() for tensor in tensors]
        )
        if keep_vector
        else None
    )
    return norm, vector


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()
    numeric = value.double()
    numel = int(value.numel())
    l2 = float(torch.linalg.vector_norm(numeric))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": digest,
        "numel": numel,
        "sum": float(numeric.sum()),
        "l2": l2,
        "mean": float(numeric.mean()) if numel else 0.0,
        "std": (
            float(torch.std(numeric, unbiased=False)) if numel else 0.0
        ),
        "rms": l2 / math.sqrt(numel) if numel else 0.0,
        "max_abs": (
            float(torch.max(torch.abs(numeric))) if numel else 0.0
        ),
        "finite": bool(torch.isfinite(numeric).all()),
    }


def _cpu_clone_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: _cpu_clone_nested(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_cpu_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_nested(item) for item in value)
    return value


def _uses_full_candidate_bank(*, dataset: str, run_role: str) -> bool:
    if run_role in {
        "M4A_H1_R1",
        "M4A_H4_SEARCH",
        "M4A_H4_RESERVED_AUDIT",
        "M4A_H7_ENGINEERING_CANARY",
        "M4A_H7_R4_ROBUST_SHORTLIST",
        "M4A_H8_ENGINEERING_CANARY",
        "M4A_H8_S_EVAL_CONFIRMATION",
        "M6F_FAST_ENGINEERING_CANARY",
        "M6F_FAST_SEARCH",
        "M7F_FAST_PRELIMINARY",
        "M7F_H1_IMDB_TRAJECTORY",
        "M7F_H1_MANUAL_BASELINE",
        "M7F_H1_BUDGET_CANARY",
        "M7F_H4_DBLP_CANARY",
        "M7F_H4_DBLP_CONFIRMATION",
        "M8_F1_CANARY",
        "M8_F1_FINAL10",
        "M8_F3R_RERUN_2G_CANARY",
        "M8_F3R_RERUN_2G_EVALUATION",
    }:
        return True
    return dataset == "freebase" and run_role in {
        "M4A_H5_SEARCH",
        "M4A_H5_THIRD_SEED_AUDIT",
        "M4A_H6_ENGINEERING_CANARY",
        "M4A_H6_THIRD_SEED_R3_AUDIT",
        "M4A_H6_R4_SEARCH",
    }


class O0TrainingSession:
    
    TRACE_DURABLE_FLUSH_INTERVAL = 1

    def __init__(
        self,
        config: O0RunConfig,
        *,
        repo_root: str | Path,
        enforce_target_gpu: bool = True,
    ):
        config.validate()
        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self.output_root = Path(config.output_root).resolve()
        self.cache_root = Path(config.cache_root).resolve()
        if enforce_target_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("O0 GPU run requires one visible RTX 4090")
            if torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "O0 GPU run requires exactly one visible CUDA device"
                )
            if "4090" not in torch.cuda.get_device_name(0):
                raise RuntimeError(
                    f"O0 GPU run requires RTX 4090, observed "
                    f"{torch.cuda.get_device_name(0)}"
                )
        self.device = torch.device(config.device)
        complete = read_json(
            self.cache_root / "SANITIZED_CACHE_COMPLETE.json"
        )
        if complete.get("cache_identity") != config.cache_identity:
            raise ValueError("cache complete marker identity mismatch")
        if config.dataset == "acm":
            self.adapter = ACMSanitizedAdapter(self.cache_root)
        elif config.dataset == "freebase":
            self.adapter = FreebaseSanitizedAdapter(self.cache_root)
        else:
            self.adapter = FastSanitizedAdapter(
                self.cache_root, expected_dataset=config.dataset
            )
        bank = self.adapter.candidate_bank()
        full_candidate_role = _uses_full_candidate_bank(
            dataset=config.dataset,
            run_role=config.run_role,
        )
        candidate_rows = (
            bank["all_sets"]["sets"]
            if full_candidate_role
            and config.dataset in {"freebase", "dblp", "imdb", "aminer"}
            else bank["fixed_cardinality_sets"]["sets"]
        )
        if full_candidate_role:
            required_candidate_fields = {"path_ids"}
            for row in candidate_rows:
                if (
                    not required_candidate_fields.issubset(row)
                    or not isinstance(
                        row.get("set_id", row.get("path_set_id", row.get("candidate_set_id"))),
                        str,
                    )
                    or not isinstance(
                        row.get("candidate_set_hash", row.get("set_sha256")), str
                    )
                    or int(
                        row.get(
                            "set_ordinal",
                            row.get("ordinal", row.get("candidate_set_ordinal", -1)),
                        )
                    )
                    < 0
                ):
                    raise ValueError("M4A-H1 candidate bank row schema is incomplete")
        matching_sets = [
            row
            for row in candidate_rows
            if row.get("set_id", row.get("path_set_id", row.get("candidate_set_id")))
            == config.path_set_id
            and row.get("candidate_set_hash", row.get("set_sha256"))
            == config.candidate_set_hash
            and tuple(row["path_ids"]) == config.path_ids
            and (
                not full_candidate_role
                or int(
                    row.get(
                        "set_ordinal",
                        row.get("ordinal", row.get("candidate_set_ordinal", -1)),
                    )
                )
                == config.candidate_set_ordinal
            )
        ]
        if len(matching_sets) != 1:
            raise ValueError("runner path set did not resolve from candidate bank")
        self.features = self.adapter.features().to(self.device)
        views = self.adapter.projected_views(config.path_ids)
        self.edge_indices = [
            views[path_id].to(self.device) for path_id in config.path_ids
        ]
        self._model_module = _load_module(
            "_metachorus_frozen_o0_model",
            self.repo_root / "model.py",
        )
        self._set_random_seed = _load_frozen_seed_function(
            self.repo_root / "utils.py"
        )
        self._set_random_seed(config.model_seed)
        encoder = self._model_module.Encoder(
            in_dim=int(self.features.shape[1]),
            hid_dim=config.hid_dim,
            num_layers=config.num_layers,
        )
        self.model = self._model_module.TripletRepresentationLearner(
            encoder=encoder,
            hid_dim=config.hid_dim,
            num_relations=len(config.path_ids),
            tau=config.tau,
            pe=config.edge_drop_rate,
            pf=config.feature_drop_rate,
            alpha=config.alpha,
            moving_average_decay=config.moving_average_decay,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.current_update = 0
        self._last_branch_tracer: BranchLossTracer | None = None
        self.contrast_count = (
            len(config.path_ids) * (len(config.path_ids) + 1) // 2
        )
        self._contrast_code = self.model.contrast.__func__.__code__

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "config_hash": self.config.identity(),
            "source_commit": self.config.source_commit,
            "source_tree": self.config.source_tree,
            "model_py_sha256": sha256_file(self.repo_root / "model.py"),
            "data_cache_hash": self.config.cache_identity,
            "path_set_hash": self.config.candidate_set_hash,
            "model_seed": self.config.model_seed,
            "augmentation_namespace": self.config.augmentation_namespace,
            "trainer_id": self.config.trainer_id,
            "dataset": self.config.dataset,
            "physical_run_id": self.config.physical_run_id,
            "candidate_set_ordinal": self.config.candidate_set_ordinal,
            "source_data_identity": self.config.source_data_identity,
            "projection_identity": self.config.projection_identity,
            "score_execution_contract_hash": self.config.score_execution_contract_hash,
            "diagnostic_source_identity_hash": self.config.diagnostic_source_identity_hash,
            "diagnostic_config_hash": self.config.diagnostic_config_hash,
            "bank_contract_hash": self.config.bank_contract_hash,
            "candidate_bank_hash": self.config.candidate_bank_hash,
            "plan_seal_hash": self.config.plan_seal_hash,
            "precision_runtime_contract_hash": self.config.precision_runtime_contract_hash,
        }

    def make_branch_tracer(self) -> BranchLossTracer:
        return BranchLossTracer(self._contrast_code)

    def backward_loss(
        self,
        loss: torch.Tensor,
        tracer: BranchLossTracer,
    ) -> dict[str, Any]:
        loss.backward()
        return {}

    def run_update(
        self,
        *,
        capture_gradient_vector: bool = False,
    ) -> dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = end_event = None
        started = time.perf_counter()
        tracer = self.make_branch_tracer()
        self._last_branch_tracer = tracer
        with tracer:
            loss = self.model.loss(self.features, self.edge_indices)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite O0 loss")
        backward_metadata = self.backward_loss(loss, tracer)
        gradient_norm, gradient_vector = _gradient_norm_and_vector(
            self.model,
            keep_vector=capture_gradient_vector,
        )
        self.optimizer.step()
        self.model.update_ma()
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize(self.device)
            gpu_event_ms = float(start_event.elapsed_time(end_event))
        else:
            gpu_event_ms = 0.0
        step_wall = time.perf_counter() - started
        self.current_update += 1
        branch = tracer.summary(self.contrast_count)
        resources = current_resources(self.device)
        row: dict[str, Any] = {
            "update": self.current_update,
            "augmentation_id": canonical_json_hash(
                {
                    "augmentation_namespace": self.config.augmentation_namespace,
                    "update": self.current_update,
                }
            ),
            **{
                name: branch[name]
                for name in (
                    "recon_structure_loss",
                    "l_cn",
                    "l_cv",
                    "l_gl",
                )
            },
            **{
                f"weight_{index}": branch["weights"][index]
                for index in range(4)
            },
            "total_loss": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
            "learning_rate": float(
                self.optimizer.param_groups[0]["lr"]
            ),
            "step_wall_time_seconds": step_wall,
            "gpu_event_time_ms": gpu_event_ms,
            **resources,
            "rng_state_sha256": rng_state_sha256(),
            "finite": True,
            **backward_metadata,
        }
        if gradient_vector is not None:
            row["_gradient_vector"] = gradient_vector
        return row

    def branch_tracer_metadata(self) -> dict[str, Any]:
        if self._last_branch_tracer is None:
            return {
                "tracer_payload_incomplete": False,
                "incomplete_events": [],
                "original_exception": None,
                "captured_contrast_count": 0,
                "retains_autograd_graph": False,
            }
        return self._last_branch_tracer.diagnostic_metadata()

    def representation_summaries(self) -> dict[str, Any]:
        with torch.no_grad():
            online = self.model(
                self.features, self.edge_indices, self.config.combine
            )
            target1 = torch.concat(
                [
                    self.model.target_encoder1(self.features, edge)
                    for edge in self.edge_indices
                ],
                dim=-1,
            )
            target2 = torch.concat(
                [
                    self.model.target_encoder2(self.features, edge)
                    for edge in self.edge_indices
                ],
                dim=-1,
            )
        return {
            "online": _tensor_summary(online),
            "target1": _tensor_summary(target1),
            "target2": _tensor_summary(target2),
        }

    def state_snapshot(self) -> dict[str, Any]:
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        optimizer_state = _cpu_clone_nested(
            self.optimizer.state_dict()
        )
        target_state = {
            key: value
            for key, value in model_state.items()
            if key.startswith("target_encoder")
        }
        return {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "target_state": target_state,
            "model_state_sha256": nested_state_sha256(model_state),
            "optimizer_state_sha256": nested_state_sha256(optimizer_state),
            "target_state_sha256": nested_state_sha256(target_state),
            "state_dict_keys": sorted(model_state),
        }

    def resume(self, checkpoint: str | Path) -> dict[str, Any]:
        receipt = load_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            scheduler=None,
            expected_metadata=self.checkpoint_metadata(),
        )
        self.current_update = int(receipt["payload"]["exact_update"])
        return receipt

    def run(
        self,
        *,
        resume_checkpoint: str | Path | None = None,
        audit_updates: Sequence[int] = (),
        signature_updates: Sequence[int] = (),
        stop_at_update: int | None = None,
    ) -> dict[str, Any]:
        if self.output_root.exists() and resume_checkpoint is None:
            raise FileExistsError(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
        if resume_checkpoint is not None:
            self.resume(resume_checkpoint)
        invocation_start_update = self.current_update
        prior_run = None
        if resume_checkpoint is not None:
            prior_marker = (
                self.output_root
                / f"RUN_PAUSED_{self.current_update:06d}.json"
            )
            if not prior_marker.is_file():
                raise FileNotFoundError(prior_marker)
            prior_run = read_json(prior_marker)
            if int(prior_run.get("stop_update", -1)) != self.current_update:
                raise ValueError("paused run/checkpoint update mismatch")
        stop_update = (
            self.config.max_updates
            if stop_at_update is None
            else int(stop_at_update)
        )
        if not self.current_update < stop_update <= self.config.max_updates:
            raise ValueError(
                "stop_at_update must be above the current update and no "
                "greater than max_updates"
            )
        trace_path = self.output_root / "RESOURCE_TRACE.csv"
        prior_sidecar_receipt = (
            self.output_root
            / f"SIDECAR_CAPTURE_RECEIPT_PART_{self.current_update:06d}.json"
        )
        sidecar = (
            O0Sidecar(
                self.output_root,
                checkpoint_schedule=self.config.checkpoint_schedule,
                representation_schedule=(
                    self.config.resolved_representation_schedule()
                ),
                metadata=self.checkpoint_metadata(),
                combine=self.config.combine,
                resume_receipt=(
                    prior_sidecar_receipt
                    if resume_checkpoint is not None
                    and prior_sidecar_receipt.is_file()
                    else None
                ),
                expected_resume_update=(
                    self.current_update if resume_checkpoint is not None else None
                ),
            )
            if self.config.capture_mode == "on"
            else None
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        rows: list[dict[str, Any]] = []
        requested_audits = frozenset(int(value) for value in audit_updates)
        requested_signatures = frozenset(
            int(value) for value in signature_updates
        )
        self.audit_snapshots: dict[int, dict[str, Any]] = {}
        self.signature_snapshots: dict[int, dict[str, Any]] = {}
        started = time.perf_counter()
        with TraceWriter(
            trace_path,
            resume=resume_checkpoint is not None and trace_path.exists(),
            expected_last_update=(
                self.current_update if resume_checkpoint is not None else None
            ),
            durable_flush_interval=self.TRACE_DURABLE_FLUSH_INTERVAL,
        ) as writer:
            while self.current_update < stop_update:
                next_update = self.current_update + 1
                row = self.run_update(
                    capture_gradient_vector=next_update in requested_audits
                )
                public_row = {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
                writer.write(public_row)
                rows.append(public_row)
                if self.current_update in requested_signatures:
                    rng_before = row["rng_state_sha256"]
                    representations = self.representation_summaries()
                    rng_after = rng_state_sha256()
                    if rng_after != rng_before:
                        raise RuntimeError(
                            "engineering signature capture changed RNG"
                        )
                    self.signature_snapshots[self.current_update] = {
                        "trace": public_row,
                        "representations": representations,
                        "rng_before_representation_sha256": rng_before,
                        "rng_after_representation_sha256": rng_after,
                    }
                if self.current_update in requested_audits:
                    self.audit_snapshots[self.current_update] = {
                        "state": self.state_snapshot(),
                        "rng_state_sha256": row["rng_state_sha256"],
                        "gradient_vector": row["_gradient_vector"],
                    }
                if sidecar is not None:
                    sidecar.maybe_capture(
                        update=self.current_update,
                        model=self.model,
                        optimizer=self.optimizer,
                        features=self.features,
                        edge_indices=self.edge_indices,
                    )
                if self.current_update in self.config.checkpoint_schedule:
                    
                    
                    writer.flush()
        elapsed = time.perf_counter() - started
        complete = self.current_update == self.config.max_updates
        sidecar_receipt = None
        if sidecar is not None:
            sidecar_receipt = sidecar.publish_receipt(
                filename=(
                    "SIDECAR_CAPTURE_RECEIPT.json"
                    if complete
                    else (
                        "SIDECAR_CAPTURE_RECEIPT_PART_"
                        f"{self.current_update:06d}.json"
                    )
                )
            )
        with trace_path.open("r", encoding="utf-8", newline="") as stream:
            all_rows = list(__import__("csv").DictReader(stream))
        summary = percentile_summary(all_rows, warmup_updates=20)
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(self.device))
            if torch.cuda.is_available()
            else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(self.device))
            if torch.cuda.is_available()
            else 0
        )
        if prior_run is not None:
            peak_allocated = max(
                peak_allocated,
                int(prior_run["peak_gpu_allocated_bytes"]),
            )
            peak_reserved = max(
                peak_reserved,
                int(prior_run["peak_gpu_reserved_bytes"]),
            )
        cumulative_elapsed = elapsed + (
            float(prior_run["wall_time_seconds"])
            if prior_run is not None
            else 0.0
        )
        result = {
            "schema_version": "metachorus.v53.o0-engineering-run.v1",
            "created_at_utc": utc_now(),
            "status": "PASS" if complete else "PAUSED_AT_CHECKPOINT",
            "config_hash": self.config.identity(),
            "path_set_role": self.config.path_set_role,
            "model_seed": self.config.model_seed,
            "start_update": rows[0]["update"] if rows else self.current_update,
            "stop_update": self.current_update,
            "optimizer_steps": self.current_update,
            "optimizer_steps_this_invocation": (
                self.current_update - invocation_start_update
            ),
            "wall_time_seconds": cumulative_elapsed,
            "model_total_seconds": cumulative_elapsed,
            "wall_time_seconds_this_invocation": elapsed,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "resource_summary": summary,
            "trace_sha256": sha256_file(trace_path),
            "capture_mode": self.config.capture_mode,
            "sidecar": sidecar_receipt,
            "run_role": self.config.run_role,
            "eligible_for_scientific_result": (
                self.config.eligible_for_scientific_result
            ),
            "eligible_for_selector_development": (
                self.config.eligible_for_selector_development
            ),
            "labels_mounted": False,
            "downstream_evaluations": 0,
            "continuation_from_checkpoint": (
                invocation_start_update if prior_run is not None else None
            ),
            "continuation_same_physical_run": prior_run is not None,
        }
        marker = (
            self.output_root / "RUN_COMPLETE.json"
            if complete
            else self.output_root / f"RUN_PAUSED_{self.current_update:06d}.json"
        )
        atomic_write_json(marker, result)
        return result
