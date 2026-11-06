"""Contrato versionado dos registros do ledger (schema v1, DA-03).

Cada linha do ledger é um registro com exatamente os campos de :data:`FIELDS`.
Uma execução gera um registro ``started`` (com a configuração completa) e, depois,
no máximo um registro terminal (``completed``, ``failed`` ou ``aborted``), que repete
só a identidade (experiment_id, run_id, run_kind, fingerprint) e deixa os campos de
configuração como ``null``. Assim o JSONL fica leve e a configuração aparece uma vez.

``record_hash`` = SHA-256 da forma canônica do registro sem o próprio ``record_hash``.
``previous_record_hash`` encadeia o registro anterior; o primeiro usa :data:`GENESIS_HASH`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from frente_e.experiments.canonical import canonical_bytes, sha256_hex
from frente_e.experiments.identity import (
    EXPERIMENT_ID_PATTERN,
    HEX64_PATTERN,
    RUN_ID_PATTERN,
    RunKind,
)

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
GENESIS_HASH = "0" * 64
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")

FIELDS = (
    "schema_version",
    "sequence",
    "experiment_id",
    "run_id",
    "run_kind",
    "configuration_fingerprint",
    "timestamp_utc",
    "status",
    "parameters",
    "universe_reference",
    "preregistration_reference",
    "random_seeds",
    "software_environment",
    "code_revision",
    "input_data_references",
    "results_reference",
    "error_information",
    "previous_record_hash",
    "record_hash",
)
CONFIG_FIELDS = (
    "parameters",
    "universe_reference",
    "preregistration_reference",
    "random_seeds",
    "software_environment",
    "code_revision",
    "input_data_references",
)
ARTIFACT_FIELDS = frozenset({"relative_path", "sha256", "size_bytes", "media_type", "description"})


class RunStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.ABORTED})


class RecordShapeError(ValueError):
    """Registro com forma inválida (convertido em LedgerIntegrityError pelo leitor)."""


def compute_record_hash(record: Mapping[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_hash"}
    return sha256_hex(canonical_bytes(body))


def validate_record(record: Mapping[str, Any]) -> RunStatus:
    """Valida a forma de um registro isolado. Não verifica a cadeia nem as transições."""
    version = record.get("schema_version")
    if isinstance(version, bool) or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise RecordShapeError(f"schema_version não suportada: {version!r}")
    keys = set(record)
    if keys != set(FIELDS):
        raise RecordShapeError(
            f"campos inválidos: ausentes={sorted(set(FIELDS) - keys)}, "
            f"desconhecidos={sorted(keys - set(FIELDS))}"
        )
    _check(_is_int(record["sequence"]) and record["sequence"] >= 0, "sequence inválida")
    _check_pattern(record, "experiment_id", EXPERIMENT_ID_PATTERN)
    _check_pattern(record, "run_id", RUN_ID_PATTERN)
    _check_pattern(record, "configuration_fingerprint", HEX64_PATTERN)
    _check_pattern(record, "timestamp_utc", TIMESTAMP_PATTERN)
    _check_pattern(record, "previous_record_hash", HEX64_PATTERN)
    _check_pattern(record, "record_hash", HEX64_PATTERN)
    _check(record["run_kind"] in {k.value for k in RunKind}, "run_kind inválido")
    _check(record["status"] in {s.value for s in RunStatus}, "status inválido")
    status = RunStatus(record["status"])

    if status is RunStatus.STARTED:
        _check(isinstance(record["parameters"], dict) and record["parameters"], "parameters")
        _check(_is_optional_str(record["universe_reference"]), "universe_reference")
        _check(
            record["preregistration_reference"] is None
            or isinstance(record["preregistration_reference"], dict),
            "preregistration_reference",
        )
        _check(isinstance(record["random_seeds"], dict), "random_seeds")
        _check(isinstance(record["software_environment"], dict), "software_environment")
        _check(isinstance(record["code_revision"], dict), "code_revision")
        _check(isinstance(record["input_data_references"], list), "input_data_references")
        _check(record["results_reference"] is None, "results_reference deve ser null em started")
        _check(record["error_information"] is None, "error_information deve ser null em started")
        return status

    for name in CONFIG_FIELDS:
        _check(record[name] is None, f"{name} deve ser null em registro {status}")
    if status is RunStatus.COMPLETED:
        results = record["results_reference"]
        _check(isinstance(results, list), "results_reference deve ser lista em completed")
        for item in results:
            _check(isinstance(item, dict) and set(item) == ARTIFACT_FIELDS, "artefato inválido")
        _check(record["error_information"] is None, "error_information deve ser null")
    else:
        _check(record["results_reference"] is None, f"results_reference deve ser null em {status}")
        error = record["error_information"]
        _check(
            isinstance(error, dict)
            and isinstance(error.get("type"), str)
            and isinstance(error.get("message"), str),
            "error_information exige 'type' e 'message'",
        )
    return status


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_optional_str(value: object) -> bool:
    return value is None or (isinstance(value, str) and bool(value.strip()))


def _check(condition: object, message: str) -> None:
    if not condition:
        raise RecordShapeError(message)


def _check_pattern(record: Mapping[str, Any], name: str, pattern: re.Pattern[str]) -> None:
    value = record[name]
    _check(isinstance(value, str) and pattern.match(value), f"{name} inválido: {value!r}")
