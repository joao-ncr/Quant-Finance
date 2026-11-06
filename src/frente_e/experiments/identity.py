"""Identidade de experimentos, execuções e configurações (DA-03).

- ``experiment_id``: nome estável e legível do estudo lógico (escolhido por pessoa).
- ``run_id``: uma execução individual; aleatório (UUID4), nunca derivado da configuração.
- ``configuration_fingerprint``: SHA-256 da configuração científica canônica.

O fingerprint cobre exatamente: parâmetros científicos (família, parâmetros da
estratégia, períodos, métrica de seleção, configuração estatística, conforme o
estudo declarar), referência do universo, referência e versão do pré-registro e
sementes aleatórias. Ficam de fora: ``experiment_id``, ``run_kind``, horário, ambiente,
revisão do código e referências de dados de entrada. Esses itens são registrados no
ledger, mas não definem *qual* configuração foi testada.

Consequências:
- mesma configuração e sementes diferentes geram fingerprints diferentes;
- uma repetição idêntica tem o mesmo fingerprint e um ``run_id`` novo. Ela deve
  reproduzir os resultados quando também coincidirem as referências de dados, a
  revisão do código e o ambiente (todos registrados).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from frente_e.errors import ContractViolationError
from frente_e.experiments.canonical import JsonValue, canonical_bytes, sha256_hex, to_canonical

FINGERPRINT_VERSION = 1
EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
RUN_ID_PATTERN = re.compile(r"^run-[0-9a-f]{32}$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SEED_UPPER_BOUND = 2**63  # sementes cabem em int64 para leitores JSON de outras linguagens


class RunKind(StrEnum):
    DEVELOPMENT = "development"  # exploração de código; nunca reportável
    SYNTHETIC = "synthetic"  # dados sintéticos (validação do procedimento)
    STUDY = "study"  # dados reais; exige pré-registro e código commitado


class InputKind(StrEnum):
    FILE = "file"
    DATASET = "dataset"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class InputReference:
    kind: InputKind
    uri: str
    description: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ContractViolationError("InputReference.uri vazio")
        if self.sha256 is not None and not HEX64_PATTERN.match(self.sha256):
            raise ContractViolationError(f"sha256 inválido: {self.sha256!r}")
        if self.kind in (InputKind.FILE, InputKind.DATASET) and self.sha256 is None:
            raise ContractViolationError(f"Entrada do tipo {self.kind} exige sha256 do conteúdo")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "kind": str(self.kind),
            "uri": self.uri,
            "description": self.description,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PreregistrationReference:
    path: str
    version: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ContractViolationError("PreregistrationReference.path vazio")
        if isinstance(self.version, bool) or self.version < 1:
            raise ContractViolationError("PreregistrationReference.version deve ser >= 1")
        if not HEX64_PATTERN.match(self.sha256):
            raise ContractViolationError(f"sha256 inválido: {self.sha256!r}")

    def to_json(self) -> dict[str, JsonValue]:
        return {"path": self.path, "version": self.version, "sha256": self.sha256}


@dataclass(frozen=True)
class RunSpec:
    """Tudo o que define uma execução antes de ela começar."""

    experiment_id: str
    run_kind: RunKind
    parameters: Mapping[str, Any]
    random_seeds: Mapping[str, int]
    universe_reference: str | None = None
    preregistration_reference: PreregistrationReference | None = None
    input_data_references: tuple[InputReference, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not EXPERIMENT_ID_PATTERN.match(self.experiment_id):
            raise ContractViolationError(
                f"experiment_id inválido {self.experiment_id!r}: use [a-z0-9._-], 3 a 80 caracteres"
            )
        if not isinstance(self.parameters, Mapping) or not self.parameters:
            raise ContractViolationError("parameters deve ser um mapeamento não vazio")
        to_canonical(self.parameters, "parameters")  # valida tipos já na construção
        for name, seed in self.random_seeds.items():
            if not isinstance(name, str) or not name:
                raise ContractViolationError("nomes de semente devem ser str não vazias")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ContractViolationError(f"semente {name!r} deve ser int")
            if not 0 <= seed < SEED_UPPER_BOUND:
                raise ContractViolationError(f"semente {name!r} fora de [0, 2^63)")
        if self.universe_reference is not None and not self.universe_reference.strip():
            raise ContractViolationError("universe_reference vazio; use None se não se aplica")


def configuration_fingerprint(spec: RunSpec) -> str:
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "parameters": spec.parameters,
        "universe_reference": spec.universe_reference,
        "preregistration_reference": (
            None
            if spec.preregistration_reference is None
            else spec.preregistration_reference.to_json()
        ),
        "random_seeds": dict(spec.random_seeds),
    }
    return sha256_hex(canonical_bytes(payload))


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"
