"""Manifesto das execuções de DE-07: leitura estrita e verificação (protocolo v1 §8.3, D5, D7).

O manifesto (TOML versionado) congela, antes da Etapa C: protocolo e seu SHA-256, tabela
de SR verdadeiros (JSON versionado, por SHA-256), commit do código, sementes, células
(hipótese, parâmetros, R, métricas, justificativa, custo) e ordem de execução.

Verificações (erro explícito em qualquer falha; nenhuma correção silenciosa):
- chaves exatas em cada tabela (ausente ou desconhecida é erro);
- SHA-256 do protocolo e da tabela de verdades iguais aos arquivos no disco;
- sementes raiz distintas entre si e da semente da calibração;
- ids e ordens de células únicos; ordem = 1..n;
- células com sinal apontam para uma entrada da tabela de verdades;
- **fase confirmatória:** R = 20 000 (gaussiano), 1000 (momentum H₀), 500 (momentum com
  sinal), D5; soma dos custos estimados ≤ orçamento; ``n_workers`` do orçamento ≥ 1.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frente_e.core.momentum import momentum_family
from frente_e.errors import ContractViolationError
from frente_e.experiments.de07_design import validate_signal

PHASES = ("development", "confirmatory")
D5_REPLICATES = {("gaussian", "H0"): 20_000, ("momentum", "H0"): 1000}
D5_SIGNAL_MOMENTUM = 500
D5_GAUSSIAN = 20_000

_TOP = {"manifest", "seeds", "budget", "evaluation", "common", "scenarios", "cells"}
_MANIFEST = {
    "name",
    "version",
    "phase",
    "created",
    "protocol_path",
    "protocol_version",
    "protocol_sha256",
    "truth_path",
    "truth_sha256",
    "calibration_run_id",
    "calibration_artifact_sha256",
    "code_commit",
    "uv_lock_sha256",
    "python",
    "numpy",
    "scipy",
    "execution_order",
    "notes",
}
_SEEDS = {"gaussian", "momentum", "calibration"}
_BUDGET = {
    "max_hours",
    "n_workers",
    "parallel_speedup",
    "seconds_per_replicate",
    "estimated_hours",
    "basis",
}
_EVALUATION = {
    "alpha",
    "fwer",
    "ci_level",
    "psr_config",
    "n_blocks",
    "pairs",
    "v_candidates",
    "power_cells",
    "pbo_quantiles",
    "pbo_tail_points",
    "decision_rule",
}
_COMMON_CELL = {
    "id",
    "order",
    "level",
    "hypothesis",
    "replicates",
    "metrics",
    "justification",
    "estimated_seconds",
}
_GAUSSIAN_CELL = _COMMON_CELL | {"structure", "n_obs", "signal_pattern", "signal_delta"}
_MOMENTUM_CELL = _COMMON_CELL | {"scenario", "families", "cscv_families"}
_MOMENTUM_OPTIONAL = frozenset({"signal", "truth_key"})
HYPOTHESES = {"gaussian": {"H0", "H1-gauss"}, "momentum": {"H0", "H1-mom", "H1-com"}}


def _exact(
    table: Mapping[str, Any], keys: set[str], where: str, optional: frozenset[str] = frozenset()
) -> None:
    missing = keys - set(table)
    unknown = set(table) - keys - optional
    if missing or unknown:
        raise ContractViolationError(
            f"{where}: chaves ausentes {sorted(missing)}, desconhecidas {sorted(unknown)}"
        )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Manifest:
    path: str
    sha256: str
    version: int
    phase: str
    protocol_path: str
    protocol_version: int
    protocol_sha256: str
    code_commit: str
    seeds: Mapping[str, int]
    budget: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    common: Mapping[str, Any]
    scenarios: Mapping[str, Any]
    cells: tuple[Mapping[str, Any], ...]
    truth: Mapping[str, Any]
    raw: Mapping[str, Any]

    def cell(self, cell_id: str) -> tuple[Mapping[str, Any], int]:
        for cell in self.cells:
            if cell["id"] == cell_id:
                return cell, int(cell["order"])
        raise ContractViolationError(f"célula desconhecida: {cell_id!r}")

    def truth_for(self, cell: Mapping[str, Any]) -> dict[str, float] | None:
        """SR verdadeiros por configuração (F144) da célula; ``None`` sob H₀ (todos 0,
        sobreajuste verdadeiro indefinido)."""
        if cell["hypothesis"] == "H0":
            return None
        entry = self.truth["cells"][cell["truth_key"]]["truth"]
        return dict(zip(entry["config_ids"], entry["true_sharpe"], strict=True))

    def estimated_hours(self) -> float:
        """Horas de relógio: células de momentum em ``n_workers`` processos (custo
        sequencial / ganho medido); células gaussianas num único processo."""
        speedup = float(self.budget["parallel_speedup"])
        seconds = sum(
            float(c["estimated_seconds"]) / (speedup if c["level"] == "momentum" else 1.0)
            for c in self.cells
        )
        return seconds / 3600.0


def load_manifest(path: Path, *, project_root: Path) -> Manifest:
    """Lê e verifica o manifesto. ``path`` pode ser relativo à raiz do projeto."""
    full = path if path.is_absolute() else project_root / path
    data = full.read_bytes()
    raw = tomllib.loads(data.decode("utf-8"))
    _exact(raw, _TOP, "raiz")
    meta = raw["manifest"]
    _exact(meta, _MANIFEST, "[manifest]")
    if meta["phase"] not in PHASES:
        raise ContractViolationError(f"fase inválida: {meta['phase']!r}")
    _exact(raw["seeds"], _SEEDS, "[seeds]")
    seeds = {k: int(v) for k, v in raw["seeds"].items()}
    if len(set(seeds.values())) != len(seeds):
        raise ContractViolationError("sementes raiz devem ser distintas (calibração × avaliação)")
    _exact(raw["budget"], _BUDGET, "[budget]")
    _exact(raw["evaluation"], _EVALUATION, "[evaluation]")
    for rel, key in (
        (meta["protocol_path"], "protocol_sha256"),
        (meta["truth_path"], "truth_sha256"),
    ):
        actual = sha256_file(project_root / rel)
        if actual != meta[key]:
            raise ContractViolationError(
                f"{rel}: SHA-256 {actual} difere do congelado {meta[key]} (exige nova versão)"
            )
    truth = json.loads((project_root / meta["truth_path"]).read_bytes())
    f144 = [c.config_id for c in momentum_family("F144")]
    cells = tuple(raw["cells"])
    ids = [c.get("id") for c in cells]
    if len(set(ids)) != len(ids):
        raise ContractViolationError("ids de células repetidos")
    orders = sorted(int(c.get("order", 0)) for c in cells)
    if orders != list(range(1, len(cells) + 1)):
        raise ContractViolationError("ordens das células devem ser 1..n, sem repetição")
    if list(meta["execution_order"]) != [c["id"] for c in sorted(cells, key=lambda c: c["order"])]:
        raise ContractViolationError("execution_order não corresponde às ordens das células")
    confirmatory = meta["phase"] == "confirmatory"
    for cell in cells:
        level = cell.get("level")
        where = f"célula {cell.get('id')!r}"
        if level == "gaussian":
            _exact(cell, _GAUSSIAN_CELL, where)
        elif level == "momentum":
            _exact(cell, _MOMENTUM_CELL, where, _MOMENTUM_OPTIONAL)
            if cell["scenario"] not in raw["scenarios"]:
                raise ContractViolationError(f"{where}: cenário desconhecido")
            if not set(cell["cscv_families"]) <= set(cell["families"]):
                raise ContractViolationError(f"{where}: CSCV só em famílias da célula")
            for family in cell["families"]:
                momentum_family(family)
        else:
            raise ContractViolationError(f"{where}: nível inválido {level!r}")
        if cell["hypothesis"] not in HYPOTHESES[level]:
            raise ContractViolationError(f"{where}: hipótese inválida para o nível")
        signal_needed = level == "momentum" and cell["hypothesis"] != "H0"
        if level == "momentum":
            if signal_needed != ("signal" in cell) or signal_needed != ("truth_key" in cell):
                raise ContractViolationError(f"{where}: sinal e truth_key só (e sempre) com sinal")
            if signal_needed:
                validate_signal(cell["signal"])
                entry = truth["cells"].get(cell["truth_key"])
                if entry is None or entry["truth"]["config_ids"] != f144:
                    raise ContractViolationError(f"{where}: verdade ausente ou incompleta")
                if entry["signal"] != cell["signal"]:
                    raise ContractViolationError(f"{where}: sinal difere do calibrado")
        if confirmatory:
            expected = (
                D5_GAUSSIAN
                if level == "gaussian"
                else D5_REPLICATES[("momentum", "H0")]
                if cell["hypothesis"] == "H0"
                else D5_SIGNAL_MOMENTUM
            )
            if int(cell["replicates"]) != expected:
                raise ContractViolationError(
                    f"{where}: R = {cell['replicates']} viola D5 (esperado {expected})"
                )
    manifest = Manifest(
        path=str(path.as_posix()),
        sha256=hashlib.sha256(data).hexdigest(),
        version=int(meta["version"]),
        phase=str(meta["phase"]),
        protocol_path=str(meta["protocol_path"]),
        protocol_version=int(meta["protocol_version"]),
        protocol_sha256=str(meta["protocol_sha256"]),
        code_commit=str(meta["code_commit"]),
        seeds=seeds,
        budget=raw["budget"],
        evaluation=raw["evaluation"],
        common=raw["common"],
        scenarios=raw["scenarios"],
        cells=cells,
        truth=truth,
        raw=raw,
    )
    if confirmatory and manifest.estimated_hours() > float(raw["budget"]["max_hours"]):
        raise ContractViolationError(
            f"custo estimado {manifest.estimated_hours():.2f} h excede o orçamento "
            f"{raw['budget']['max_hours']} h (D5)"
        )
    return manifest


# ------------------------------------------------------------------ escrita determinística
_BARE_KEY = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")


def _key(key: str) -> str:
    return key if key and set(key) <= _BARE_KEY else json.dumps(key, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractViolationError("TOML do manifesto não aceita valores não finitos")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }"
    raise ContractViolationError(f"tipo não suportado no manifesto: {type(value).__name__}")


def to_toml(data: Mapping[str, Any]) -> str:
    """TOML determinístico para o formato do manifesto: tabelas de topo (com subtabelas
    de um nível), arrays de tabelas e valores inline. A ordem das chaves é preservada."""
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
            for item in value:
                lines.append(f"[[{_key(key)}]]")
                lines.extend(f"{_key(k)} = {_value(v)}" for k, v in item.items())
                lines.append("")
        elif isinstance(value, Mapping):
            scalars = {k: v for k, v in value.items() if not isinstance(v, Mapping)}
            nested = {k: v for k, v in value.items() if isinstance(v, Mapping)}
            if scalars or not nested:
                lines.append(f"[{_key(key)}]")
                lines.extend(f"{_key(k)} = {_value(v)}" for k, v in scalars.items())
                lines.append("")
            for sub, table in nested.items():
                lines.append(f"[{_key(key)}.{_key(sub)}]")
                lines.extend(f"{_key(k)} = {_value(v)}" for k, v in table.items())
                lines.append("")
        else:
            raise ContractViolationError(f"chave de topo {key!r} deve ser tabela")
    return "\n".join(lines).rstrip("\n") + "\n"
