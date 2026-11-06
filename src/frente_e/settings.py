"""Configuração central do projeto: caminhos e orçamento de execução.

Fonte única: ``configs/project.toml``. Chaves ausentes ou desconhecidas e tipos
inválidos geram :class:`ConfigError`. Não há valores padrão silenciosos.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frente_e.errors import ConfigError

PROJECT_CONFIG_RELATIVE = Path("configs") / "project.toml"

_PATH_KEYS = (
    "raw_data",
    "processed_data",
    "manifests",
    "experiments",
    "artifacts",
    "ledger",
    "reports",
)
_RUNTIME_KEYS = ("memory_fraction", "max_workers", "native_threads_per_worker")


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_data: Path
    processed_data: Path
    manifests: Path
    experiments: Path
    artifacts: Path
    ledger: Path
    reports: Path


@dataclass(frozen=True)
class RuntimeBudget:
    """Orçamento provisório de execução (DA-07)."""

    memory_fraction: float
    max_workers: int
    native_threads_per_worker: int

    def __post_init__(self) -> None:
        if not 0.0 < self.memory_fraction <= 1.0:
            raise ConfigError(
                f"runtime.memory_fraction deve estar em (0, 1]; recebido {self.memory_fraction}"
            )
        if self.max_workers < 1:
            raise ConfigError(f"runtime.max_workers deve ser >= 1; recebido {self.max_workers}")
        if self.native_threads_per_worker < 1:
            raise ConfigError(
                "runtime.native_threads_per_worker deve ser >= 1; "
                f"recebido {self.native_threads_per_worker}"
            )


@dataclass(frozen=True)
class Settings:
    paths: Paths
    runtime: RuntimeBudget


def find_project_root(start: Path | None = None) -> Path:
    """Sobe a partir de ``start`` (ou do diretório atual) até achar configs/project.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECT_CONFIG_RELATIVE).is_file():
            return candidate
    raise ConfigError(
        f"Raiz do projeto não encontrada: nenhum {PROJECT_CONFIG_RELATIVE.as_posix()} "
        f"em {current} ou diretórios acima."
    )


def load_settings(root: Path | None = None) -> Settings:
    project_root = find_project_root() if root is None else root.resolve()
    config_path = project_root / PROJECT_CONFIG_RELATIVE
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Arquivo de configuração ausente: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML inválido em {config_path}: {exc}") from exc
    return parse_settings(raw, project_root)


def parse_settings(raw: dict[str, Any], root: Path) -> Settings:
    _require_exact_keys(raw, ("paths", "runtime"), "raiz")

    paths_raw = _table(raw, "paths")
    _require_exact_keys(paths_raw, _PATH_KEYS, "[paths]")
    resolved = {key: _resolve_path(paths_raw[key], root, f"paths.{key}") for key in _PATH_KEYS}
    paths = Paths(root=root, **resolved)

    runtime_raw = _table(raw, "runtime")
    _require_exact_keys(runtime_raw, _RUNTIME_KEYS, "[runtime]")
    runtime = RuntimeBudget(
        memory_fraction=_as_float(runtime_raw["memory_fraction"], "runtime.memory_fraction"),
        max_workers=_as_int(runtime_raw["max_workers"], "runtime.max_workers"),
        native_threads_per_worker=_as_int(
            runtime_raw["native_threads_per_worker"], "runtime.native_threads_per_worker"
        ),
    )
    return Settings(paths=paths, runtime=runtime)


def _require_exact_keys(table: dict[str, Any], expected: tuple[str, ...], where: str) -> None:
    missing = sorted(set(expected) - table.keys())
    unknown = sorted(table.keys() - set(expected))
    if missing or unknown:
        raise ConfigError(
            f"Chaves inválidas em {where}: ausentes={missing}, desconhecidas={unknown}"
        )


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw[key]
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] deve ser uma tabela TOML")
    return value


def _resolve_path(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} deve ser um caminho não vazio")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} deve ser numérico; recebido {value!r}")
    return float(value)


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} deve ser inteiro; recebido {value!r}")
    return value
