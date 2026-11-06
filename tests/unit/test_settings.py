from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from frente_e.errors import ConfigError
from frente_e.settings import find_project_root, load_settings, parse_settings

VALID: dict[str, Any] = {
    "paths": {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "manifests": "data/manifests",
        "experiments": "experiments",
        "artifacts": "experiments/artifacts",
        "ledger": "experiments/ledger.jsonl",
        "reports": "reports",
    },
    "runtime": {"memory_fraction": 0.6, "max_workers": 2, "native_threads_per_worker": 1},
}


def _with(section: str, key: str, value: object) -> dict[str, Any]:
    raw = copy.deepcopy(VALID)
    raw[section][key] = value
    return raw


def test_repository_config_loads(project_root: Path) -> None:
    settings = load_settings(project_root)
    assert settings.paths.root == project_root.resolve()
    assert settings.paths.ledger == project_root.resolve() / "experiments" / "ledger.jsonl"
    assert 0 < settings.runtime.memory_fraction <= 1


def test_relative_paths_resolve_against_root(tmp_path: Path) -> None:
    settings = parse_settings(VALID, tmp_path)
    assert settings.paths.raw_data == tmp_path / "data" / "raw"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    absolute = (tmp_path / "elsewhere").resolve()
    settings = parse_settings(_with("paths", "raw_data", str(absolute)), tmp_path)
    assert settings.paths.raw_data == absolute


@pytest.mark.parametrize("section", ["paths", "runtime"])
def test_missing_key_is_error(section: str, tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw[section].popitem()
    with pytest.raises(ConfigError, match="ausentes"):
        parse_settings(raw, tmp_path)


@pytest.mark.parametrize("section", ["paths", "runtime"])
def test_unknown_key_is_error(section: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="desconhecidas"):
        parse_settings(_with(section, "typo_key", 1), tmp_path)


def test_unknown_top_level_section_is_error(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["extra"] = {}
    with pytest.raises(ConfigError):
        parse_settings(raw, tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("memory_fraction", 0.0),
        ("memory_fraction", 1.5),
        ("memory_fraction", "0.6"),
        ("memory_fraction", True),
        ("max_workers", 0),
        ("max_workers", 2.0),
        ("max_workers", True),
        ("native_threads_per_worker", 0),
    ],
)
def test_invalid_runtime_values_are_rejected(key: str, value: object, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        parse_settings(_with("runtime", key, value), tmp_path)


@pytest.mark.parametrize("value", ["", "   ", 3])
def test_invalid_paths_are_rejected(value: object, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        parse_settings(_with("paths", "ledger", value), tmp_path)


def test_malformed_toml_is_config_error(project_root: Path) -> None:
    (project_root / "configs" / "project.toml").write_text("[paths\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="TOML inválido"):
        load_settings(project_root)


def test_find_project_root_walks_up(project_root: Path) -> None:
    nested = project_root / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == project_root.resolve()


def test_find_project_root_fails_explicitly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Raiz do projeto"):
        find_project_root(tmp_path)
