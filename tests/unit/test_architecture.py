"""Invariantes de camadas (DA-01) verificadas por análise estática do código-fonte."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[2] / "src" / "frente_e" / "core"
FORBIDDEN_IN_CORE = (
    "frente_e.data",
    "frente_e.experiments",
    "frente_e.reporting",
    "frente_e.settings",
    "frente_e.runtime",
    "io",
    "pathlib",
    "shutil",
    "sqlite3",
    "subprocess",
    "polars",
    "matplotlib",
)
# np.random.<x> permitido no núcleo apenas para tipos; estado global é proibido.
ALLOWED_NP_RANDOM_ATTRS = {"Generator", "SeedSequence"}

CORE_FILES = sorted(CORE_DIR.rglob("*.py"))


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_core_has_source_files() -> None:
    assert CORE_FILES, "nenhum arquivo encontrado em core/; o teste não verificaria nada"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_does_not_import_io_or_other_layers(path: Path) -> None:
    modules = _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
    offending = {m for m in modules for f in FORBIDDEN_IN_CORE if m == f or m.startswith(f + ".")}
    assert not offending, f"{path.name} importa {sorted(offending)}"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_does_not_use_global_numpy_rng(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offending = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "random"
        and node.attr not in ALLOWED_NP_RANDOM_ATTRS
    ]
    assert not offending, f"{path.name} usa np.random.{offending}: receba um Generator"
