from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "frente_e", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def test_info_runs_from_project_root(project_root: Path) -> None:
    result = _run("info", "--root", str(project_root), cwd=project_root)
    assert result.returncode == 0, result.stderr
    assert "frente-e" in result.stdout
    assert "orçamento de memória" in result.stdout


def test_ledger_commands_on_empty_and_torn_ledger(project_root: Path) -> None:
    root = str(project_root)
    result = _run("ledger", "--root", root, "verify", cwd=project_root)
    assert result.returncode == 0, result.stderr
    assert "OK: 0 registros" in result.stdout

    ledger = project_root / "experiments" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b'{"partial')
    torn = _run("ledger", "--root", root, "verify", cwd=project_root)
    assert torn.returncode == 2
    assert "incompleta" in torn.stderr

    recovered = _run("ledger", "--root", root, "recover-tail", cwd=project_root)
    assert recovered.returncode == 0, recovered.stderr
    assert "quarentena" in recovered.stdout
    head = _run("ledger", "--root", root, "head", cwd=project_root)
    assert head.stdout.strip() == "0:" + "0" * 64


def test_info_without_config_fails_cleanly(tmp_path: Path) -> None:
    result = _run("info", cwd=tmp_path)
    assert result.returncode == 2
    assert "Raiz do projeto" in result.stderr
    assert "Traceback" not in result.stderr
