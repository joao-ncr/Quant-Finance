"""Proveniência com repositórios Git reais e temporários (nunca o repositório do projeto)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from frente_e.experiments.provenance import code_revision, software_environment

pytestmark = pytest.mark.integration


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=Teste", "-c", "user.email=teste@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_outside_repository_reports_absence(tmp_path: Path) -> None:
    revision = code_revision(tmp_path)
    assert revision["commit"] is None
    assert revision["dirty"] is None
    assert "não é um repositório" in str(revision["detail"])


def test_repository_without_commits_reports_absence(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    revision = code_revision(tmp_path)
    assert revision["commit"] is None
    assert revision["detail"] == "repositório sem commits"


def test_clean_dirty_and_ledger_exclusion(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "ledger.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")

    clean = code_revision(tmp_path)
    assert isinstance(clean["commit"], str)
    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False

    (tmp_path / "experiments" / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
    assert code_revision(tmp_path)["dirty"] is False  # o ledger não torna o código "sujo"

    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    assert code_revision(tmp_path)["dirty"] is True


def test_software_environment_records_versions_and_lockfile(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    env = software_environment(tmp_path)
    packages = env["packages"]
    assert isinstance(packages, dict)
    assert packages["numpy"]
    assert isinstance(env["uv_lock_sha256"], str)
    assert len(env["uv_lock_sha256"]) == 64
    assert software_environment(tmp_path / "missing")["uv_lock_sha256"] is None
