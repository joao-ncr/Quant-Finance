"""Proveniência de execuções: revisão do código e ambiente de software (DA-04).

Nunca inventa informação: sem Git ou sem commits, ``commit`` é ``None`` e ``detail``
explica o motivo. Estudos com dados reais (``RunKind.STUDY``) exigem commit, código
sem alterações não commitadas e pré-registro (ver :func:`require_reproducible`).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from frente_e.errors import ReproducibilityError
from frente_e.experiments.canonical import JsonValue, sha256_file
from frente_e.experiments.identity import RunKind, RunSpec

# Distribuições cujas versões afetam resultados numéricos ou a execução.
TRACKED_DISTRIBUTIONS = ("frente-e", "numpy", "scipy", "polars", "matplotlib")
# O próprio ledger muda a cada execução; não conta como código "sujo".
DIRTY_CHECK_EXCLUDES = ("experiments/ledger.jsonl",)


def code_revision(root: Path) -> dict[str, JsonValue]:
    try:
        inside = _git(root, "rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        return _no_commit("git não encontrado no PATH")
    if inside.returncode != 0:
        return _no_commit("diretório não é um repositório Git")
    head = _git(root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        return _no_commit("repositório sem commits")
    excludes = [f":(exclude){p}" for p in DIRTY_CHECK_EXCLUDES]
    status = _git(root, "status", "--porcelain", "--untracked-files=normal", "--", ".", *excludes)
    if status.returncode != 0:
        return {
            "vcs": "git",
            "commit": head.stdout.strip(),
            "dirty": None,
            "detail": f"git status falhou: {status.stderr.strip()}",
        }
    changed = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "vcs": "git",
        "commit": head.stdout.strip(),
        "dirty": bool(changed),
        "detail": f"{len(changed)} caminho(s) alterado(s)" if changed else "limpo",
    }


def software_environment(root: Path) -> dict[str, JsonValue]:
    packages: dict[str, JsonValue] = {}
    for dist in TRACKED_DISTRIBUTIONS:
        try:
            packages[dist] = version(dist)
        except PackageNotFoundError:
            packages[dist] = None
    lockfile = root / "uv.lock"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "uv_lock_sha256": sha256_file(lockfile) if lockfile.is_file() else None,
        "executable": sys.executable,
    }


def require_reproducible(spec: RunSpec, revision: dict[str, JsonValue]) -> None:
    """Bloqueia estudos com dados reais que não seriam reconstruíveis."""
    if spec.run_kind is not RunKind.STUDY:
        return
    problems = []
    if revision.get("commit") is None:
        problems.append(f"sem commit ({revision.get('detail')})")
    if revision.get("dirty") is not False:
        problems.append("código com alterações não commitadas ou estado desconhecido")
    if spec.preregistration_reference is None:
        problems.append("sem pré-registro")
    if problems:
        raise ReproducibilityError("Estudo com dados reais bloqueado: " + "; ".join(problems))


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def _no_commit(detail: str) -> dict[str, JsonValue]:
    return {"vcs": "git", "commit": None, "dirty": None, "detail": detail}
