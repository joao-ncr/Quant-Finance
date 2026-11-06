"""Integridade e coerência interna dos golden files gerados em R.

Estes testes NÃO validam implementações do projeto (PBO e N efetivo ainda não existem).
Garantem que as referências estão intactas (hashes do manifest) e coerentes com as
definições do próprio pacote, para que possam ser usadas na Etapa 3 sem precisar do R.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from frente_e.experiments.canonical import sha256_file

pytestmark = pytest.mark.reference

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
CASES = sorted(p for p in GOLDEN.iterdir() if p.is_dir())
PBO_CASES = [p for p in CASES if p.name.startswith("pbo_")]
MEFF_CASES = [p for p in CASES if p.name.startswith("meff_")]


def _manifest(case: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    return data


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _summary(path: Path) -> dict[str, float]:
    return {row["key"]: float(row["value"]) for row in _rows(path)}


def _n_configs(case: Path) -> int:
    with (case / "input.csv").open(encoding="utf-8") as fh:
        return len(fh.readline().strip().split(","))


def test_golden_cases_exist() -> None:
    assert PBO_CASES
    assert MEFF_CASES


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_manifest_hashes_match(case: Path) -> None:
    manifest = _manifest(case)
    listed = [
        manifest["input"],
        manifest["input_generation"],
        manifest["meta"],
        *manifest["expected"],
    ]
    for entry in listed:
        assert sha256_file(case / entry["file"]) == entry["sha256"], entry["file"]
    on_disk = {p.name for p in case.glob("expected_*.csv")}
    assert on_disk == {e["file"] for e in manifest["expected"]}, "arquivo não congelado"
    assert manifest["reference"]["pbo_version"]
    assert manifest["reference"]["r_version"]
    assert manifest["tolerance"]


@pytest.mark.parametrize("case", PBO_CASES, ids=lambda p: p.name)
def test_pbo_reference_is_internally_consistent(case: Path) -> None:
    n = _n_configs(case)
    for summary_path in sorted(case.glob("expected_pbo_summary_s*.csv")):
        summary = _summary(summary_path)
        s = int(summary["s"])
        rows = _rows(case / f"expected_pbo_s{s}.csv")
        assert len(rows) == math.comb(s, s // 2) == int(summary["n_combos"])
        blocks = [tuple(int(b) for b in r["is_blocks"].split(";")) for r in rows]
        assert len(set(blocks)) == len(blocks)
        assert all(len(b) == s // 2 and set(b) <= set(range(1, s + 1)) for b in blocks)
        lambdas = []
        for r in rows:
            assert 1 <= int(r["n_star"]) <= n
            omega = float(r["omega_bar"])
            assert omega == pytest.approx(float(r["os_rank"]) / n, rel=1e-15)  # pbo: rank/N
            raw = float(r["lambda_raw"])
            if omega < 1:
                assert raw == pytest.approx(math.log(omega / (1 - omega)), rel=1e-12)
                assert float(r["lambda"]) == raw
            else:  # melhor OOS: logit infinito, substituído por inf_sub
                assert math.isinf(raw)
                assert float(r["lambda"]) == summary["inf_sub"]
            lambdas.append(float(r["lambda"]))
        phi = sum(lam <= 0 for lam in lambdas) / len(lambdas)
        assert summary["phi"] == pytest.approx(phi, rel=1e-15)


def test_planted_signal_reference_has_zero_pbo() -> None:
    for path in (GOLDEN / "pbo_signal").glob("expected_pbo_summary_s*.csv"):
        assert _summary(path)["phi"] == 0.0


@pytest.mark.parametrize("case", MEFF_CASES, ids=lambda p: p.name)
def test_meff_reference_is_internally_consistent(case: Path) -> None:
    n = _n_configs(case)
    eigenvalues = [float(r["eigenvalue"]) for r in _rows(case / "expected_eigenvalues.csv")]
    assert len(eigenvalues) == n
    assert sum(eigenvalues) == pytest.approx(n, rel=1e-12)  # traço da matriz de correlação
    assert all(v > 0 for v in eigenvalues)
    assert eigenvalues == sorted(eigenvalues, reverse=True)
    for row in _rows(case / "expected_meff.csv"):
        value = float(row["meff_floor"])
        assert value == int(value)
        assert 1 <= value <= n
