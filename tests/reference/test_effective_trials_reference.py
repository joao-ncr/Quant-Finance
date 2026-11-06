"""Estimadores de N efetivo comparados com os golden files de ``poolr::meff`` (DE-03.a).

O pacote aplica ``floor()``; o projeto não. A comparação justa é ``floor(nosso) ==
poolr`` (igualdade exata) mais os autovalores com a tolerância do manifesto (rtol 1e-9).
Isso não verifica a parte fracionária de Nyholt e Galwey, porque o pacote a descarta.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from frente_e.core.effective_trials import ESTIMATORS, correlation_eigenvalues

pytestmark = pytest.mark.reference

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
CASES = sorted(p for p in GOLDEN.iterdir() if p.name.startswith("meff_"))
EIGENVALUE_RTOL = 1e-9  # manifest.json: tolerance.eigenvalue


def _correlation(case: Path) -> NDArray[np.float64]:
    values = np.loadtxt(case / "input.csv", delimiter=",", skiprows=1, dtype=np.float64)
    return np.asarray(np.corrcoef(values, rowvar=False), dtype=np.float64)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_cases_exist() -> None:
    assert [c.name for c in CASES] == ["meff_blocks", "meff_independent"]


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_eigenvalues_match_r(case: Path) -> None:
    expected = np.array([float(r["eigenvalue"]) for r in _rows(case / "expected_eigenvalues.csv")])
    np.testing.assert_allclose(
        correlation_eigenvalues(_correlation(case)), expected, rtol=EIGENVALUE_RTOL, atol=0
    )


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_floor_of_continuous_estimate_matches_poolr(case: Path) -> None:
    corr = _correlation(case)
    for row in _rows(case / "expected_meff.csv"):
        ours = ESTIMATORS[row["method"]](corr)
        assert math.floor(ours) == int(row["meff_floor"]), row["method"]
