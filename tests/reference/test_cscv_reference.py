"""CSCV/PBO do projeto contra os golden files do pacote R ``pbo`` 1.3.5.

O pacote usa ω̄ = rank/N; o projeto segue o artigo, ω̄ = rank/(N + 1) (DE-05.a). Por isso:
- grandezas que não dependem da convenção são comparadas diretamente: blocos IS, n*,
  posto OOS de n*, R[n*] e R̄[n*];
- ω̄, λ e φ do pacote são reconstruídos a partir do posto com a regra do pacote, o que valida
  a interpretação da referência;
- o φ do projeto é comparado com a definição do artigo aplicada ao posto do R, e o caso
  ``pbo_odd`` (N ímpar) prova que as duas convenções foram de fato separadas.

Tolerâncias: as do manifest.json de cada caso, congeladas antes desta implementação.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from frente_e.core.contracts import ReturnsMatrix
from frente_e.core.cscv import cscv_pbo, partition_performance, rank_partitions
from frente_e.errors import ContractViolationError

pytestmark = pytest.mark.reference

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
CASES = sorted(p for p in GOLDEN.iterdir() if p.name.startswith("pbo_"))
FLOAT_RTOL, FLOAT_ATOL = 1e-12, 1e-15  # manifest.json: tolerance.float_rule


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _values(case: Path) -> NDArray[np.float64]:
    return np.loadtxt(case / "input.csv", delimiter=",", skiprows=1, dtype=np.float64)


def _matrix(case: Path) -> ReturnsMatrix:
    values = _values(case)
    ids = tuple(f"c{j}" for j in range(values.shape[1]))
    return ReturnsMatrix(values, np.arange(values.shape[0]), ids)


def _s_values(case: Path) -> list[int]:
    return sorted(int(p.stem.rsplit("_s", 1)[1]) for p in case.glob("expected_pbo_s*.csv"))


def _summary(case: Path, s: int) -> dict[str, float]:
    return {r["key"]: float(r["value"]) for r in _rows(case / f"expected_pbo_summary_s{s}.csv")}


PARAMS = [(case, s) for case in CASES for s in _s_values(case)]
IDS = [f"{case.name}-S{s}" for case, s in PARAMS]


def test_manifest_tolerance_is_the_one_used_here() -> None:
    manifest: dict[str, Any] = json.loads((CASES[0] / "manifest.json").read_text("utf-8"))
    assert "rtol=1e-12, atol=1e-15" in manifest["tolerance"]["float_rule"]


@pytest.mark.parametrize(("case", "s"), PARAMS, ids=IDS)
def test_intermediate_quantities_match_package(case: Path, s: int) -> None:
    golden = _rows(case / f"expected_pbo_s{s}.csv")
    blocks, perf_is, perf_oos = partition_performance(_values(case), s)
    best, _, rank = rank_partitions(perf_is, perf_oos)
    rows = np.arange(best.size)

    assert [";".join(str(b + 1) for b in combo) for combo in blocks] == [
        g["is_blocks"] for g in golden
    ]
    assert (best + 1).tolist() == [int(g["n_star"]) for g in golden]
    assert rank.tolist() == [float(g["os_rank"]) for g in golden]
    np.testing.assert_allclose(
        perf_is[rows, best], [float(g["rn"]) for g in golden], rtol=FLOAT_RTOL, atol=FLOAT_ATOL
    )
    np.testing.assert_allclose(
        perf_oos[rows, best], [float(g["rbn"]) for g in golden], rtol=FLOAT_RTOL, atol=FLOAT_ATOL
    )


@pytest.mark.parametrize(("case", "s"), PARAMS, ids=IDS)
def test_package_convention_is_reconstructed_from_ranks(case: Path, s: int) -> None:
    """Valida a leitura da referência: ω̄ = rank/N, λ = +inf -> inf_sub, φ = #(λ ≤ 0)/C."""
    golden = _rows(case / f"expected_pbo_s{s}.csv")
    summary = _summary(case, s)
    n = int(summary["n"])
    for g in golden:
        rank = float(g["os_rank"])
        assert float(g["omega_bar"]) == pytest.approx(rank / n, rel=1e-15)
        if rank < n:
            assert float(g["lambda_raw"]) == pytest.approx(math.log(rank / (n - rank)), rel=1e-12)
        else:
            assert math.isinf(float(g["lambda_raw"]))
    package_phi = sum(float(g["os_rank"]) / n <= 0.5 for g in golden) / len(golden)
    assert summary["phi"] == pytest.approx(package_phi, rel=1e-15)


@pytest.mark.parametrize(("case", "s"), PARAMS, ids=IDS)
def test_project_follows_paper_definition(case: Path, s: int) -> None:
    golden = _rows(case / f"expected_pbo_s{s}.csv")
    n = int(_summary(case, s)["n"])
    ranks = np.array([float(g["os_rank"]) for g in golden])
    paper_omega = ranks / (n + 1)
    paper_phi = float(np.mean(paper_omega <= 0.5))

    _, perf_is, perf_oos = partition_performance(_values(case), s)
    best, _, rank = rank_partitions(perf_is, perf_oos)
    np.testing.assert_array_equal(rank, ranks)
    if case.name == "pbo_ties":
        # colunas 1 e 2 idênticas: o projeto rejeita (DE-05.c); o nível baixo acima ainda
        # reproduz o pacote, inclusive postos médios.
        with pytest.raises(ContractViolationError, match="idênticas"):
            cscv_pbo(_matrix(case), s)
        return
    result = cscv_pbo(_matrix(case), s)
    np.testing.assert_array_equal(result.best_in_sample, best)
    np.testing.assert_allclose(result.omega, paper_omega, rtol=1e-15)
    np.testing.assert_allclose(
        result.logit, np.log(ranks / (n + 1 - ranks)), rtol=1e-12, atol=1e-15
    )
    assert np.all(np.isfinite(result.logit))  # rank/(N+1) nunca gera ±inf
    assert result.pbo == paper_phi


def test_odd_case_discriminates_and_project_takes_paper_side() -> None:
    case = GOLDEN / "pbo_odd"
    matrix = _matrix(case)
    for s, discriminates in ((4, False), (8, True), (12, True)):
        package_phi = _summary(case, s)["phi"]
        result = cscv_pbo(matrix, s)
        median_partitions = int(np.count_nonzero(2 * result.oos_rank == matrix.n_configs + 1))
        assert (median_partitions > 0) is discriminates
        assert (result.pbo != package_phi) is discriminates
        if discriminates:
            # a diferença é exatamente a fração de partições na mediana
            assert result.pbo - package_phi == pytest.approx(
                median_partitions / result.n_partitions, rel=1e-12
            )
        # posto OOS máximo: pacote λ = +inf (-> 6); artigo λ = ln N, finito
        top = result.oos_rank == matrix.n_configs
        assert np.any(top)
        np.testing.assert_allclose(result.logit[top], math.log(matrix.n_configs), rtol=1e-15)


def test_signal_case_has_zero_pbo() -> None:
    result = cscv_pbo(_matrix(GOLDEN / "pbo_signal"), 8)
    assert result.pbo == 0.0
    assert set(result.best_in_sample.tolist()) == {0}
