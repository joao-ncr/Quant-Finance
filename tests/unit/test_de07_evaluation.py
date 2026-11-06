"""Regra D3 de seleção de (N, V), critério 2b do PSR e resumos da PBO (DE-07)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from frente_e.experiments import de07_evaluation as ev

EV: dict[str, Any] = {
    "alpha": 0.05,
    "fwer": 0.05,
    "ci_level": 0.95,
    "power_cells": ["H1a", "H1b"],
    "pbo_quantiles": [0.05, 0.5, 0.95],
    "pbo_tail_points": [0.5, 0.9],
}


def _pair(k: int, n: int, *, undefined: int = 0, power_hits: int | None = None) -> dict[str, Any]:
    hits = k if power_hits is None else power_hits
    return {
        "undefined": undefined,
        "rejection": ev.rate(k, n, 0.95),
        "true_discovery": ev.rate(hits, n, 0.95),
        "indicators": [True] * hits + [False] * (n - hits),
    }


def _cells(
    spec: dict[str, dict[str, tuple[int, int]]], undefined: frozenset[str] = frozenset()
) -> Any:
    """spec[cell][pair] = (k, n); a mesma família 'F16' em todas as células."""
    return {
        cell: {
            "F16": {
                "pairs": {
                    pair: _pair(k, n, undefined=int(f"{cell}:{pair}" in undefined))
                    for pair, (k, n) in pairs.items()
                }
            }
        }
        for cell, pairs in spec.items()
    }


HYP = {"N1": "H0", "N2": "H0", "H1a": "H1-mom", "H1b": "H1-mom"}
PAIRS = ["a", "b", "c"]


def test_size_violation_excludes_a_pair_and_maximin_picks_the_best_worst_case() -> None:
    cells = _cells(
        {
            "N1": {"a": (50, 1000), "b": (10, 1000), "c": (95, 1000)},  # c: 9,5% ≫ 5%
            "N2": {"a": (45, 1000), "b": (12, 1000), "c": (40, 1000)},
            "H1a": {"a": (300, 500), "b": (200, 500), "c": (400, 500)},
            "H1b": {"a": (250, 500), "b": (260, 500), "c": (450, 500)},
        }
    )
    out = ev.select_pair(cells, HYP, PAIRS, EV)
    assert out["pairs"]["c"]["size_violations"] == ["N1/F16"]
    assert not out["pairs"]["c"]["admissible"]
    assert out["pairs"]["a"]["min_power"] == pytest.approx(0.5)
    assert out["pairs"]["b"]["min_power"] == pytest.approx(0.4)
    assert out["selected"] == "a"


def test_nominal_rate_is_not_a_violation_and_holm_is_applied() -> None:
    """Exatamente 5% não é violação; 6,2% em 1000 (valor-p ≈ 0,05) não é rejeitado por Holm
    com 2 testes do par (menor limiar 0,025): a decisão usa a incerteza de Monte Carlo."""
    cells = _cells(
        {
            "N1": {"a": (50, 1000), "b": (62, 1000)},
            "N2": {"a": (50, 1000), "b": (50, 1000)},
            "H1a": {"a": (100, 500), "b": (100, 500)},
        }
    )
    out = ev.select_pair(cells, {**HYP}, ["a", "b"], {**EV, "power_cells": ["H1a"]})
    assert out["pairs"]["a"]["size_violations"] == []
    assert out["pairs"]["b"]["size_violations"] == []


def test_undefined_pair_is_inadmissible() -> None:
    cells = _cells(
        {"N1": {"a": (10, 1000)}, "H1a": {"a": (300, 500)}}, undefined=frozenset({"H1a:a"})
    )
    out = ev.select_pair(cells, HYP, ["a"], {**EV, "power_cells": ["H1a"]})
    assert out["pairs"]["a"]["undefined_in"] == ["H1a/F16"]
    assert out["selected"] is None
    assert "nenhum par admissível" in out["reason"]


def test_no_admissible_pair_selects_nothing() -> None:
    cells = _cells({"N1": {"a": (200, 1000)}, "H1a": {"a": (500, 500)}})
    out = ev.select_pair(cells, HYP, ["a"], {**EV, "power_cells": ["H1a"]})
    assert out["selected"] is None


def test_ties_within_two_paired_standard_errors_break_by_mean_power() -> None:
    cells = _cells(
        {
            "N1": {"a": (10, 1000), "b": (10, 1000)},
            "H1a": {"a": (251, 500), "b": (250, 500)},
            "H1b": {"a": (300, 500), "b": (400, 500)},
        }
    )
    out = ev.select_pair(cells, HYP, ["a", "b"], EV)
    assert set(out["tied"]) == {"a", "b"}
    assert out["selected"] == "b"  # maior poder médio


def test_psr_level_bonferroni() -> None:
    cells = {
        "P0": {"pvalue_two_sided": 0.5, "rate": 0.05},
        "P1": {"pvalue_two_sided": 0.01, "rate": 0.08},
    }
    out = ev.psr_level_tests(cells, 0.05)
    assert out["P0"]["bonferroni_reject"] is False
    assert out["P1"]["bonferroni_reject"] is True


def test_pbo_distribution_and_auc() -> None:
    rng = np.random.default_rng(0)
    null = rng.uniform(size=1000)
    signal = rng.uniform(0.0, 0.5, size=500)
    dist = ev.pbo_distribution(list(null), EV)
    assert dist["tail"]["0.5"] == pytest.approx(float(np.mean(null >= 0.5)))
    point, lo, hi = dist["quantiles"]["0.05"]
    assert lo <= point <= hi
    assert ev.pbo_auc(signal, null) == pytest.approx(0.75, abs=0.03)
    diff = ev.paired_mean_difference([0.2, 0.4], [0.1, 0.1])
    assert diff["diff"] == pytest.approx(0.2)
