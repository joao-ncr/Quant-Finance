"""DE-08: reprodução das 7 âncoras publicadas (PSR, DSR, MinTRL).

Os valores esperados vêm de tests/fixtures/published/sharpe_anchors.json, digitados como
impressos nos artigos. A tolerância de cada âncora tem duas parcelas, ambas derivadas da
própria publicação, e nenhuma é escolhida para o teste passar:

1. meia unidade da última casa decimal impressa do resultado;
2. propagação do arredondamento das entradas que o artigo imprimiu arredondadas
   (``rounded_inputs``): para cada uma, o maior deslocamento do resultado quando a entrada
   varia ± meia unidade da sua última casa (soma de primeira ordem, sem termos cruzados).

Histórico: a revisão de 2026-11-06 propôs só a parcela 1. A âncora A6 a excede em 1e-5
(0,95351 contra 0,953 ± 0,0005), porque ŜR = 0,458, γ₃ = −2,448 e γ₄ = 10,164 foram
impressos arredondados. Com a parcela 2, a tolerância de A6 cobre a diferença.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from frente_e.core.sharpe_inference import (
    deflated_benchmark,
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)

pytestmark = pytest.mark.reference

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "published" / "sharpe_anchors.json"
DATA: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
ANCHORS: list[dict[str, Any]] = DATA["anchors"]
_RATIO_OF_SQRT = re.compile(r"^(\d+(?:\.\d+)?)/sqrt\((\d+(?:\.\d+)?)\)$")

Inputs = dict[str, float]


def _number(value: float | int | str) -> float:
    """Entradas na notação do artigo: número ou 'a/sqrt(b)'."""
    if isinstance(value, str):
        match = _RATIO_OF_SQRT.match(value)
        if match is None:
            raise ValueError(f"notação não suportada: {value!r}")
        return float(match.group(1)) / math.sqrt(float(match.group(2)))
    return float(value)


def _half_unit(printed: str) -> float:
    """Meia unidade da última casa decimal de um número impresso."""
    exponent = Decimal(printed).as_tuple().exponent
    assert isinstance(exponent, int)
    return 0.5 * 10.0**exponent


def _input_rounding(
    function: Callable[[Inputs], float], inputs: Inputs, raw: dict[str, Any], rounded: list[str]
) -> float:
    """Parcela 2: arredondamento das entradas impressas (ver docstring do módulo)."""
    base = function(inputs)
    total = 0.0
    for name in rounded:
        step = _half_unit(repr(raw[name]))
        shifted = [function({**inputs, name: inputs[name] + s}) for s in (-step, step)]
        total += max(abs(value - base) for value in shifted)
    return total


def _check(
    function: Callable[[Inputs], float],
    inputs: Inputs,
    anchor: dict[str, Any],
    published: str,
    rounded: list[str],
) -> None:
    tolerance = _half_unit(published) + _input_rounding(function, inputs, anchor["inputs"], rounded)
    assert function(inputs) == pytest.approx(float(published), abs=tolerance), anchor["id"]


# ----------------------------------------------------------------------------- funções
def _psr(i: Inputs) -> float:
    return float(
        probabilistic_sharpe_ratio(
            i["sharpe"], i["benchmark"], i["n_obs"], i["skewness"], i["kurtosis"]
        )
    )


def _daily(i: dict[str, Any]) -> Inputs:
    """Conversão feita no próprio artigo: ŜR/√p e V/p, com p = 250 observações/ano."""
    periods = float(i["periods_per_year"])
    return {
        "sharpe": i["sharpe_annualized"] / math.sqrt(periods),
        "sharpe_variance": i["sharpe_variance_annualized"] / periods,
        "n_obs": float(i["n_obs"]),
        "n_trials": float(i["n_trials"]),
        "skewness": float(i["skewness"]),
        "kurtosis": float(i["kurtosis"]),
    }


def _dsr(i: Inputs) -> float:
    return float(
        deflated_sharpe_ratio(
            i["sharpe"],
            i["n_obs"],
            i["skewness"],
            i["kurtosis"],
            n_trials=i["n_trials"],
            sharpe_variance=i["sharpe_variance"],
        )
    )


def _sr0(i: Inputs) -> float:
    return float(deflated_benchmark(i["n_trials"], i["sharpe_variance"]))


# ------------------------------------------------------------------------------- testes
def test_fixture_covers_the_seven_anchors() -> None:
    assert [a["id"] for a in ANCHORS] == [f"A{i}" for i in range(1, 8)]
    assert {a["source"] for a in ANCHORS} == set(DATA["sources"])
    for anchor in ANCHORS:
        for value in anchor["published"].values():
            assert isinstance(value, str), "valores publicados ficam como texto impresso"
        assert set(anchor["rounded_inputs"]) <= set(anchor["inputs"])


@pytest.mark.parametrize(
    "anchor", [a for a in ANCHORS if a["quantity"] == "dsr"], ids=lambda a: a["id"]
)
def test_dsr_anchor(anchor: dict[str, Any]) -> None:
    assert anchor["rounded_inputs"] == []  # entradas hipotéticas, exatas
    inputs = _daily(anchor["inputs"])
    functions = {"deflated_benchmark": _sr0, "dsr": _dsr}
    for name, published in anchor["published"].items():
        _check(functions[name], inputs, anchor, published, [])


@pytest.mark.parametrize(
    "anchor", [a for a in ANCHORS if a["quantity"] == "psr"], ids=lambda a: a["id"]
)
def test_psr_anchor(anchor: dict[str, Any]) -> None:
    inputs = {k: _number(v) for k, v in anchor["inputs"].items()}
    _check(_psr, inputs, anchor, anchor["published"]["psr"], anchor["rounded_inputs"])


def test_min_trl_anchor() -> None:
    (anchor,) = (a for a in ANCHORS if a["quantity"] == "min_trl")
    raw = anchor["inputs"]
    inputs = {k: _number(v) for k, v in raw.items()}

    def trl(i: Inputs) -> float:
        return float(
            min_track_record_length(
                i["sharpe"], i["benchmark"], i["skewness"], i["kurtosis"], confidence=0.95
            )
        )

    def psr_at_trl(i: Inputs) -> float:
        return _psr({**i, "n_obs": i["n_obs_for_psr_check"]})

    assert raw["confidence"] == 0.95
    _check(trl, inputs, anchor, anchor["published"]["min_trl"], [])
    _check(
        psr_at_trl, inputs, anchor, anchor["published"]["psr_at_min_trl"], anchor["rounded_inputs"]
    )


def test_half_unit_rule() -> None:
    assert _half_unit("0.9004") == pytest.approx(5e-5)
    assert _half_unit("0.982") == pytest.approx(5e-4)
    assert _half_unit("59.895") == pytest.approx(5e-4)
    assert _half_unit("0.95") == pytest.approx(5e-3)


def test_input_rounding_term_is_needed_where_inputs_were_rounded() -> None:
    # A6 excede só a parcela 1: documenta por que a parcela 2 existe.
    (a6,) = (a for a in ANCHORS if a["id"] == "A6")
    inputs = {k: _number(v) for k, v in a6["inputs"].items()}
    gap = abs(_psr(inputs) - float(a6["published"]["psr"]))
    half = _half_unit(a6["published"]["psr"])
    assert gap > half
    assert gap < half + _input_rounding(_psr, inputs, a6["inputs"], a6["rounded_inputs"])
