"""Taxa livre de risco por intervalo de retorno (DE-02 §1.3, proposta).

Convenção proposta (aguardando confirmação):
- fonte: CDI diário do SGS/BCB, série 12, publicada em **% ao dia** (a confirmar no
  portal do BCB durante a ingestão). A conversão é ``r_d = valor / 100``. Para séries
  anuais (base 252), ``r_d = (1 + r_a)^{1/252} - 1``;
- a taxa datada em *d* remunera de *d* até o próximo dia útil do CDI;
- o retorno de um ativo no intervalo de pregões (t₋₁, t] é comparado com
  ``rf = Π_{d ∈ [t₋₁, t)} (1 + r_d) - 1``, ou seja, todas as datas do CDI entre o pregão
  anterior (inclusive) e o atual (exclusive). Isso trata feriados locais da B3 em que o
  CDI corre (o intervalo acumula mais de um dia de CDI);
- cada início de intervalo precisa ser uma data presente no CDI. Caso contrário há
  desalinhamento de calendários, e o erro é explícito: nunca se assume taxa zero ou se
  repete a taxa anterior.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from frente_e.core.metrics import TRADING_DAYS_PER_YEAR
from frente_e.errors import ContractViolationError, DataValidationError

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


def daily_rate_from_percent_per_day(values: FloatArray) -> FloatArray:
    arr = _finite_float(values, "values")
    result: FloatArray = arr / 100.0
    return result


def daily_rate_from_annual(
    annual_rate: FloatArray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> FloatArray:
    """``r_d = (1 + r_a)^{1/periods} - 1`` com ``r_a`` em fração (0.10 = 10% a.a.)."""
    arr = _finite_float(annual_rate, "annual_rate")
    if np.any(arr <= -1.0):
        raise ContractViolationError("taxa anual <= -100% não tem conversão diária")
    result: FloatArray = np.expm1(np.log1p(arr) / periods_per_year)
    return result


def risk_free_per_interval(
    rate_dates: DateArray,
    daily_rates: FloatArray,
    interval_starts: DateArray,
    interval_ends: DateArray,
) -> FloatArray:
    """Retorno livre de risco composto em cada intervalo ``[start, end)`` do calendário CDI."""
    dates = np.asarray(rate_dates)
    rates = _finite_float(daily_rates, "daily_rates")
    starts = np.asarray(interval_starts)
    ends = np.asarray(interval_ends)
    for name, arr in (("rate_dates", dates), ("interval_starts", starts), ("interval_ends", ends)):
        if not np.issubdtype(arr.dtype, np.datetime64) or arr.ndim != 1:
            raise ContractViolationError(f"{name} deve ser datetime64 1-D")
    if dates.shape != rates.shape or dates.size == 0:
        raise ContractViolationError("rate_dates e daily_rates devem ter o mesmo tamanho (> 0)")
    if not np.all(dates[1:] > dates[:-1]):
        raise ContractViolationError("rate_dates deve ser estritamente crescente")
    if starts.shape != ends.shape:
        raise ContractViolationError("interval_starts e interval_ends com tamanhos diferentes")
    if not np.all(ends > starts):
        raise ContractViolationError("cada intervalo exige end > start")
    if np.any(rates <= -1.0):
        raise DataValidationError("taxa diária <= -100%")

    first = np.searchsorted(dates, starts, side="left")
    last = np.searchsorted(dates, ends, side="left")
    in_range = first < dates.size
    aligned = np.zeros(starts.shape, dtype=bool)
    aligned[in_range] = dates[first[in_range]] == starts[in_range]
    if not np.all(aligned):
        bad = starts[~aligned][:5]
        raise DataValidationError(
            f"{int(np.count_nonzero(~aligned))} início(s) de intervalo sem taxa do CDI na data "
            f"(ex.: {bad.tolist()}); calendários desalinhados"
        )
    # Sem uma data do CDI >= end não se distingue "dias sem CDI" de "série terminou".
    if np.any(last >= dates.size):
        raise DataValidationError(
            "intervalo termina além da cobertura da série do CDI (sem data >= fim do intervalo)"
        )

    cumulative = np.concatenate(([0.0], np.cumsum(np.log1p(rates))))
    result: FloatArray = np.expm1(cumulative[last] - cumulative[first])
    return result


def _finite_float(values: FloatArray, name: str) -> FloatArray:
    arr = np.asarray(values)
    if arr.dtype != np.float64:
        raise ContractViolationError(f"{name} deve ser float64; recebido {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ContractViolationError(f"{name} contém valores não finitos")
    return arr
