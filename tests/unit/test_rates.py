"""Taxa livre de risco por intervalo (DE-02 §1.3, proposta)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from frente_e.core.rates import (
    daily_rate_from_annual,
    daily_rate_from_percent_per_day,
    risk_free_per_interval,
)
from frente_e.errors import ContractViolationError, DataValidationError


def _d(*days: str) -> np.ndarray[Any, np.dtype[np.datetime64]]:
    return np.array(days, dtype="datetime64[D]")


def _f(*values: float) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array(values, dtype=np.float64)


# Segunda 2026-03-02 a sexta 2026-03-06, seguida da segunda 2026-03-09.
CDI_DATES = _d("2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-09")
CDI_RATES = _f(0.00040, 0.00041, 0.00042, 0.00043, 0.00044, 0.00045)


def test_percent_per_day_conversion() -> None:
    np.testing.assert_allclose(daily_rate_from_percent_per_day(_f(0.054266)), [0.00054266])


@given(st.floats(-0.5, 2.0))
def test_annual_to_daily_compounds_back(annual: float) -> None:
    daily = daily_rate_from_annual(_f(annual))
    assert (1 + daily[0]) ** 252 == pytest.approx(1 + annual, rel=1e-12)


def test_consecutive_trading_days_use_start_date_rate() -> None:
    starts = _d("2026-03-02", "2026-03-03")
    ends = _d("2026-03-03", "2026-03-04")
    rf = risk_free_per_interval(CDI_DATES, CDI_RATES, starts, ends)
    np.testing.assert_allclose(rf, [0.00040, 0.00041], rtol=1e-12)


def test_weekend_interval_accrues_friday_rate() -> None:
    rf = risk_free_per_interval(CDI_DATES, CDI_RATES, _d("2026-03-06"), _d("2026-03-09"))
    np.testing.assert_allclose(rf, [0.00044], rtol=1e-12)


def test_b3_holiday_with_cdi_compounds_two_days() -> None:
    # B3 fechada na quarta, CDI corre: intervalo terça -> quinta acumula terça e quarta.
    rf = risk_free_per_interval(CDI_DATES, CDI_RATES, _d("2026-03-03"), _d("2026-03-05"))
    np.testing.assert_allclose(rf, [(1.00041 * 1.00042) - 1], rtol=1e-12)


def test_variable_cdi_full_path() -> None:
    starts, ends = CDI_DATES[:-1], CDI_DATES[1:]
    rf = risk_free_per_interval(CDI_DATES, CDI_RATES, starts, ends)
    np.testing.assert_allclose(rf, CDI_RATES[:-1], rtol=1e-12)
    assert len(set(rf.tolist())) == 5  # nenhuma taxa repetida ou "atual" aplicada ao passado


def test_start_without_cdi_is_explicit_error() -> None:
    with pytest.raises(DataValidationError, match="desalinhados"):
        risk_free_per_interval(CDI_DATES, CDI_RATES, _d("2026-03-07"), _d("2026-03-09"))


def test_interval_beyond_coverage_is_error() -> None:
    with pytest.raises(DataValidationError, match="cobertura"):
        risk_free_per_interval(CDI_DATES, CDI_RATES, _d("2026-03-09"), _d("2026-03-10"))


def test_missing_cdi_value_is_rejected_not_filled() -> None:
    rates = CDI_RATES.copy()
    rates[2] = np.nan
    with pytest.raises(ContractViolationError, match="não finitos"):
        risk_free_per_interval(CDI_DATES, rates, _d("2026-03-02"), _d("2026-03-03"))


def test_invalid_intervals_are_rejected() -> None:
    with pytest.raises(ContractViolationError, match="end > start"):
        risk_free_per_interval(CDI_DATES, CDI_RATES, _d("2026-03-03"), _d("2026-03-03"))
    with pytest.raises(ContractViolationError, match="estritamente crescente"):
        risk_free_per_interval(CDI_DATES[::-1], CDI_RATES, _d("2026-03-03"), _d("2026-03-04"))
