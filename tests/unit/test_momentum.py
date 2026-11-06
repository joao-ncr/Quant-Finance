"""Carteira de momentum DE-01: regras temporais e financeiras.

O teste central compara o motor vetorizado com uma implementação de referência escrita
aqui, dia a dia, que aplica a regra literal de METHODOLOGY_CONVENTIONS.md §6.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from frente_e.core.cscv import cscv_pbo
from frente_e.core.momentum import (
    MAX_MONTHS,
    TRADING_DAYS_PER_MONTH,
    AssetPanel,
    MomentumConfig,
    MomentumGridResult,
    build_momentum_returns,
    momentum_family,
    month_end_indices,
    select_winners,
    valuation_returns,
)
from frente_e.core.null_models import business_day_calendar, simulate_factor_returns
from frente_e.errors import ContractViolationError, PendingDecisionError
from frente_e.experiments.cscv_input import trim_oldest_to_multiple
from frente_e.experiments.seeds import generator_for

SEED = 20261106
Floats = NDArray[np.float64]


def _prices(returns: Floats) -> Floats:
    """Índice de retorno total começando em 1: TR_0 = 1, TR_t = TR_{t−1}·(1 + r_t)."""
    tri = np.ones_like(returns)
    tri[1:] = np.cumprod(1.0 + returns[1:], axis=0)
    return tri


def _panel(returns: Floats, *, missing: NDArray[np.bool_] | None = None) -> AssetPanel:
    tri = _prices(returns)
    if missing is not None:
        tri = np.where(missing, np.nan, tri)
    n_obs, n_assets = tri.shape
    return AssetPanel(
        business_day_calendar("2000-01-03", n_obs),
        tri,
        np.ones((n_obs, n_assets), dtype=bool),
        tuple(f"a{i:02d}" for i in range(n_assets)),
    )


def _random_returns(stream: int, n_obs: int = 700, n_assets: int = 9) -> Floats:
    rng = generator_for(SEED, 40, stream)
    return simulate_factor_returns(
        n_obs,
        rng,
        mean=0.0002,
        market_vol=0.01,
        betas=rng.uniform(0.5, 1.5, n_assets),
        idiosyncratic_vols=rng.uniform(0.01, 0.03, n_assets),
        vol_persistence=0.0,
        vol_of_vol=0.0,
    )


def _by_date(result: MomentumGridResult) -> dict[str, float]:
    m = result.excess_returns
    return {str(d): float(v) for d, v in zip(m.index, m.values[:, 0], strict=True)}


# ------------------------------------------------------------- referência dia a dia
def _reference(panel: AssetPanel, rf: Floats, config: MomentumConfig) -> Floats:
    tri = panel.total_return_index
    n_obs, n_assets = tri.shape
    traded = np.isfinite(tri)
    last = np.full(n_assets, np.nan)
    value = np.full((n_obs, n_assets), np.nan)  # último preço conhecido até t
    for t in range(n_obs):
        last = np.where(traded[t], tri[t], last)
        value[t] = last
    months = [d.astype("datetime64[M]") for d in panel.dates]
    formations = [
        t for t in range(n_obs - 1) if months[t] != months[t + 1] and t >= 21 * MAX_MONTHS
    ]
    first_row = formations[MAX_MONTHS - 1] + 2
    lag = TRADING_DAYS_PER_MONTH * config.formation_months
    k = config.holding_months
    cohorts = []
    for m, f in enumerate(formations):
        ok = panel.eligible[f] & traded[f] & traded[f - lag]
        with np.errstate(invalid="ignore"):
            members, _ = select_winners(tri[f] / tri[f - lag] - 1.0, ok)
        execution = f + 1
        exit_row = formations[m + k] + 1 if m + k < len(formations) else n_obs - 1
        entries = {}
        for i in members.tolist():
            later = [t for t in range(execution, n_obs) if traded[t, i]]
            entries[i] = min(later[0], exit_row) if later else exit_row
        cohorts.append((execution, exit_row, members.tolist(), entries))
    out = []
    for t in range(first_row, n_obs):
        total = 0.0
        for execution, exit_row, members, entries in cohorts:
            if not execution < t <= exit_row:
                continue
            share = 1.0 / (k * len(members))
            for i in members:
                if entries[i] < t and entries[i] < exit_row:
                    total += share * (value[t, i] / value[t - 1, i] - 1.0)
                else:
                    total += share * rf[t]
        out.append(total - rf[t])
    return np.array(out)


# ------------------------------------------------------------------------------ grade
def test_families_are_nested_ordered_and_sized() -> None:
    f4, f16, f144 = (momentum_family(n) for n in ("F4", "F16", "F144"))
    assert (len(f4), len(f16), len(f144)) == (4, 16, 144)
    assert set(f4) < set(f16) < set(f144)
    assert list(f144) == sorted(f144)  # j crescente, depois k crescente
    assert f144[0].config_id == "j01_k01"
    assert f144[-1].config_id == "j12_k12"
    assert momentum_family("F144") == f144  # determinística
    with pytest.raises(ContractViolationError):
        momentum_family("F9")
    with pytest.raises(ContractViolationError):
        MomentumConfig(13, 1)


def test_nested_families_produce_identical_columns() -> None:
    panel = _panel(_random_returns(1))
    rf = np.full(panel.dates.size, 0.0002)
    big = build_momentum_returns(panel, rf, momentum_family("F16"))
    small = build_momentum_returns(panel, rf, momentum_family("F4"))
    np.testing.assert_array_equal(small.excess_returns.index, big.excess_returns.index)
    for j, cid in enumerate(small.excess_returns.config_ids):
        col = big.excess_returns.config_ids.index(cid)
        np.testing.assert_array_equal(
            small.excess_returns.values[:, j], big.excess_returns.values[:, col]
        )


# --------------------------------------------------------------------- calendário
def test_month_ends_exclude_the_last_session() -> None:
    dates = np.array(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28"], dtype="M8[D]")
    assert month_end_indices(dates).tolist() == [1]


# ------------------------------------------------------------------------- seleção
def test_tercile_size_order_and_boundary_tie() -> None:
    signal = np.array([0.1, 0.5, 0.3, 0.3, -0.2, 0.0, 0.2])
    members, tie = select_winners(signal, np.ones(7, dtype=bool))
    assert members.tolist() == [1, 2, 3]  # ⌈7/3⌉ = 3
    assert not tie
    tied = np.array([0.5, 0.3, 0.1, 0.3])
    members, tie = select_winners(tied, np.ones(4, dtype=bool))  # ⌈4/3⌉ = 2
    assert members.tolist() == [0, 1]  # empate 0,3: índice menor vence (DE-01.g)
    assert tie


def test_only_eligible_assets_are_ranked() -> None:
    signal = np.array([9.0, 0.1, 0.2, np.nan])
    members, _ = select_winners(signal, np.array([False, True, True, False]))
    assert members.tolist() == [2]


def test_no_eligible_asset_is_an_error() -> None:
    with pytest.raises(ContractViolationError, match="nenhum ativo elegível"):
        select_winners(np.zeros(3), np.zeros(3, dtype=bool))


# ------------------------------------------------------------- avaliação de posição
def test_stale_valuation_and_catch_up() -> None:
    tri = np.array([[1.0], [1.1], [np.nan], [np.nan], [1.21]])
    r = valuation_returns(tri)
    assert np.isnan(r[0, 0])
    np.testing.assert_allclose(r[1:, 0], [0.1, 0.0, 0.0, 0.1])


# ---------------------------------------------------------------- regra completa
@pytest.mark.parametrize(
    "config", [MomentumConfig(1, 1), MomentumConfig(3, 2), MomentumConfig(12, 12)], ids=str
)
def test_engine_matches_day_by_day_reference(config: MomentumConfig) -> None:
    panel = _panel(_random_returns(2))
    rf = np.full(panel.dates.size, 0.0002)
    result = build_momentum_returns(panel, rf, [config])
    np.testing.assert_allclose(
        result.excess_returns.values[:, 0], _reference(panel, rf, config), rtol=0, atol=1e-15
    )


def test_engine_matches_reference_with_missing_trades() -> None:
    """Dias sem negócio fora das execuções/saídas: afetam sinal (elegibilidade) e avaliação."""
    returns = _random_returns(3)
    dates = business_day_calendar("2000-01-03", returns.shape[0])
    months = dates.astype("datetime64[M]")
    after_month_end = np.zeros(dates.size, dtype=bool)
    after_month_end[1:] = months[1:] != months[:-1]
    rng = generator_for(SEED, 41)
    missing = rng.random(returns.shape) < 0.05
    # execução de uma coorte = saída de outra; falta de negócio aí é DE-01.h (teste próprio)
    missing[after_month_end] = False
    missing[0] = False
    panel = _panel(returns, missing=missing)
    rf = np.full(dates.size, 0.0002)
    configs = [MomentumConfig(2, 3), MomentumConfig(6, 1)]
    result = build_momentum_returns(panel, rf, configs)
    for col, config in enumerate(configs):
        np.testing.assert_allclose(
            result.excess_returns.values[:, col], _reference(panel, rf, config), atol=1e-15
        )
    assert result.diagnostics["j02_k03"].stale_position_days > 0


def test_deferred_entry_stays_in_cash_until_first_trade() -> None:
    n_obs, n_assets = 700, 3
    dates = business_day_calendar("2000-01-03", n_obs)
    usable = month_end_indices(dates)
    usable = usable[usable >= 252]
    f = int(usable[MAX_MONTHS])
    returns = np.zeros((n_obs, n_assets))
    returns[f - 3, 2] = 0.5  # ativo 2 lidera só em f (antes, empates em 0 → ativo 0)
    returns[f + 2, 2] = 0.2  # (f+1, f+2]: sem posição ainda (compra adiada)
    returns[f + 3, 2] = 0.1  # (f+2, f+3]: já comprado no fechamento de f+2
    missing = np.zeros((n_obs, n_assets), dtype=bool)
    missing[f + 1, 2] = True  # sem negócio no dia da execução
    panel = _panel(returns, missing=missing)
    rf = np.full(n_obs, 0.0001)
    result = build_momentum_returns(panel, rf, [MomentumConfig(1, 1)])
    series = _by_date(result)
    # (f+1, f+2]: fatia em caixa rende r_f -> excesso 0; o salto acumulado de f+2
    # (avaliado a partir do último preço, de f) não entra porque a compra é no fechamento de f+2
    assert series[str(dates[f + 2])] == pytest.approx(0.0, abs=1e-15)
    assert series[str(dates[f + 3])] == pytest.approx(0.1 - 0.0001, abs=1e-15)
    assert result.diagnostics["j01_k01"].deferred_entries == 1
    np.testing.assert_allclose(
        result.excess_returns.values[:, 0], _reference(panel, rf, MomentumConfig(1, 1)), atol=1e-15
    )


def test_formation_and_execution_are_distinct_days() -> None:
    """Um salto no dia da execução (t+1) não entra; um salto em t+2 entra com o peso certo."""
    n_obs, n_assets = 700, 3
    returns = np.zeros((n_obs, n_assets))
    dates = business_day_calendar("2000-01-03", n_obs)
    formations = month_end_indices(dates)
    usable = formations[formations >= 252]
    f = int(usable[MAX_MONTHS])  # primeira formação cuja coorte está no calendário comum
    returns[f - 5, 2] = 0.5  # ativo 2 lidera o sinal de 1 mês em f
    returns[f + 1, 2] = 0.3  # intervalo (f, f+1]: antes da execução no fechamento de f+1
    returns[f + 2, 2] = 0.2  # intervalo (f+1, f+2]: primeiro retorno da nova coorte
    panel = _panel(returns)
    result = build_momentum_returns(panel, np.zeros(n_obs), [MomentumConfig(1, 1)])
    series = _by_date(result)
    # ⌈3/3⌉ = 1 ativo; antes de f, empates em zero escolhem o ativo 0
    assert series[str(dates[f + 1])] == 0.0
    assert series[str(dates[f + 2])] == pytest.approx(0.2, abs=1e-15)


def test_overlapping_cohorts_split_capital_equally() -> None:
    """Com k = 2, duas coortes de 1/2; se só a mais nova contém o ativo, o peso é 1/2."""
    n_obs, n_assets = 700, 3
    returns = np.zeros((n_obs, n_assets))
    dates = business_day_calendar("2000-01-03", n_obs)
    usable = month_end_indices(dates)
    usable = usable[usable >= 252]
    f = int(usable[MAX_MONTHS])
    returns[f - 3, 1] = 0.4  # ativo 1 lidera apenas na formação f
    returns[f + 2, 1] = 0.1
    panel = _panel(returns)
    result = build_momentum_returns(panel, np.zeros(n_obs), [MomentumConfig(1, 2)])
    series = _by_date(result)
    assert series[str(dates[f + 2])] == pytest.approx(0.05, abs=1e-15)  # 1/2 × 1 ativo × 10%


def test_holding_period_controls_exit() -> None:
    """Coorte de k meses sai no fechamento do pregão seguinte ao k-ésimo fim de mês."""
    n_obs, n_assets = 700, 3
    dates = business_day_calendar("2000-01-03", n_obs)
    usable = month_end_indices(dates)
    usable = usable[usable >= 252]
    f = int(usable[MAX_MONTHS])
    for k in (1, 3):
        returns = np.zeros((n_obs, n_assets))
        returns[f - 3, 1] = 0.4  # ativo 1 escolhido só pela coorte formada em f
        exit_row = int(usable[MAX_MONTHS + k]) + 1
        returns[exit_row, 1] = 0.1  # último intervalo da coorte: ainda conta
        returns[exit_row + 1, 1] = 0.1  # depois da saída: não conta
        result = build_momentum_returns(_panel(returns), np.zeros(n_obs), [MomentumConfig(1, k)])
        series = _by_date(result)
        assert series[str(dates[exit_row])] == pytest.approx(0.1 / k, abs=1e-15)
        assert series[str(dates[exit_row + 1])] == 0.0


def test_formation_window_changes_the_selection() -> None:
    returns = _random_returns(4)
    panel = _panel(returns)
    rf = np.full(panel.dates.size, 0.0002)
    result = build_momentum_returns(panel, rf, [MomentumConfig(1, 1), MomentumConfig(12, 1)])
    a, b = result.excess_returns.values.T
    assert not np.array_equal(a, b)


# ------------------------------------------------------------------ casos de erro
def test_exit_without_trade_is_a_pending_decision() -> None:
    returns = _random_returns(5)
    dates = business_day_calendar("2000-01-03", returns.shape[0])
    months = dates.astype("datetime64[M]")
    exits = np.flatnonzero(months[1:] != months[:-1]) + 1
    missing = np.zeros(returns.shape, dtype=bool)
    missing[exits[-3], :] = True  # ninguém negocia num dia de saída
    with pytest.raises(PendingDecisionError, match=r"DE-01\.h"):
        build_momentum_returns(
            _panel(returns, missing=missing), np.zeros(dates.size), [MomentumConfig(1, 1)]
        )


def test_insufficient_history_is_rejected() -> None:
    with pytest.raises(ContractViolationError, match="dados insuficientes"):
        build_momentum_returns(
            _panel(_random_returns(6, n_obs=400)), np.zeros(400), [MomentumConfig(1, 1)]
        )


def test_universe_without_eligible_assets_is_rejected() -> None:
    panel = _panel(_random_returns(7))
    empty = AssetPanel(
        panel.dates,
        panel.total_return_index,
        np.zeros(panel.eligible.shape, dtype=np.bool_),
        panel.asset_ids,
    )
    with pytest.raises(ContractViolationError, match="nenhum ativo elegível"):
        build_momentum_returns(empty, np.zeros(panel.dates.size), [MomentumConfig(1, 1)])


def test_panel_contract() -> None:
    dates = business_day_calendar("2000-01-03", 3)
    good = np.ones((3, 2))
    with pytest.raises(ContractViolationError, match="positivos"):
        AssetPanel(dates, np.array([[1.0, 0.0]] * 3), np.ones((3, 2), bool), ("a", "b"))
    with pytest.raises(ContractViolationError, match="crescentes"):
        AssetPanel(dates[::-1], good, np.ones((3, 2), bool), ("a", "b"))
    with pytest.raises(ContractViolationError, match="únicos"):
        AssetPanel(dates, good, np.ones((3, 2), bool), ("a", "a"))
    not_bool: Any = np.ones((3, 2))
    with pytest.raises(ContractViolationError, match="bool"):
        AssetPanel(dates, good, not_bool, ("a", "b"))


def test_duplicate_configs_rejected() -> None:
    panel = _panel(_random_returns(8))
    with pytest.raises(ContractViolationError, match="repetições"):
        build_momentum_returns(
            panel, np.zeros(panel.dates.size), [MomentumConfig(1, 1), MomentumConfig(1, 1)]
        )


# ------------------------------------------------------------ sem sinal e saída
def test_weights_sum_to_one_so_excess_is_zero_when_all_assets_earn_rf() -> None:
    n_obs = 700
    returns = np.full((n_obs, 6), 0.0003)
    panel = _panel(returns)
    result = build_momentum_returns(panel, np.full(n_obs, 0.0003), momentum_family("F4"))
    np.testing.assert_allclose(result.excess_returns.values, 0.0, atol=1e-15)


def test_no_signal_gives_zero_mean_excess_across_replicates() -> None:
    """Sem sinal (média = r_f), o excesso médio é zero; o erro-padrão vem das réplicas."""
    means = []
    for rep in range(40):
        returns = _random_returns(100 + rep, n_obs=700, n_assets=9)
        panel = _panel(returns)
        result = build_momentum_returns(panel, np.full(700, 0.0002), [MomentumConfig(3, 3)])
        means.append(result.excess_returns.values.mean())
    arr = np.array(means)
    assert abs(arr.mean()) < 4 * arr.std(ddof=1) / math.sqrt(arr.size)


def test_deterministic_and_reproducible_by_seed() -> None:
    a = build_momentum_returns(_panel(_random_returns(9)), np.zeros(700), momentum_family("F4"))
    b = build_momentum_returns(_panel(_random_returns(9)), np.zeros(700), momentum_family("F4"))
    np.testing.assert_array_equal(a.excess_returns.values, b.excess_returns.values)


def test_output_feeds_cscv() -> None:
    panel = _panel(_random_returns(10, n_obs=900))
    result = build_momentum_returns(panel, np.zeros(900), momentum_family("F16"))
    trimmed, _ = trim_oldest_to_multiple(result.excess_returns, 4)
    pbo = cscv_pbo(trimmed, 4)
    assert pbo.n_partitions == 6
    assert 0.0 <= pbo.pbo <= 1.0
