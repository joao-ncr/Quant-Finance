"""Carteira de momentum DE-01 (docs/METHODOLOGY_CONVENTIONS.md §6), núcleo puro.

Regra implementada (cada item remete à seção de DE-01):

- **Formação (§6.2):** último pregão de cada mês do calendário de entrada. O último pregão
  da série nunca é data de formação, porque não haveria pregão seguinte para executar.
- **Sinal:** ``s_i(t) = TR_i(t)/TR_i(t − 21·j) − 1``. Elegível: no universo em t e com
  negócio em t e em t − 21·j (sem preço defasado no sinal).
- **Seleção (§6.4):** tercil superior, ``⌈n_elegíveis/3⌉`` ativos. Empate na fronteira
  resolvido pela ordem canônica **crescente** do identificador (DE-01.g) e contado.
- **Execução:** fechamento de t + 1. A compra de ativo sem negócio nesse dia fica em caixa
  até o primeiro negócio. A saída é no fechamento do pregão seguinte ao k-ésimo fim de mês
  após t. Saída adiada por falta de negócio depende de DE-01.h e gera
  :class:`PendingDecisionError`.
- **Sobreposição:** k coortes, cada uma com 1/k do capital e pesos iguais dentro da coorte,
  constantes entre execuções (rebalanceamento diário implícito).
- **Retorno (§6.5):** ``r_p,t = Σ w_i,t·r_i,t + w_caixa,t·r_f,t``; excesso ``r_p,t − r_f,t``.
  Ativo mantido sem negócio vale o último preço (retorno 0 no dia); o retorno acumulado
  entra no próximo negócio.
- **Calendário comum:** a matriz começa quando todas as configurações com j, k ≤ 12 têm
  k coortes completas. A primeira formação usada é o primeiro fim de mês com 21·12 pregões
  de histórico, e a saída começa no retorno seguinte à execução da 12ª formação. Assim F4
  e F16 são subconjuntos exatos das colunas de F144.

Convenção de linhas: a linha t de retornos é o intervalo (t − 1, t].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from frente_e.core.contracts import ReturnsMatrix
from frente_e.errors import ContractViolationError, PendingDecisionError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

TRADING_DAYS_PER_MONTH = 21
MAX_MONTHS = 12  # j, k ∈ {1, …, 12} (§6.3)
WEIGHT_TOLERANCE = 1e-12  # soma dos pesos = 1 a menos de arredondamento (k ≤ 12 parcelas)


@dataclass(frozen=True, order=True)
class MomentumConfig:
    formation_months: int  # j
    holding_months: int  # k

    def __post_init__(self) -> None:
        for name, value in (("j", self.formation_months), ("k", self.holding_months)):
            if isinstance(value, bool) or not 1 <= value <= MAX_MONTHS:
                raise ContractViolationError(f"{name} deve estar em 1..{MAX_MONTHS}")

    @property
    def config_id(self) -> str:
        return f"j{self.formation_months:02d}_k{self.holding_months:02d}"


def momentum_family(name: str) -> tuple[MomentumConfig, ...]:
    """F4, F16 ou F144 (§6.3), em ordem estável: j crescente, depois k crescente."""
    months = {"F4": (6, 12), "F16": (3, 6, 9, 12), "F144": tuple(range(1, MAX_MONTHS + 1))}
    if name not in months:
        raise ContractViolationError(f"família desconhecida: {name!r}")
    return tuple(MomentumConfig(j, k) for j in months[name] for k in months[name])


@dataclass(frozen=True, eq=False)
class AssetPanel:
    """Entrada da carteira: preços de retorno total por ativo, com ausências explícitas.

    ``total_return_index[t, i]`` é NaN quando não houve negócio em t (DD-05). ``eligible``
    é a pertinência ao universo em t, ponto no tempo, decidida pela camada de dados.
    """

    dates: NDArray[np.datetime64]
    total_return_index: FloatArray
    eligible: BoolArray
    asset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        dates = np.array(self.dates, dtype="datetime64[D]", copy=True)
        tri = np.array(self.total_return_index, copy=True)
        eligible = np.array(self.eligible, copy=True)
        if tri.dtype != np.float64 or tri.ndim != 2:
            raise ContractViolationError("total_return_index deve ser float64 T × A")
        n_obs, n_assets = tri.shape
        if dates.shape != (n_obs,) or not bool(np.all(dates[1:] > dates[:-1])):
            raise ContractViolationError("dates deve ter T datas estritamente crescentes")
        if eligible.dtype != np.bool_ or eligible.shape != tri.shape:
            raise ContractViolationError("eligible deve ser bool com o shape dos preços")
        observed = tri[np.isfinite(tri)]
        if np.any(np.isinf(tri)) or np.any(observed <= 0.0):
            raise ContractViolationError("preços de retorno total devem ser positivos ou NaN")
        ids = tuple(self.asset_ids)
        if len(ids) != n_assets or len(set(ids)) != n_assets or not all(ids):
            raise ContractViolationError("asset_ids devem ser únicos, não vazios, um por coluna")
        for arr in (dates, tri, eligible):
            arr.flags.writeable = False
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "total_return_index", tri)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "asset_ids", ids)

    @property
    def traded(self) -> BoolArray:
        return np.isfinite(self.total_return_index)


@dataclass(frozen=True)
class ConfigDiagnostics:
    boundary_ties: int  # formações com empate na fronteira do tercil (DE-01.g)
    deferred_entries: int  # compras adiadas por falta de negócio no dia da execução
    stale_position_days: int  # (dia, ativo) mantidos sem negócio, avaliados pelo último preço


@dataclass(frozen=True, eq=False)
class MomentumGridResult:
    excess_returns: ReturnsMatrix  # T_saída × N, colunas na ordem das configurações
    formation_dates: NDArray[np.datetime64]  # formações usadas
    diagnostics: dict[str, ConfigDiagnostics] = field(default_factory=dict)


def month_end_indices(dates: NDArray[np.datetime64]) -> IntArray:
    """Último pregão de cada mês, exceto o último pregão da série (sem execução possível)."""
    months = np.asarray(dates, dtype="datetime64[M]")
    return np.flatnonzero(months[1:] != months[:-1]).astype(np.int64)


def valuation_returns(total_return_index: FloatArray) -> FloatArray:
    """Retorno diário para avaliar posições: último preço em dia sem negócio (retorno 0) e
    retorno acumulado no negócio seguinte. NaN antes do primeiro negócio do ativo."""
    filled = _forward_fill(total_return_index)
    out = np.full_like(filled, np.nan)
    out[1:] = filled[1:] / filled[:-1] - 1.0
    return out


def select_winners(signal: FloatArray, eligible: BoolArray) -> tuple[IntArray, bool]:
    """Tercil superior dos elegíveis por sinal decrescente; empate → menor índice (DE-01.g).

    Devolve (índices selecionados em ordem de seleção, houve empate na fronteira).
    """
    candidates = np.flatnonzero(eligible)
    if candidates.size == 0:
        raise ContractViolationError(
            "nenhum ativo elegível na data de formação: o universo deve garantir elegíveis"
        )
    values = signal[candidates]
    if not np.all(np.isfinite(values)):
        raise ContractViolationError("sinal não finito para ativo elegível")
    order = np.lexsort((candidates, -values))  # chave primária: −sinal; depois índice
    n_selected = math.ceil(candidates.size / 3)
    tie = bool(
        n_selected < candidates.size and values[order[n_selected - 1]] == values[order[n_selected]]
    )
    return candidates[order[:n_selected]].astype(np.int64), tie


def build_momentum_returns(
    panel: AssetPanel, risk_free: FloatArray, configs: Sequence[MomentumConfig]
) -> MomentumGridResult:
    """Retornos excedentes diários das configurações, no calendário comum de DE-01."""
    if not configs or len(set(configs)) != len(configs):
        raise ContractViolationError("configs deve ser não vazia e sem repetições")
    rf = np.asarray(risk_free)
    n_obs, n_assets = panel.total_return_index.shape
    if rf.dtype != np.float64 or rf.shape != (n_obs,) or not np.all(np.isfinite(rf[1:])):
        raise ContractViolationError("risk_free deve ser float64 (T,), finito a partir de t=1")

    traded = panel.traded
    tri = panel.total_return_index
    asset_returns = valuation_returns(tri)
    next_trade = _next_trade_index(traded)
    formations = month_end_indices(panel.dates)
    history = TRADING_DAYS_PER_MONTH * MAX_MONTHS
    usable = formations[formations >= history]
    if usable.size < MAX_MONTHS + 1:
        raise ContractViolationError(
            "dados insuficientes: são necessários 21·12 pregões de histórico e ao menos "
            f"{MAX_MONTHS + 1} fins de mês utilizáveis; há {usable.size}"
        )
    first_row = int(usable[MAX_MONTHS - 1]) + 2  # retorno após a execução da 12ª formação
    if first_row >= n_obs:
        raise ContractViolationError("dados insuficientes após o aquecimento")

    selections: dict[int, list[IntArray]] = {}
    ties: dict[int, int] = {}
    for j in sorted({c.formation_months for c in configs}):
        lag = TRADING_DAYS_PER_MONTH * j
        chosen, n_ties = [], 0
        for f in usable:
            ok = panel.eligible[f] & traded[f] & traded[f - lag]
            with np.errstate(invalid="ignore", divide="ignore"):
                signal = tri[f] / tri[f - lag] - 1.0
            members, tie = select_winners(signal, ok)
            chosen.append(members)
            n_ties += int(tie)
        selections[j], ties[j] = chosen, n_ties

    columns: list[FloatArray] = []
    diagnostics: dict[str, ConfigDiagnostics] = {}
    for config in configs:
        weights = np.zeros((n_obs, n_assets), dtype=np.float64)
        cash = np.zeros(n_obs, dtype=np.float64)
        k = config.holding_months
        deferred = 0
        for m, f in enumerate(usable):
            execution = int(f) + 1
            exit_row = int(usable[m + k]) + 1 if m + k < usable.size else n_obs - 1
            if exit_row <= first_row - 1:
                continue  # coorte encerrada antes do calendário comum
            members = selections[config.formation_months][m]
            share = 1.0 / (k * members.size)
            # compra no primeiro negócio a partir da execução; sem negócio até a saída,
            # a fatia fica em caixa a vida toda da coorte
            entries = np.minimum(next_trade[execution, members], exit_row)
            entered = entries < exit_row
            if m + k < usable.size and np.any(entered & ~traded[exit_row, members]):
                stuck = members[entered & ~traded[exit_row, members]]
                raise PendingDecisionError(
                    "DE-01.h",
                    f"saída de {[panel.asset_ids[i] for i in stuck]} adiada por falta de "
                    f"negócio em {panel.dates[exit_row]}; regra de capital com coortes "
                    "sobrepostas não formalizada",
                )
            on_time = members[entries == execution]
            weights[execution + 1 : exit_row + 1, on_time] += share
            for i, entry in zip(members.tolist(), entries.tolist(), strict=True):
                if entry == execution:
                    continue
                deferred += 1
                cash[execution + 1 : entry + 1] += share
                weights[entry + 1 : exit_row + 1, i] += share
        rows = slice(first_row, n_obs)
        total = weights[rows].sum(axis=1) + cash[rows]
        if not np.allclose(total, 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE):
            raise ContractViolationError(
                f"{config.config_id}: pesos não somam 1 no calendário comum (erro interno)"
            )
        held = weights[rows] > 0.0
        if np.any(np.isnan(asset_returns[rows][held])):
            raise ContractViolationError(f"{config.config_id}: posição sem preço de avaliação")
        gross = np.where(held, weights[rows] * np.nan_to_num(asset_returns[rows]), 0.0).sum(1)
        portfolio = gross + cash[rows] * rf[rows]
        columns.append(portfolio - rf[rows])
        diagnostics[config.config_id] = ConfigDiagnostics(
            boundary_ties=ties[config.formation_months],
            deferred_entries=deferred,
            stale_position_days=int(np.count_nonzero(held & ~traded[rows])),
        )

    matrix = ReturnsMatrix(
        np.column_stack(columns),
        panel.dates[first_row:],
        tuple(c.config_id for c in configs),
    )
    return MomentumGridResult(matrix, panel.dates[usable], diagnostics)


def _forward_fill(values: FloatArray) -> FloatArray:
    filled = np.array(values, copy=True)
    valid = np.isfinite(filled)
    index = np.where(valid, np.arange(filled.shape[0])[:, None], 0)
    np.maximum.accumulate(index, axis=0, out=index)
    out = filled[index, np.arange(filled.shape[1])]
    started = np.maximum.accumulate(valid, axis=0)
    out[~started] = np.nan
    return out


def _next_trade_index(traded: BoolArray) -> IntArray:
    """[t, i] = primeiro pregão ≥ t com negócio do ativo i; T (fora da série) se não houver."""
    n_obs = traded.shape[0]
    rows = np.where(traded, np.arange(n_obs)[:, None], n_obs)
    out: IntArray = np.minimum.accumulate(rows[::-1], axis=0)[::-1].astype(np.int64)
    return out
