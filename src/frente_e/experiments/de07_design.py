"""Cenários de DE-07: painel com sinal, fluxos aleatórios e execução paralela determinística.

Compartilhado pela calibração do sinal (``de07_calibration``) e pelo executor
confirmatório (``de07_confirmatory``). Este módulo **não** importa DSR, PSR, PBO, CSCV ou
estimadores de N efetivo: a separação entre gerar dados e avaliar métodos é testada.

Fluxos (``generator_for``): para uma réplica com caminho ``path``,

- ``(*path, STREAM_NOISE)``: fator, choques idiossincráticos e volatilidade, exatamente
  como nos cenários nulos de ``momentum_scenarios``;
- ``(*path, STREAM_SIGNAL)``: só as médias específicas μ (D4).

Com sinal nulo, o fluxo de sinal não é usado e o painel é o cenário nulo original, bit a
bit. Com o mesmo ruído, ligar o sinal muda só o termo de média (números aleatórios
comuns entre intensidades, usados na calibração).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.core.momentum import (
    AssetPanel,
    MomentumGridResult,
    build_momentum_returns,
    momentum_family,
)
from frente_e.core.signal_models import mean_shift
from frente_e.errors import ContractViolationError
from frente_e.experiments import momentum_scenarios
from frente_e.experiments.seeds import generator_for
from frente_e.runtime import native_thread_env

FloatArray = NDArray[np.float64]
STREAM_NOISE = 0
STREAM_SIGNAL = 1

SIGNAL_KEYS = frozenset({"half_life", "scale", "premium"})


def validate_signal(signal: Mapping[str, Any] | None) -> None:
    """``None`` (nulo) ou {half_life > 0, scale ≥ 0, premium finito}."""
    if signal is None:
        return
    if set(signal) != SIGNAL_KEYS:
        raise ContractViolationError(f"sinal deve ter exatamente as chaves {sorted(SIGNAL_KEYS)}")
    values = [signal[k] for k in sorted(SIGNAL_KEYS)]
    if not all(isinstance(v, float) and np.isfinite(v) for v in values):
        raise ContractViolationError("parâmetros do sinal devem ser float finitos")
    if signal["half_life"] <= 0.0 or signal["scale"] < 0.0:
        raise ContractViolationError("half_life > 0 e scale >= 0")


def signal_panel(
    scenario: Mapping[str, Any],
    common: Mapping[str, Any],
    signal: Mapping[str, Any] | None,
    root_seed: int,
    path: Sequence[int],
) -> tuple[AssetPanel, FloatArray]:
    """Painel de uma réplica: ruído do cenário + (opcional) π + μ_i,t."""
    validate_signal(signal)
    noise = generator_for(root_seed, *path, STREAM_NOISE)
    shift = None
    if signal is not None:
        shift = mean_shift(
            int(common["n_obs"]),
            int(scenario["n_assets"]),
            half_life=float(signal["half_life"]),
            scale=float(signal["scale"]),
            premium=float(signal["premium"]),
            rng=generator_for(root_seed, *path, STREAM_SIGNAL),
        )
    return momentum_scenarios.scenario_panel(scenario, common, noise, shift)


def signal_grid(
    scenario: Mapping[str, Any],
    common: Mapping[str, Any],
    signal: Mapping[str, Any] | None,
    root_seed: int,
    path: Sequence[int],
    family: str,
) -> MomentumGridResult:
    """Retornos excedentes da família pela regra DE-01 completa (``build_momentum_returns``)."""
    panel, rf = signal_panel(scenario, common, signal, root_seed, path)
    return build_momentum_returns(panel, rf, momentum_family(family))


def equal_weight_volatility(scenario: Mapping[str, Any]) -> float:
    """Desvio diário exato do excesso da carteira igualmente ponderada, sem sinal e com
    volatilidade constante: ``√(β̄²σ_m² + Σ s_i²/n²)``. Usado para fixar π (H₁-com)."""
    if float(scenario["vol_of_vol"]) != 0.0:
        raise ContractViolationError("fórmula exata só vale com volatilidade constante")
    betas, vols = momentum_scenarios.scenario_parameters(scenario)
    n = betas.size
    market = float(scenario["market_vol"])
    return float(np.sqrt((betas.mean() * market) ** 2 + np.sum(vols**2) / n**2))


# ------------------------------------------------------------------ execução paralela
@contextmanager
def _native_threads(n_threads: int) -> Iterator[None]:
    """Fixa threads nativas no ambiente herdado pelos processos filhos e restaura depois."""
    updates = native_thread_env(n_threads)
    previous = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def map_ordered[T, R](
    fn: Callable[[T], R], items: Sequence[T], *, n_workers: int, native_threads: int = 1
) -> list[R]:
    """``[fn(x) for x in items]``, opcionalmente em processos. Cada item é determinístico
    (fluxo próprio por réplica), então o resultado não depende de ``n_workers``."""
    if n_workers < 1:
        raise ContractViolationError("n_workers deve ser >= 1")
    if n_workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (4 * n_workers))
    with _native_threads(native_threads), ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, items, chunksize=chunk))
