"""Médias específicas persistentes para o desenho não nulo de DE-07 (protocolo §7.1, D4).

Modelo (docs/DE07_PROTOCOL.md §7.1)::

    r_i,t = r_f + π + μ_i,t + (termos de fator e idiossincrático sem sinal)
    μ_i,t = a_i,t − (1/n)·Σ_j a_j,t                     (média transversal zero em todo t)
    a_i,t = φ·a_i,t−1 + √(1 − φ²)·s_a·ξ_i,t,   φ = 2^(−1/h),   a_i,0 ~ N(0, s_a²)

- ``a`` é estacionário desde t = 0 (sorteio inicial na distribuição estacionária), com
  desvio ``s_a`` e meia-vida ``h`` pregões: a autocorrelação no lag h é 1/2.
- Como Σ_i μ_i,t = 0 em todo t, a carteira igualmente ponderada não recebe nenhum
  retorno esperado do sinal: o sinal é puramente transversal (momentum), separado do
  prêmio comum π (célula H₁-com).
- E[μ_i,t | passado] = φ·μ_i,t−1: retornos passados altos indicam médias altas que
  persistem, e é por isso que a regra de momentum pode lucrar. A regra não recebe μ;
  o sinal só chega a ela pelos preços.

Nada aqui conhece a regra DE-01, o DSR, a PSR ou a PBO. Puro: recebe o ``Generator``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from frente_e.errors import ContractViolationError

FloatArray = NDArray[np.float64]


def ar1_coefficient(half_life: float) -> float:
    """φ = 2^(−1/h): o efeito de um choque cai à metade em ``half_life`` pregões."""
    if not (math.isfinite(half_life) and half_life > 0.0):
        raise ContractViolationError("half_life deve ser real positivo (em pregões)")
    return float(2.0 ** (-1.0 / half_life))


def persistent_specific_means(
    n_obs: int, n_assets: int, *, half_life: float, scale: float, rng: np.random.Generator
) -> FloatArray:
    """Matriz T × n de médias específicas μ_i,t, com média transversal exatamente zero.

    ``scale`` é o desvio estacionário s_a de a_i,t (retorno diário). Com ``scale = 0`` o
    resultado é zero e nenhum número aleatório é consumido.
    """
    if isinstance(n_obs, bool) or n_obs < 1:
        raise ContractViolationError("n_obs deve ser >= 1")
    if isinstance(n_assets, bool) or n_assets < 2:
        raise ContractViolationError("n_assets deve ser >= 2 (média transversal)")
    if not (math.isfinite(scale) and scale >= 0.0):
        raise ContractViolationError("scale deve ser real >= 0")
    phi = ar1_coefficient(half_life)
    if scale == 0.0:
        return np.zeros((n_obs, n_assets), dtype=np.float64)
    shocks = rng.standard_normal((n_obs, n_assets))
    states = np.empty((n_obs, n_assets), dtype=np.float64)
    states[0] = scale * shocks[0]
    if n_obs > 1:
        innovation = scale * math.sqrt(1.0 - phi * phi)
        filtered, _ = scipy_signal.lfilter(
            [innovation], [1.0, -phi], shocks[1:], axis=0, zi=(phi * states[0])[None, :]
        )
        states[1:] = filtered
    means: FloatArray = states - states.mean(axis=1, keepdims=True)
    return means


def mean_shift(
    n_obs: int,
    n_assets: int,
    *,
    half_life: float,
    scale: float,
    premium: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Deslocamento da média dos retornos: ``π + μ_i,t`` (T × n).

    ``premium`` (π) é o prêmio comum diário sobre r_f, igual para todos os ativos e
    constante; ``scale``/``half_life`` definem μ. π = 0 e scale = 0 dão zero.
    """
    if not math.isfinite(premium):
        raise ContractViolationError("premium deve ser finito")
    shift = persistent_specific_means(n_obs, n_assets, half_life=half_life, scale=scale, rng=rng)
    out: FloatArray = shift + premium
    return out
