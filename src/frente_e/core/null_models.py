"""Modelos nulos para avaliar o máximo esperado dos Sharpe sob H₀ (DE-03/DE-04).

Escopo restrito às perguntas registradas em docs/METHODOLOGY_CONVENTIONS.md §4:
precisão da aproximação de E[máx] e estimadores de N efetivo sob correlação.

Hipótese de modelagem (registrada, não decidida para o estudo real): sob a nula global
(SR verdadeiro = 0 em todas as tentativas) e T grande, o vetor de ŜR é aproximadamente
normal com média 0 e covariância C/(T − 1), em que C é a correlação entre os retornos
das tentativas. Por isso basta simular Z ~ N(0, C) e escalar. Autocorrelação temporal
(DE-06) e caudas pesadas estão fora deste modelo.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, special

from frente_e.core.effective_trials import validate_correlation
from frente_e.errors import ContractViolationError

FloatArray = NDArray[np.float64]


# ------------------------------------------------------------------ estruturas de correlação
def equicorrelation(m: int, rho: float) -> FloatArray:
    """Todas as correlações fora da diagonal iguais a ``rho`` (definida positiva se
    −1/(M−1) < ρ < 1)."""
    _require(m >= 2, "m deve ser >= 2")
    _require(-1.0 / (m - 1) < rho < 1.0, "rho fora do intervalo que torna C definida positiva")
    c: FloatArray = np.full((m, m), rho, dtype=np.float64)
    np.fill_diagonal(c, 1.0)
    return c


def block_correlation(n_blocks: int, block_size: int, rho_within: float) -> FloatArray:
    """Blocos independentes; dentro de cada bloco, correlação constante ``rho_within``."""
    _require(n_blocks >= 1 and block_size >= 2, "exige n_blocks >= 1 e block_size >= 2")
    return np.asarray(
        np.kron(np.eye(n_blocks), equicorrelation(block_size, rho_within)), dtype=np.float64
    )


def ar1_correlation(m: int, rho: float) -> FloatArray:
    """``ρ^|i−j|``: vizinhos numa grade unidimensional são mais correlacionados."""
    _require(m >= 2, "m deve ser >= 2")
    _require(0.0 <= rho < 1.0, "rho deve estar em [0, 1)")
    idx = np.arange(m)
    c: FloatArray = rho ** np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    return c


def separable_grid_correlation(
    n_rows: int, n_cols: int, rho_rows: float, rho_cols: float
) -> FloatArray:
    """Grade bidimensional (ex.: formação × manutenção): ``ρ_r^|Δi| · ρ_c^|Δj|``."""
    grid = np.kron(ar1_correlation(n_rows, rho_rows), ar1_correlation(n_cols, rho_cols))
    return np.asarray(grid, dtype=np.float64)


# ---------------------------------------------------------------------------- amostragem
def sample_standardized_null(
    correlation: FloatArray, n_reps: int, rng: np.random.Generator, *, chunk: int
) -> tuple[FloatArray, FloatArray]:
    """Para Z ~ N(0, C), devolve por réplica ``max_i Z_i`` e a variância entre tentativas
    ``s²(Z)`` (ddof = 1), processando ``chunk`` réplicas por vez.

    Os normais sorteados não dependem de ``chunk``, mas o produto matricial (BLAS) pode
    diferir em 1 ulp conforme o formato do bloco. Para reprodutibilidade bit a bit,
    ``chunk`` faz parte dos parâmetros registrados."""
    c = validate_correlation(correlation)
    _require(n_reps >= 1 and chunk >= 1, "n_reps e chunk devem ser >= 1")
    try:
        lower = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ContractViolationError("correlação não é definida positiva") from exc
    m = c.shape[0]
    maxima = np.empty(n_reps, dtype=np.float64)
    variances = np.empty(n_reps, dtype=np.float64)
    for start in range(0, n_reps, chunk):
        stop = min(start + chunk, n_reps)
        z = rng.standard_normal((stop - start, m)) @ lower.T
        maxima[start:stop] = z.max(axis=1)
        variances[start:stop] = z.var(axis=1, ddof=1)
    return maxima, variances


def sample_correlation_estimate(
    correlation: FloatArray, n_obs: int, rng: np.random.Generator
) -> FloatArray:
    """Correlação amostral de T observações normais com correlação ``C`` (ruído de Ĉ)."""
    c = validate_correlation(correlation)
    _require(n_obs > c.shape[0], "n_obs deve exceder M para Ĉ ser não singular")
    returns = rng.standard_normal((n_obs, c.shape[0])) @ np.linalg.cholesky(c).T
    return np.asarray(np.corrcoef(returns, rowvar=False), dtype=np.float64)


# ------------------------------------------------------------------------ valor exato i.i.d.
QUADRATURE_MAX_ABS_ERROR = 1e-6
"""Erro absoluto máximo aceito na integral. Os erros de aproximação que se quer medir são
de 0,4% a 8% de valores entre 0,5 e 5, ou seja, ≥ 2e-3: quatro ordens de grandeza acima."""


def expected_max_iid_normal(n: float) -> float:
    """E[máx] exato de N normais padrão i.i.d., com F = Φᴺ::

        E[máx] = ∫₀^∞ (1 − Φ(x)ᴺ) dx − ∫₋∞⁰ Φ(x)ᴺ dx

    Definido para N real ≥ 1 (Φᴺ é uma função de distribuição). Serve só para medir o erro
    da aproximação usada no DSR. Erro numérico acima de ``QUADRATURE_MAX_ABS_ERROR`` é erro.
    """
    _require(math.isfinite(n) and n >= 1.0, "n deve ser real finito >= 1")

    def upper_tail(x: float) -> float:
        return -math.expm1(n * float(special.log_ndtr(x)))

    def lower_cdf(x: float) -> float:
        return math.exp(n * float(special.log_ndtr(x)))

    upper, err_upper = integrate.quad(upper_tail, 0.0, math.inf, epsabs=1e-13, limit=400)
    lower, err_lower = integrate.quad(lower_cdf, -math.inf, 0.0, epsabs=1e-13, limit=400)
    if err_upper + err_lower > QUADRATURE_MAX_ABS_ERROR:
        raise ContractViolationError(
            f"integração imprecisa (erro estimado {err_upper + err_lower:.2e})"
        )
    return float(upper - lower)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolationError(message)


# ------------------------------------------- calibração analítica do DSR (DE-07, protocolo v1)
def null_cross_variance_factor(correlation: FloatArray) -> float:
    """``E[s²(Z)]`` para Z ~ N(0, C), com s² a variância entre tentativas (ddof = 1).

    Identidade exata para qualquer correlação C (M × M)::

        E[s²(Z)] = (tr C − 1ᵀC1/M)/(M − 1) = 1 − ρ̄,

    com ρ̄ a média das correlações fora da diagonal. Sob a nula global com variância
    marginal σ² igual para todas as tentativas, a variância entre tentativas dos ŜR tem
    esperança σ²·(1 − ρ̄): a correlação já entra em V (docs/DE07_PROTOCOL.md §2.3).
    """
    c = validate_correlation(correlation)
    m = c.shape[0]
    return float((np.trace(c) - np.sum(c) / m) / (m - 1))


def prob_max_exceeds_equicorrelated(m: int, rho: float, threshold: float) -> float:
    """``P(máxᵢ Zᵢ > c)`` para Z ~ N(0, C) equicorrelacionado (M ≥ 1, 0 ≤ ρ < 1).

    Com ``Zᵢ = √ρ·W + √(1−ρ)·εᵢ``, condicionando em W::

        P(máx Z > c) = ∫ φ(w)·[1 − Φ((c − √ρ·w)/√(1−ρ))ᴹ] dw

    calculado pelo complemento ``−expm1(M·log Φ)`` para preservar probabilidades pequenas.
    Em ρ = 0 vale a forma fechada ``1 − Φ(c)ᴹ``. Determinístico (quadratura), sem RNG.
    """
    _require(m >= 1, "m deve ser >= 1")
    _require(0.0 <= rho < 1.0, "rho deve estar em [0, 1)")
    _require(math.isfinite(threshold), "threshold deve ser finito")
    if rho == 0.0:
        return float(-math.expm1(m * float(special.log_ndtr(threshold))))
    a, b = math.sqrt(rho), math.sqrt(1.0 - rho)

    def integrand(w: float) -> float:
        tail = -math.expm1(m * float(special.log_ndtr((threshold - a * w) / b)))
        return math.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi) * tail

    value, err = integrate.quad(integrand, -math.inf, math.inf, epsabs=1e-13, limit=400)
    if err > QUADRATURE_MAX_ABS_ERROR:
        raise ContractViolationError(f"integração imprecisa (erro estimado {err:.2e})")
    return float(min(max(value, 0.0), 1.0))


def dsr_null_rejection_rate_equicorrelated(
    m: int, rho: float, n_obs: int, confidence: float, benchmark_z: float
) -> float:
    """``P(DSR > confidence)`` sob a nula global gaussiana, tentativas equicorrelacionadas.

    Modelo (o mesmo de §4.4 de METHODOLOGY_CONVENTIONS): retornos normais (γ₃ = 0,
    γ₄ = 3), ŜRᵢ = σ·Zᵢ com σ = 1/√(T − 1) e Z ~ N(0, C). ``benchmark_z`` é SR₀/σ, o
    benchmark do DSR em unidades de σ, tratado como **fixo** (ex.: f(N) com V = V_H0, ou
    √(1 − ρ)·f(M) com V igual à esperança da variância entre tentativas).

    O DSR rejeita quando ``(x − SR₀)·√(T−1)/√(1 + x²/2) > q``, com x = máx ŜR e
    q = Φ⁻¹(confidence). Para x > SR₀ o lado esquerdo é crescente em x, então o limiar
    x* é a maior raiz de ``x²·((T−1) − q²/2) − 2·SR₀·(T−1)·x + SR₀²·(T−1) − q² = 0``.
    """
    _require(n_obs >= 3, "n_obs deve ser >= 3")
    _require(0.5 < confidence < 1.0, "confidence deve estar em (0,5; 1)")
    _require(math.isfinite(benchmark_z) and benchmark_z >= 0.0, "benchmark_z deve ser >= 0")
    q = float(special.ndtri(confidence))
    t1 = float(n_obs - 1)
    sr0 = benchmark_z / math.sqrt(t1)
    a = t1 - q * q / 2.0
    _require(a > 0.0, "T pequeno demais para o limiar (T − 1 ≤ q²/2)")
    b = -2.0 * sr0 * t1
    c = sr0 * sr0 * t1 - q * q
    x_star = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return prob_max_exceeds_equicorrelated(m, rho, x_star * math.sqrt(t1))


# ------------------------------------------------------------- painel de ativos sem sinal
def simulate_factor_returns(
    n_obs: int,
    rng: np.random.Generator,
    *,
    mean: float,
    market_vol: float,
    betas: FloatArray,
    idiosyncratic_vols: FloatArray,
    vol_persistence: float,
    vol_of_vol: float,
) -> FloatArray:
    """Retornos simples diários ``r_i,t = mean + β_i·σ_t·u_t + s_i·σ_t/σ̄·ε_i,t``, sem sinal.

    - ``u``, ``ε`` normais padrão independentes no tempo e entre ativos: **nenhum retorno
      passado prevê o futuro** (média constante), então sob qualquer regra que use só o
      passado o retorno esperado em excesso de ``mean`` é zero.
    - Dependência cross-sectional: fator de mercado comum (β_i) e volatilidades
      idiossincráticas heterogêneas (s_i).
    - Dependência temporal **só na variância**: ``log σ_t = log market_vol + h_t``,
      ``h_t = φ·h_{t−1} + η_t``, com η ~ N(0, vol_of_vol²); o mesmo regime escala fator e
      ruído (σ̄ = market_vol). Com ``vol_of_vol = 0``, a variância é constante.

    Erro se algum retorno ficar ≤ −1 (preço não positivo): o cenário seria inválido.
    """
    b = np.asarray(betas, dtype=np.float64)
    s = np.asarray(idiosyncratic_vols, dtype=np.float64)
    _require(b.ndim == 1 and b.shape == s.shape and b.size >= 1, "betas e vols: mesmo shape 1-D")
    _require(bool(np.all(s >= 0.0)) and market_vol > 0.0, "volatilidades devem ser positivas")
    _require(0.0 <= vol_persistence < 1.0 and vol_of_vol >= 0.0, "parâmetros de volatilidade")
    _require(n_obs >= 2, "n_obs deve ser >= 2")
    shocks = rng.standard_normal(n_obs) * vol_of_vol
    h = np.empty(n_obs, dtype=np.float64)
    h[0] = shocks[0] / math.sqrt(1.0 - vol_persistence**2) if vol_of_vol > 0 else 0.0
    for t in range(1, n_obs):
        h[t] = vol_persistence * h[t - 1] + shocks[t]
    regime = np.exp(h)
    market = rng.standard_normal(n_obs) * market_vol * regime
    idio = rng.standard_normal((n_obs, b.size)) * s * regime[:, None]
    returns = mean + market[:, None] * b + idio
    if np.any(returns <= -1.0):
        raise ContractViolationError("cenário gerou retorno <= -100%: parâmetros inválidos")
    out: FloatArray = returns
    return out


def business_day_calendar(start: str, n_obs: int) -> NDArray[np.datetime64]:
    """Pregões sintéticos: dias úteis (seg–sex) a partir de ``start``; sem feriados."""
    _require(n_obs >= 1, "n_obs deve ser >= 1")
    days = np.busday_offset(np.datetime64(start, "D"), np.arange(n_obs), roll="forward")
    return np.asarray(days, dtype="datetime64[D]")
