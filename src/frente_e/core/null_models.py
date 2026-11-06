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
