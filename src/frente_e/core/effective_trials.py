"""Candidatos a número efetivo de tentativas independentes (DE-03; protocolo decidido).

Todos recebem a matriz de correlação M × M entre as séries de retornos das tentativas e
devolvem N efetivo **contínuo** (DE-03.a: nenhum arredondamento). A escolha do estimador
principal está **aberta** e será feita por simulação antes de qualquer dado real
(docs/METHODOLOGY_CONVENTIONS.md §4). Ninguém deve escolher o estimador olhando o DSR.

Definições (Nyholt, Li-Ji, Gao, Galwey) copiadas do código de ``poolr::meff`` 1.2-0, lido
em 2026-11-06; os artigos originais não foram conferidos. Diferenças em relação ao pacote:

- o pacote aplica ``floor()`` a todos os métodos; aqui não (comparação: ``floor(nosso) ==
  poolr``, tests/reference);
- Li-Ji: o pacote soma ``√eps`` aos autovalores absolutos para contornar arredondamento.
  Aqui se usa uma identidade exata. Como o traço de uma correlação é M e, para λ ≥ 0,
  ``f(λ) = λ − (⌊λ⌋ − 1)·1{λ ≥ 1}``, vale ``Σ f(λᵢ) = M − Σ_{λᵢ ≥ 1} (⌊λᵢ⌋ − 1)``. O
  estimador é **inteiro por construção**; somar as partes fracionárias só acrescenta ruído
  de ponto flutuante (ex.: 6,9999999999998 em vez de 7, e ``floor`` daria 6). A identidade
  supõe autovalores ≥ 0; negativos de arredondamento entram pelo valor absoluto, como no
  pacote. A função é descontínua: quando um autovalor cruza um inteiro ≥ 2, N cai 1.

``average_correlation`` é a eq. 9 do Apêndice A.3 de Bailey e López de Prado (2014),
conferida visualmente: N̂ = ρ̂ + (1 − ρ̂)·M, com ρ̂ a média das correlações fora da diagonal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from frente_e.errors import ContractViolationError

FloatArray = NDArray[np.float64]

SYMMETRY_ATOL = 1e-12
"""Folga só para arredondamento de ponto flutuante na simetria e na diagonal unitária
(``np.corrcoef`` não garante simetria bit a bit). Não é parâmetro estatístico."""

GAO_VARIANCE_SHARE = 0.995
"""Padrão de ``poolr`` (C = 0,995) para o método de Gao et al."""


def validate_correlation(correlation: FloatArray) -> FloatArray:
    """Exige matriz float64 quadrada (M ≥ 2), finita, simétrica, com diagonal 1 e |ρ| ≤ 1."""
    c = np.asarray(correlation)
    if c.dtype != np.float64 or c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
        raise ContractViolationError(
            f"correlação deve ser float64 M × M com M >= 2; recebido {c.dtype} {c.shape}"
        )
    if not np.all(np.isfinite(c)):
        raise ContractViolationError("correlação contém valores não finitos")
    if not np.allclose(c, c.T, rtol=0.0, atol=SYMMETRY_ATOL):
        raise ContractViolationError("correlação não é simétrica")
    if not np.allclose(np.diag(c), 1.0, rtol=0.0, atol=SYMMETRY_ATOL):
        raise ContractViolationError("diagonal da correlação deve ser 1")
    if np.any(np.abs(c) > 1.0 + SYMMETRY_ATOL):
        raise ContractViolationError("correlações fora de [-1, 1]")
    return c


def correlation_eigenvalues(correlation: FloatArray) -> FloatArray:
    """Autovalores (ordem decrescente) da matriz de correlação validada."""
    values: FloatArray = np.linalg.eigvalsh(validate_correlation(correlation))[::-1]
    return values


def nyholt(correlation: FloatArray) -> float:
    """``1 + (M − 1)·(1 − Var(λ)/M)``, com variância amostral (ddof = 1), como no R."""
    eig = correlation_eigenvalues(correlation)
    m = eig.size
    return float(1.0 + (m - 1) * (1.0 - np.var(eig, ddof=1) / m))


def li_ji(correlation: FloatArray) -> float:
    """``Σ f(|λᵢ|)``, com ``f(x) = 1{x ≥ 1} + (x − ⌊x⌋)``, calculado pela identidade exata
    ``M − Σ_{|λᵢ| ≥ 1} (⌊|λᵢ|⌋ − 1)`` (ver o docstring do módulo)."""
    eig = np.abs(correlation_eigenvalues(correlation))
    large = eig[eig >= 1.0]
    return float(eig.size - int(np.sum(np.floor(large) - 1.0)))


def gao(correlation: FloatArray, variance_share: float = GAO_VARIANCE_SHARE) -> float:
    """Nº de componentes principais até a fração acumulada de variância superar ``C``.

    Inteiro por construção (é uma contagem), não por arredondamento.
    """
    if not 0.0 <= variance_share < 1.0:
        raise ContractViolationError("variance_share deve estar em [0, 1)")
    eig = correlation_eigenvalues(correlation)
    share = np.cumsum(eig) / np.sum(eig)
    return float(np.flatnonzero(share > variance_share)[0] + 1)


def galwey(correlation: FloatArray) -> float:
    """``(Σ √λᵢ⁺)² / Σ λᵢ⁺``, com autovalores negativos (arredondamento) zerados."""
    eig = np.clip(correlation_eigenvalues(correlation), 0.0, None)
    return float(np.sum(np.sqrt(eig)) ** 2 / np.sum(eig))


def average_correlation(correlation: FloatArray) -> float:
    """``ρ̂ + (1 − ρ̂)·M`` (Bailey e López de Prado 2014, A.3, eq. 8–9).

    A fórmula interpola entre ρ̂ = 1 (N = 1) e ρ̂ = 0 (N = M). Com ρ̂ < 0 ela extrapola e
    dá N̂ > M, até M + 1 no limite ρ̂ = −1/(M−1) do artigo. O valor **não** é cortado em M:
    isso seria uma modificação não prevista na fonte. Se ocorrer, deve ser reportado.
    """
    c = validate_correlation(correlation)
    m = int(c.shape[0])
    rho = (float(np.sum(c)) - m) / (m * (m - 1))
    return rho + (1.0 - rho) * m


def raw_count(correlation: FloatArray) -> float:
    """M, o nº bruto de tentativas: cota superior conservadora (ignora a correlação)."""
    return float(validate_correlation(correlation).shape[0])


ESTIMATORS: Mapping[str, Callable[[FloatArray], float]] = {
    "raw": raw_count,
    "nyholt": nyholt,
    "liji": li_ji,
    "gao": gao,
    "galwey": galwey,
    "average_correlation": average_correlation,
}
"""Conjunto pré-registrado de candidatos (DE-03). Não ampliar sem registro em DE-03."""
