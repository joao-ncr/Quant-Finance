"""Preparação da entrada da CSCV e registro do resultado (camada de experimento).

- :func:`trim_oldest_to_multiple` aplica a regra pré-registrada de DE-05.b: descartar as
  T mod S observações **mais antigas**, para que o núcleo receba T divisível por S. As
  datas descartadas são devolvidas para irem ao ledger: nada é descartado em silêncio.
- :func:`cscv_result_to_json` converte o resultado do núcleo em JSON canônico, com o
  suficiente para reconstruir e auditar cada partição.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frente_e.core.contracts import ReturnsMatrix
from frente_e.core.cscv import CSCVResult, in_sample_block_sets
from frente_e.experiments.canonical import JsonValue

CONVENTION = "rank/(N+1); n* = primeiro máximo IS; postos médios OOS; φ = #(2r̄ ≤ N+1)/C"


def trim_oldest_to_multiple(
    matrix: ReturnsMatrix, n_blocks: int
) -> tuple[ReturnsMatrix, list[JsonValue]]:
    """Remove as ``T mod S`` primeiras linhas. Devolve a matriz e os índices removidos."""
    in_sample_block_sets(n_blocks)  # valida S (par, >= 2) com a mesma regra do núcleo
    excess = matrix.n_observations % n_blocks
    dropped = [_index_to_json(v) for v in matrix.index[:excess]]
    if excess == 0:
        return matrix, dropped
    trimmed = ReturnsMatrix(matrix.values[excess:], matrix.index[excess:], matrix.config_ids)
    return trimmed, dropped


def cscv_result_to_json(
    result: CSCVResult, matrix: ReturnsMatrix, dropped_index: list[JsonValue]
) -> dict[str, JsonValue]:
    """Resultado completo por partição, com identidades das configurações."""
    ids = matrix.config_ids
    partitions: list[JsonValue] = [
        {
            "in_sample_blocks": [int(b) for b in result.in_sample_blocks[c]],
            "best_in_sample": ids[int(result.best_in_sample[c])],
            "in_sample_tie_size": int(result.in_sample_tie_size[c]),
            "oos_rank": float(result.oos_rank[c]),
            "omega": float(result.omega[c]),
            "logit": float(result.logit[c]),
            "in_sample_performance": float(result.in_sample_performance[c]),
            "out_of_sample_performance": float(result.out_of_sample_performance[c]),
        }
        for c in range(result.n_partitions)
    ]
    return {
        "convention": CONVENTION,
        "n_blocks": result.n_blocks,
        "n_configs": result.n_configs,
        "n_observations": matrix.n_observations,
        "dropped_oldest_index": dropped_index,
        "first_index": _index_to_json(matrix.index[0]),
        "last_index": _index_to_json(matrix.index[-1]),
        "config_ids": list(ids),
        "pbo": result.pbo,
        "n_partitions": result.n_partitions,
        "n_in_sample_ties": result.n_in_sample_ties,
        "probability_of_oos_loss": result.probability_of_oos_loss,
        "partitions": partitions,
    }


def _index_to_json(value: Any) -> JsonValue:
    if isinstance(value, np.datetime64):
        return str(value)
    return int(value)
