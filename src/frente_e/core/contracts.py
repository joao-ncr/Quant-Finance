"""Contratos de dados entre a camada de experimentos e o núcleo.

:class:`ReturnsMatrix` é a entrada comum de métricas, N efetivo e CSCV: uma matriz
T × N de retornos (linhas = observações em ordem temporal, colunas = configurações).

Decisões de contrato (baixo risco, revisáveis):
- dtype float64 obrigatório, sem conversão implícita (precisão dos momentos do PSR);
- valores não finitos são rejeitados. A política para retornos ausentes
  (ex.: dias sem posição ou ativo suspenso) ainda não foi definida (DE-01/DE-06);
  rejeitar força o tratamento explícito antes do núcleo, em vez de propagar NaN;
- índice estritamente crescente (datas ou posições inteiras);
- os arrays são copiados e marcados como somente leitura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.errors import ContractViolationError


@dataclass(frozen=True, eq=False)
class ReturnsMatrix:
    values: NDArray[np.float64]
    index: NDArray[Any]
    config_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        if values.dtype != np.float64:
            raise ContractViolationError(f"values deve ser float64; recebido {values.dtype}")
        if values.ndim != 2:
            raise ContractViolationError(
                f"values deve ser 2-D (T × N); recebido ndim={values.ndim}"
            )
        n_obs, n_configs = values.shape
        if n_obs < 2 or n_configs < 1:
            raise ContractViolationError(
                "São necessárias T >= 2 observações e N >= 1 configurações; "
                f"recebido {values.shape}"
            )
        non_finite = int(np.count_nonzero(~np.isfinite(values)))
        if non_finite:
            raise ContractViolationError(
                f"{non_finite} valores não finitos em values. A política para retornos ausentes "
                "não está definida (DE-01/DE-06): trate-os explicitamente antes do núcleo."
            )

        index = np.array(self.index, copy=True)
        if index.ndim != 1 or index.shape[0] != n_obs:
            raise ContractViolationError(
                f"index deve ser 1-D com {n_obs} elementos; recebido shape {index.shape}"
            )
        if not (
            np.issubdtype(index.dtype, np.integer) or np.issubdtype(index.dtype, np.datetime64)
        ):
            raise ContractViolationError(
                f"index deve ser inteiro ou datetime64; recebido {index.dtype}"
            )
        if not bool(np.all(index[1:] > index[:-1])):
            raise ContractViolationError(
                "index deve ser estritamente crescente (sem NaT nem duplicatas)"
            )

        config_ids = tuple(self.config_ids)
        if len(config_ids) != n_configs:
            raise ContractViolationError(
                f"config_ids tem {len(config_ids)} elementos; esperado {n_configs}"
            )
        if not all(isinstance(c, str) and c for c in config_ids):
            raise ContractViolationError("config_ids devem ser strings não vazias")
        if len(set(config_ids)) != n_configs:
            raise ContractViolationError("config_ids devem ser únicos")

        values.flags.writeable = False
        index.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "config_ids", config_ids)

    @property
    def n_observations(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_configs(self) -> int:
        return int(self.values.shape[1])
