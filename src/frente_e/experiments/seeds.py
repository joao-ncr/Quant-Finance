"""Sementes e fluxos aleatórios reproduzíveis (DA-07).

Cada fluxo é derivado de (semente raiz, caminho de inteiros) por
``SeedSequence(entropy=raiz, spawn_key=caminho)``. O fluxo da réplica *i* depende só
de *i*, não da ordem de execução nem do número de processos. O núcleo recebe o
``Generator`` pronto e nunca toca no estado global.
"""

from __future__ import annotations

import secrets

import numpy as np

from frente_e.errors import ContractViolationError
from frente_e.experiments.identity import SEED_UPPER_BOUND


def new_root_seed() -> int:
    """Semente raiz nova (63 bits, da fonte de entropia do SO). Deve ir para o ledger."""
    return secrets.randbits(63)


def generator_for(root_seed: int, *path: int) -> np.random.Generator:
    if isinstance(root_seed, bool) or not 0 <= root_seed < SEED_UPPER_BOUND:
        raise ContractViolationError("root_seed deve estar em [0, 2^63)")
    if any(isinstance(p, bool) or p < 0 for p in path):
        raise ContractViolationError("o caminho do fluxo deve conter inteiros >= 0")
    sequence = np.random.SeedSequence(entropy=root_seed, spawn_key=tuple(path))
    return np.random.Generator(np.random.PCG64(sequence))
