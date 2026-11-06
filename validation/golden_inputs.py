"""Gera as ENTRADAS sintéticas dos golden files (passo 1 de 3).

Uso:  python -m uv run python validation/golden_inputs.py

Passos completos (validation/r/README.md):
  1. este script grava tests/fixtures/golden/<caso>/input.csv (semente fixa);
  2. validation/r/scripts/make_golden.R calcula as referências com pbo/poolr;
  3. validation/freeze_golden.py congela tudo em manifest.json com SHA-256.

Os números são escritos com repr() (menor representação que reconstrói o double
exato), então o R lê exatamente os mesmos valores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frente_e.experiments.seeds import generator_for

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "golden"
ROOT_SEED = 20261106  # fixa e registrada; mudar exige regenerar e recongelar tudo

T = 480  # divisível por todos os S usados (o pacote pbo exige T % S == 0)
N = 20


def _write(case: str, matrix: np.ndarray, stream: int) -> None:
    folder = GOLDEN / case
    folder.mkdir(parents=True, exist_ok=True)
    header = ",".join(f"c{j}" for j in range(matrix.shape[1]))
    lines = [header] + [",".join(repr(float(v)) for v in row) for row in matrix]
    (folder / "input.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (folder / "input_generation.txt").write_text(
        f"root_seed={ROOT_SEED}\nstream={stream}\nshape={matrix.shape}\n"
        f"generator=numpy.random.PCG64 via SeedSequence(entropy=root_seed, spawn_key=(stream,))\n"
        f"numpy={np.__version__}\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    # Caso 1: ruído puro, N configurações i.i.d. N(0, 1%) -> PBO deve ficar perto de 0,5.
    rng = generator_for(ROOT_SEED, 1)
    _write("pbo_noise", rng.normal(0.0, 0.01, size=(T, N)), 1)

    # Caso 2: sinal plantado na coluna 0 (média diária +0,3%) -> PBO perto de 0.
    rng = generator_for(ROOT_SEED, 2)
    signal = rng.normal(0.0, 0.01, size=(T, N))
    signal[:, 0] += 0.003
    _write("pbo_signal", signal, 2)

    # Caso 3: empates exatos (colunas 1 e 2 idênticas) -> exercita as regras de empate.
    rng = generator_for(ROOT_SEED, 3)
    ties = rng.normal(0.0, 0.01, size=(T, N))
    ties[:, 2] = ties[:, 1]
    _write("pbo_ties", ties, 3)

    # Caso 4: N efetivo, configurações independentes -> N_eff perto de N.
    rng = generator_for(ROOT_SEED, 4)
    _write("meff_independent", rng.normal(0.0, 0.01, size=(T, N)), 4)

    # Caso 5: N efetivo, 4 blocos de 5 colunas fortemente correlacionadas -> N_eff perto de 4.
    rng = generator_for(ROOT_SEED, 5)
    factors = rng.normal(0.0, 0.01, size=(T, 4))
    noise = rng.normal(0.0, 0.002, size=(T, N))
    _write("meff_blocks", np.repeat(factors, 5, axis=1) + noise, 5)


if __name__ == "__main__":
    main()
