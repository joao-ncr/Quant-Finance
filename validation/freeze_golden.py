"""Passo 3 de 3: congela os golden files num manifest.json por caso.

Uso:  python -m uv run python validation/freeze_golden.py

O manifest registra pacote/versão (de meta.csv, gravado pelo R), a configuração, os
SHA-256 de entrada e saídas e a tolerância numérica com justificativa. Os testes Python
(tests/reference/) conferem os hashes antes de usar qualquer valor.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from frente_e.experiments.canonical import sha256_file

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "golden"

TOLERANCES = {
    "pbo": {
        "discrete_fields": ["is_blocks", "n_star", "n_max_oos", "os_rank"],
        "discrete_rule": "igualdade exata (argmax, ranks e partições são inteiros)",
        "derived_from_ranks": ["omega_bar", "lambda_raw", "lambda", "phi"],
        "derived_rule": "rtol=1e-12: funções elementares de inteiros (rank/N, logit)",
        "float_fields": ["rn", "rbn"],
        "float_rule": (
            "rtol=1e-12, atol=1e-15: Sharpe de até 480 observações; a diferença de ordem "
            "de soma entre R e NumPy gera erro relativo da ordem de T·eps ≈ 1e-13"
        ),
    },
    "meff": {
        "meff_floor": "igualdade exata do valor inteiro (poolr aplica floor)",
        "eigenvalue": (
            "rtol=1e-9: autovalores via LAPACK nos dois lados; matriz 20×20 bem "
            "condicionada. Risco conhecido: liji e gao têm descontinuidades (floor, "
            "limiar 0,995), então um autovalor no limite pode mudar o inteiro"
        ),
    },
}


def _read_meta(folder: Path) -> dict[str, str]:
    with (folder / "meta.csv").open(encoding="utf-8", newline="") as fh:
        return {row["key"]: row["value"] for row in csv.DictReader(fh)}


def main() -> None:
    for folder in sorted(p for p in GOLDEN.iterdir() if p.is_dir()):
        kind = folder.name.split("_", 1)[0]
        expected = sorted(folder.glob("expected_*.csv"))
        if not expected:
            raise SystemExit(f"{folder.name}: sem expected_*.csv; rode o passo 2 (R)")
        manifest = {
            "case": folder.name,
            "kind": kind,
            "reference": _read_meta(folder),
            "input": {"file": "input.csv", "sha256": sha256_file(folder / "input.csv")},
            "input_generation": {
                "file": "input_generation.txt",
                "sha256": sha256_file(folder / "input_generation.txt"),
            },
            "meta": {"file": "meta.csv", "sha256": sha256_file(folder / "meta.csv")},
            "expected": [{"file": p.name, "sha256": sha256_file(p)} for p in expected],
            "tolerance": TOLERANCES[kind],
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"frozen {folder.name}: {len(expected)} expected file(s)")


if __name__ == "__main__":
    main()
