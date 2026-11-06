"""Simulação preliminar de N efetivo e V (docs/METHODOLOGY_CONVENTIONS.md §4.4).

Responde, sob a nula global e com estruturas de correlação **genéricas**:
- quanto cada candidato a N efetivo erra o máximo esperado usado no DSR (§4.2);
- quanto a variância entre tentativas encolhe com a correlação (DE-04.b, §4.3);
- a fração de "descobertas" com DSR > 0,95 (descritivo, insumo de DE-07);
- o erro da aproximação f(N) contra o valor exato i.i.d.

**Não** escolhe o estimador principal: o conjunto final de estruturas inclui a grade de
momentum (após DE-01). Execução: ``python -m frente_e.experiments.neff_study``, sempre
registrada no ledger como ``RunKind.SYNTHETIC``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from frente_e.core import null_models
from frente_e.core.effective_trials import ESTIMATORS, GAO_VARIANCE_SHARE
from frente_e.core.sharpe_inference import (
    MIN_TRIALS,
    deflated_sharpe_ratio,
    expected_max_factor,
)
from frente_e.errors import ContractViolationError
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import JsonValue, canonical_bytes
from frente_e.experiments.identity import InputKind, InputReference, RunKind, RunSpec
from frente_e.experiments.ledger import LedgerWriter
from frente_e.experiments.provenance import code_revision, software_environment
from frente_e.experiments.seeds import generator_for, new_root_seed
from frente_e.settings import load_settings

FloatArray = NDArray[np.float64]

EXPERIMENT_ID = "sim-neff-preliminary"

STRUCTURES: dict[str, JsonValue] = {
    "independent": {"kind": "equicorrelation", "m": 100, "rho": 0.0},
    "equi_0.5": {"kind": "equicorrelation", "m": 100, "rho": 0.5},
    "equi_0.9": {"kind": "equicorrelation", "m": 100, "rho": 0.9},
    "blocks_10x10_0.9": {"kind": "blocks", "n_blocks": 10, "block_size": 10, "rho": 0.9},
    "blocks_4x25_0.95": {"kind": "blocks", "n_blocks": 4, "block_size": 25, "rho": 0.95},
    "ar1_0.9": {"kind": "ar1", "m": 100, "rho": 0.9},
    "ar1_0.99": {"kind": "ar1", "m": 100, "rho": 0.99},
    "grid_10x10_0.9_0.9": {
        "kind": "grid",
        "rows": 10,
        "cols": 10,
        "rho_rows": 0.9,
        "rho_cols": 0.9,
    },
    "grid_10x10_0.97_0.8": {
        "kind": "grid",
        "rows": 10,
        "cols": 10,
        "rho_rows": 0.97,
        "rho_cols": 0.8,
    },
}

PARAMETERS: dict[str, JsonValue] = {
    "protocol": "docs/METHODOLOGY_CONVENTIONS.md §4.4 (fixado em 2026-11-06)",
    "structures": STRUCTURES,
    "n_reps": 200_000,
    "chunk": 20_000,
    "n_obs": 1250,
    "n_correlation_draws": 20,
    "estimators": [name for name in sorted(ESTIMATORS)],
    "gao_variance_share": GAO_VARIANCE_SHARE,
    "dsr_threshold": 0.95,
    "approximation_table_n": [2, 3, 5, 10, 20, 50, 100, 1000],
}


def build_correlation(spec: Mapping[str, Any]) -> FloatArray:
    kind = spec["kind"]
    if kind == "equicorrelation":
        return null_models.equicorrelation(int(spec["m"]), float(spec["rho"]))
    if kind == "blocks":
        return null_models.block_correlation(
            int(spec["n_blocks"]), int(spec["block_size"]), float(spec["rho"])
        )
    if kind == "ar1":
        return null_models.ar1_correlation(int(spec["m"]), float(spec["rho"]))
    if kind == "grid":
        return null_models.separable_grid_correlation(
            int(spec["rows"]), int(spec["cols"]), float(spec["rho_rows"]), float(spec["rho_cols"])
        )
    raise ContractViolationError(f"estrutura desconhecida: {kind!r}")


def implied_n(expected_max: float) -> float | None:
    """N* com f(N*) = E[máx]. ``None`` se E[máx] < f(2), ou seja, N* fora do domínio."""
    if expected_max < float(expected_max_factor(MIN_TRIALS)):
        return None
    upper = 1e15
    if expected_max > float(expected_max_factor(upper)):
        raise ContractViolationError("E[máx] acima do alcance numérico de f")
    root = optimize.brentq(
        lambda n: float(expected_max_factor(n)) - expected_max, MIN_TRIALS, upper, xtol=1e-12
    )
    return float(root)


def approximation_table(ns: Sequence[int]) -> list[JsonValue]:
    rows: list[JsonValue] = []
    for n in ns:
        approx = float(expected_max_factor(n))
        exact = null_models.expected_max_iid_normal(float(n))
        rows.append({"n": n, "approx": approx, "exact": exact, "rel_error": approx / exact - 1})
    return rows


def evaluate_structure(
    correlation: FloatArray,
    params: Mapping[str, Any],
    rng_max: np.random.Generator,
    rng_corr: np.random.Generator,
) -> dict[str, JsonValue]:
    n_obs = int(params["n_obs"])
    threshold = float(params["dsr_threshold"])
    maxima, variances = null_models.sample_standardized_null(
        correlation, int(params["n_reps"]), rng_max, chunk=int(params["chunk"])
    )
    target = float(maxima.mean())
    target_se = float(maxima.std(ddof=1) / math.sqrt(maxima.size))
    mean_sd = float(np.sqrt(variances).mean())

    # ŜR ≈ Z·σ, com σ² = 1/(T−1): variância assintótica de ŜR sob SR = 0 (eq. 2)
    sigma = 1.0 / math.sqrt(n_obs - 1)
    best_sharpe = maxima * sigma
    v_cross = variances * sigma**2
    v_h0 = sigma**2

    estimated = [
        null_models.sample_correlation_estimate(correlation, n_obs, rng_corr)
        for _ in range(int(params["n_correlation_draws"]))
    ]
    estimators: dict[str, JsonValue] = {}
    for name in params["estimators"]:
        estimator = ESTIMATORS[name]
        n_hat = estimator(correlation)
        on_estimates = np.array([estimator(c) for c in estimated])
        entry: dict[str, JsonValue] = {
            "n_hat": n_hat,
            "n_hat_estimated_mean": float(on_estimates.mean()),
            "n_hat_estimated_sd": float(on_estimates.std(ddof=1)),
        }
        if n_hat < MIN_TRIALS:
            entry["below_domain"] = True
        else:
            factor = float(expected_max_factor(n_hat))
            rejections = {}
            for label, variance in (("v_cross", v_cross), ("v_h0", v_h0)):
                dsr = deflated_sharpe_ratio(
                    best_sharpe, n_obs, 0.0, 3.0, n_trials=n_hat, sharpe_variance=variance
                )
                rejections[label] = float(np.mean(dsr > threshold))
            entry.update(
                {
                    "below_domain": False,
                    "sr0_rel_error": factor / target - 1,
                    "sr0_rel_error_with_v_cross": mean_sd * factor / target - 1,
                    "rejection_rate_v_cross": rejections["v_cross"],
                    "rejection_rate_v_h0": rejections["v_h0"],
                }
            )
        estimators[name] = entry
    return {
        "expected_max": target,
        "expected_max_se": target_se,
        "implied_n": implied_n(target),
        "mean_cross_variance": float(variances.mean()),
        "mean_cross_sd": mean_sd,
        "estimators": estimators,
    }


def run_study(params: Mapping[str, Any], root_seed: int) -> dict[str, JsonValue]:
    """Determinístico dado (params, root_seed): cada estrutura tem fluxos próprios.

    O índice do fluxo vem da ordem **canônica** (alfabética) dos nomes, a mesma do JSON
    gravado no ledger. Usar a ordem de inserção do dicionário faria a reconstrução a
    partir do ledger sortear fluxos diferentes (defeito detectado pelo teste de
    integração em 2026-11-06).
    """
    structures: dict[str, JsonValue] = {}
    for index, name in enumerate(sorted(params["structures"])):
        correlation = build_correlation(params["structures"][name])
        structures[name] = evaluate_structure(
            correlation,
            params,
            generator_for(root_seed, index, 0),
            generator_for(root_seed, index, 1),
        )
    return {
        "structures": structures,
        "approximation": approximation_table(params["approximation_table_n"]),
    }


def execute(
    params: Mapping[str, Any],
    *,
    project_root: Path,
    ledger_path: Path,
    artifacts_root: Path,
    root_seed: int | None = None,
) -> tuple[str, ArtifactRef, dict[str, JsonValue]]:
    """Executa dentro do ledger e grava o resultado como JSON canônico endereçado por hash."""
    spec = RunSpec(
        experiment_id=EXPERIMENT_ID,
        run_kind=RunKind.SYNTHETIC,
        parameters=dict(params),
        random_seeds={"root": new_root_seed() if root_seed is None else root_seed},
        input_data_references=(
            InputReference(
                InputKind.SYNTHETIC,
                "generator://gaussian(0,C)",
                "Z ~ N(0, C) por estrutura; Ĉ de T normais com correlação C",
            ),
        ),
    )
    with (
        LedgerWriter(ledger_path) as ledger,
        ledger.run(
            spec,
            software_environment=software_environment(project_root),
            code_revision=code_revision(project_root),
        ) as run,
    ):
        result = run_study(params, spec.random_seeds["root"])
        ref = ArtifactStore(artifacts_root).put_bytes(
            canonical_bytes(result),
            suffix=".json",
            media_type="application/json",
            description=f"{EXPERIMENT_ID}: resultados por estrutura e tabela de aproximação",
        )
        run.set_results([ref])
    return run.run_id, ref, result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    settings = load_settings()
    run_id, ref, result = execute(
        PARAMETERS,
        project_root=settings.paths.root,
        ledger_path=settings.paths.ledger,
        artifacts_root=settings.paths.artifacts,
    )
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"run_id": run_id, "artifact": ref.to_json()}, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
