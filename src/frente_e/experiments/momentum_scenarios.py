"""Cenários sintéticos sem sinal para a grade de momentum (DE-01) e execução de mecânica.

Os cenários foram fixados em 2026-11-06, **antes** de qualquer estudo de N efetivo ou de
embargo (docs/METHODOLOGY_CONVENTIONS.md §7). Todos têm média dos ativos = r_f, logo SR
verdadeiro zero para qualquer configuração. Eles variam só a **dependência**:

- ``base``: fator de mercado forte, volatilidade constante;
- ``low_common``: fator de mercado fraco (menor correlação entre ativos);
- ``vol_clustering``: base com volatilidade estocástica comum (dependência temporal na
  variância, não na média);
- ``heterogeneous``: betas e volatilidades idiossincráticas muito dispersos.

Os parâmetros cross-sectionais (betas, volatilidades) de cada cenário vêm de um fluxo fixo
(``param_stream``), idêntico em todas as réplicas; só os choques dependem da semente raiz.
T = 6550 pregões ≈ 2000–2025 (§6.7).

Execução: ``python -m frente_e.experiments.momentum_scenarios`` (registrada no ledger).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.core.momentum import (
    AssetPanel,
    MomentumGridResult,
    build_momentum_returns,
    momentum_family,
)
from frente_e.core.null_models import business_day_calendar, simulate_factor_returns
from frente_e.errors import ContractViolationError
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import JsonValue, canonical_bytes
from frente_e.experiments.identity import InputKind, InputReference, RunKind, RunSpec
from frente_e.experiments.ledger import LedgerWriter
from frente_e.experiments.provenance import code_revision, software_environment
from frente_e.experiments.seeds import generator_for, new_root_seed
from frente_e.settings import load_settings

FloatArray = NDArray[np.float64]

PARAM_SEED = 20261106  # semente fixa dos parâmetros cross-sectionais (não das réplicas)

SCENARIOS: dict[str, JsonValue] = {
    "base": {
        "param_stream": 1,
        "n_assets": 60,
        "market_vol": 0.012,
        "beta_range": [0.7, 1.3],
        "idio_vol_range": [0.012, 0.025],
        "vol_persistence": 0.0,
        "vol_of_vol": 0.0,
    },
    "low_common": {
        "param_stream": 2,
        "n_assets": 60,
        "market_vol": 0.005,
        "beta_range": [0.7, 1.3],
        "idio_vol_range": [0.012, 0.025],
        "vol_persistence": 0.0,
        "vol_of_vol": 0.0,
    },
    "vol_clustering": {
        "param_stream": 3,
        "n_assets": 60,
        "market_vol": 0.012,
        "beta_range": [0.7, 1.3],
        "idio_vol_range": [0.012, 0.025],
        "vol_persistence": 0.98,
        # emenda 3 (2026-11-06): era 0,15 -> desvio estacionário de log σ ≈ 0,75, que gerou
        # retorno <= −100% (run-7dd0ad1965784ee8ae05681dc337c168, falha registrada)
        "vol_of_vol": 0.06,
    },
    "heterogeneous": {
        "param_stream": 4,
        "n_assets": 60,
        "market_vol": 0.012,
        "beta_range": [0.3, 1.7],
        "idio_vol_range": [0.005, 0.04],
        "vol_persistence": 0.0,
        "vol_of_vol": 0.0,
    },
}

COMMON: dict[str, JsonValue] = {
    "n_obs": 6550,
    "start": "2000-01-03",
    "risk_free_daily": 0.0004,
}


def scenario_panel(
    spec: Mapping[str, Any], common: Mapping[str, Any], rng: np.random.Generator
) -> tuple[AssetPanel, FloatArray]:
    """Painel completo (sem ausências, todos elegíveis) e r_f constante do cenário."""
    params = generator_for(PARAM_SEED, int(spec["param_stream"]))
    n_assets, n_obs = int(spec["n_assets"]), int(common["n_obs"])
    beta_low, beta_high = (float(v) for v in spec["beta_range"])
    vol_low, vol_high = (float(v) for v in spec["idio_vol_range"])
    betas = params.uniform(beta_low, beta_high, size=n_assets)
    vols = params.uniform(vol_low, vol_high, size=n_assets)
    rf = float(common["risk_free_daily"])
    returns = simulate_factor_returns(
        n_obs,
        rng,
        mean=rf,
        market_vol=float(spec["market_vol"]),
        betas=betas,
        idiosyncratic_vols=vols,
        vol_persistence=float(spec["vol_persistence"]),
        vol_of_vol=float(spec["vol_of_vol"]),
    )
    tri = np.ones_like(returns)
    tri[1:] = np.cumprod(1.0 + returns[1:], axis=0)
    panel = AssetPanel(
        business_day_calendar(str(common["start"]), n_obs),
        tri,
        np.ones(tri.shape, dtype=np.bool_),
        tuple(f"s{i:02d}" for i in range(n_assets)),
    )
    return panel, np.full(n_obs, rf)


def scenario_grid(
    name: str, params: Mapping[str, Any], rng: np.random.Generator, family: str = "F144"
) -> MomentumGridResult:
    scenarios = params["scenarios"]
    if name not in scenarios:
        raise ContractViolationError(f"cenário desconhecido: {name!r}")
    panel, rf = scenario_panel(scenarios[name], params["common"], rng)
    return build_momentum_returns(panel, rf, momentum_family(family))


def mechanics_summary(result: MomentumGridResult) -> dict[str, JsonValue]:
    """Resumo descritivo da mecânica (não é resultado de desempenho)."""
    m = result.excess_returns
    corr = np.asarray(np.corrcoef(m.values, rowvar=False), dtype=np.float64)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    diag = result.diagnostics.values()
    return {
        "n_rows": m.n_observations,
        "n_configs": m.n_configs,
        "first_date": str(m.index[0]),
        "last_date": str(m.index[-1]),
        "n_formations": int(result.formation_dates.size),
        "boundary_ties": int(sum(d.boundary_ties for d in diag)),
        "deferred_entries": int(sum(d.deferred_entries for d in diag)),
        "stale_position_days": int(sum(d.stale_position_days for d in diag)),
        "config_correlation_min": float(off.min()),
        "config_correlation_median": float(np.median(off)),
        "config_correlation_max": float(off.max()),
    }


PARAMETERS: dict[str, JsonValue] = {
    "protocol": "docs/METHODOLOGY_CONVENTIONS.md §7.1 (mecânica; fixado em 2026-11-06)",
    "scenarios": SCENARIOS,
    "common": COMMON,
    "family": "F144",
}


def run_mechanics(params: Mapping[str, Any], root_seed: int) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {}
    for index, name in enumerate(sorted(params["scenarios"])):
        result = scenario_grid(name, params, generator_for(root_seed, index), params["family"])
        out[name] = mechanics_summary(result)
    return out


def execute(
    params: Mapping[str, Any],
    *,
    project_root: Path,
    ledger_path: Path,
    artifacts_root: Path,
    root_seed: int | None = None,
) -> tuple[str, ArtifactRef, dict[str, JsonValue]]:
    spec = RunSpec(
        experiment_id="sim-momentum-mechanics",
        run_kind=RunKind.SYNTHETIC,
        parameters=dict(params),
        random_seeds={"root": new_root_seed() if root_seed is None else root_seed},
        input_data_references=(
            InputReference(
                InputKind.SYNTHETIC,
                "generator://factor-model(mean=r_f)",
                "painel sem sinal por cenário (core.null_models.simulate_factor_returns)",
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
        result = run_mechanics(params, spec.random_seeds["root"])
        ref = ArtifactStore(artifacts_root).put_bytes(
            canonical_bytes(result),
            suffix=".json",
            media_type="application/json",
            description="sim-momentum-mechanics: resumo por cenário",
        )
        run.set_results([ref])
    return run.run_id, ref, result


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
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
