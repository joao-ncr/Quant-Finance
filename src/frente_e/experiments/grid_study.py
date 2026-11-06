"""Estudo da grade de momentum sem sinal: N efetivo, V e embargo (METHODOLOGY §7.3–7.4).

Por réplica e cenário (``experiments.momentum_scenarios``), com dois conjuntos
independentes A e B gerados com a mesma estrutura:

- em A, para F4, F16 e F144: ŜR diários, máximo, V entre tentativas (ddof = 1), Ĉ e os
  seis candidatos a N efetivo; em F144, a autocorrelação média dos retornos das
  configurações nos lags 1, 5, 21 e 63;
- CSCV (S fixado no protocolo) em A sem embargo e com embargo h, e a linha de base de
  **metades independentes** (IS de A, OOS do mesmo bloco complementar de B), que preserva
  a estrutura transversal e elimina a dependência temporal entre metades (emenda 2).

Fluxos aleatórios: ``generator_for(raiz, cenário, réplica, conjunto)``, independentes da
ordem de execução.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from frente_e.core.cscv import block_partition_performance, rank_partitions
from frente_e.core.effective_trials import ESTIMATORS
from frente_e.core.metrics import sharpe_ratio
from frente_e.core.momentum import momentum_family
from frente_e.core.sharpe_inference import MIN_TRIALS, expected_max_factor
from frente_e.errors import ContractViolationError
from frente_e.experiments import momentum_scenarios
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import JsonValue, canonical_bytes, sha256_hex
from frente_e.experiments.cscv_input import trim_oldest_to_multiple
from frente_e.experiments.identity import InputKind, InputReference, RunKind, RunSpec
from frente_e.experiments.ledger import LedgerWriter
from frente_e.experiments.provenance import code_revision, software_environment
from frente_e.experiments.seeds import generator_for, new_root_seed
from frente_e.settings import load_settings

FloatArray = NDArray[np.float64]
FAMILIES = ("F4", "F16", "F144")
ACF_LAGS = (1, 5, 21, 63)

PARAMETERS: dict[str, JsonValue] = {
    "protocol": "docs/METHODOLOGY_CONVENTIONS.md §7.3–7.4 (fixado antes da execução, 2026-11-06)",
    "scenarios": momentum_scenarios.SCENARIOS,
    "common": momentum_scenarios.COMMON,
    "replicates": 50,
    "n_blocks": 16,
    "embargoes": [21, 63],
    "families": list(FAMILIES),
    "estimators": [name for name in sorted(ESTIMATORS)],
    "acf_lags": list(ACF_LAGS),
}


# ------------------------------------------------------------------ réplica (determinística)
def _pbo(rank: FloatArray, n_configs: int) -> float:
    return float(np.mean(2.0 * rank <= n_configs + 1))


def _acf(x: FloatArray, lag: int) -> float:
    """Autocorrelação média (entre configurações) no lag dado."""
    centered = x - x.mean(axis=0)
    num = np.sum(centered[lag:] * centered[:-lag], axis=0)
    den = np.sum(centered**2, axis=0)
    return float(np.mean(num / den))


def replicate_metrics(
    params: Mapping[str, Any], scenario: str, scenario_index: int, rep: int, root_seed: int
) -> dict[str, JsonValue]:
    grid_a = momentum_scenarios.scenario_grid(
        scenario, params, generator_for(root_seed, scenario_index, rep, 0)
    ).excess_returns
    grid_b = momentum_scenarios.scenario_grid(
        scenario, params, generator_for(root_seed, scenario_index, rep, 1)
    ).excess_returns
    s = int(params["n_blocks"])
    trimmed_a, _ = trim_oldest_to_multiple(grid_a, s)
    trimmed_b, _ = trim_oldest_to_multiple(grid_b, s)
    # mesma permutação de linhas para todas as configurações: destrói a dependência temporal,
    # preserva a correlação transversal e a complementaridade das metades (emenda 2b)
    permutation = generator_for(root_seed, scenario_index, rep, 2).permutation(
        trimmed_a.n_observations
    )
    out: dict[str, JsonValue] = {"n_obs": grid_a.n_observations}
    for family in params["families"]:
        cols = [grid_a.config_ids.index(c.config_id) for c in momentum_family(family)]
        x = grid_a.values[:, cols]
        sr = sharpe_ratio(x)
        corr = np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)
        entry: dict[str, JsonValue] = {
            "sharpe": [float(v) for v in sr],
            "max_sharpe": float(sr.max()),
            "v_cross": float(np.var(sr, ddof=1)),
            "n_hat": {name: ESTIMATORS[name](corr) for name in params["estimators"]},
            "corr_offdiag_mean": float(corr[~np.eye(len(cols), dtype=bool)].mean()),
        }
        xa, xb = trimmed_a.values[:, cols], trimmed_b.values[:, cols]
        _, is_a, oos_a = block_partition_performance(xa, s)
        pbo: dict[str, JsonValue] = {"plain": _pbo(rank_partitions(is_a, oos_a)[2], len(cols))}
        _, _, oos_b = block_partition_performance(xb, s)
        pbo["independent_halves"] = _pbo(rank_partitions(is_a, oos_b)[2], len(cols))
        _, is_p, oos_p = block_partition_performance(xa[permutation], s)
        pbo["permuted_rows"] = _pbo(rank_partitions(is_p, oos_p)[2], len(cols))
        for h in params["embargoes"]:
            _, is_e, oos_e = block_partition_performance(xa, s, embargo=int(h))
            pbo[f"embargo_{h}"] = _pbo(rank_partitions(is_e, oos_e)[2], len(cols))
        entry["pbo"] = pbo
        if family == "F144":
            entry["acf"] = {str(lag): _acf(x, lag) for lag in params["acf_lags"]}
        out[family] = entry
    return out


# ------------------------------------------------------------------------------ agregação
def implied_n(target_factor: float) -> float | None:
    """N com f(N) = alvo; None se o alvo está abaixo de f(2) (fora do domínio, DE-04.c)."""
    if target_factor < float(expected_max_factor(MIN_TRIALS)):
        return None
    return float(
        optimize.brentq(
            lambda n: float(expected_max_factor(n)) - target_factor, MIN_TRIALS, 1e15, xtol=1e-12
        )
    )


def _mean_se(values: FloatArray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / math.sqrt(values.size)),
    }


def aggregate(records: Sequence[Mapping[str, Any]], estimators: Sequence[str]) -> dict[str, Any]:
    """Resumo por família de um cenário: E[máx], variâncias, pares (N̂, V), PBO."""
    n_obs = int(records[0]["n_obs"])
    v_h0 = 1.0 / (n_obs - 1)
    out: dict[str, Any] = {}
    for family in FAMILIES:
        rows = [r[family] for r in records]
        maxima = np.array([r["max_sharpe"] for r in rows])
        sharpe = np.array([r["sharpe"] for r in rows])  # (R, M)
        v_cross = np.array([r["v_cross"] for r in rows])
        v_marg = float(np.mean(np.var(sharpe, axis=0, ddof=1)))
        e_max_stats = _mean_se(maxima)
        e_max, e_max_se = e_max_stats["mean"], e_max_stats["se"]
        pairs: dict[str, Any] = {}
        for name in estimators:
            n_hat = np.array([r["n_hat"][name] for r in rows])
            ok = n_hat >= MIN_TRIALS
            entry: dict[str, Any] = {
                "n_hat_mean": float(n_hat.mean()),
                "n_hat_sd": float(n_hat.std(ddof=1)),
                "below_domain_fraction": float(np.mean(~ok)),
            }
            if np.all(ok):
                factor = expected_max_factor(n_hat)
                h0 = np.full_like(v_cross, v_h0)
                for label, variance in (("v_cross", v_cross), ("v_h0", h0)):
                    pred = np.sqrt(variance) * factor
                    error = float(pred.mean() / e_max - 1.0)
                    # erro-padrão do erro relativo (método delta, pares por réplica)
                    resid = pred - (1.0 + error) * maxima
                    se = float(resid.std(ddof=1) / math.sqrt(resid.size) / abs(e_max))
                    entry[label] = {"rel_error": error, "se": se}
            pairs[name] = entry
        pbo = {key: _mean_se(np.array([r["pbo"][key] for r in rows])) for key in rows[0]["pbo"]}
        diffs = {}
        for baseline in ("independent_halves", "permuted_rows"):
            for key in rows[0]["pbo"]:
                if key in ("independent_halves", "permuted_rows"):
                    continue
                d = np.array([r["pbo"][key] - r["pbo"][baseline] for r in rows])
                diffs[f"{key}_minus_{baseline}"] = _mean_se(d)
        for key in rows[0]["pbo"]:
            if key.startswith("embargo_"):
                d = np.array([r["pbo"][key] - r["pbo"]["plain"] for r in rows])
                diffs[f"{key}_minus_plain"] = _mean_se(d)
        family_out: dict[str, Any] = {
            "n_configs": len(rows[0]["sharpe"]),
            "expected_max": e_max,
            "expected_max_se": e_max_se,
            "v_marginal": v_marg,
            "v_h0": v_h0,
            "v_cross_mean": float(v_cross.mean()),
            "implied_n_marginal": implied_n(e_max / math.sqrt(v_marg)),
            "corr_offdiag_mean": float(np.mean([r["corr_offdiag_mean"] for r in rows])),
            "pairs": pairs,
            "pbo_mean_se": pbo,
            "pbo_differences": diffs,
        }
        if family == "F144":
            family_out["acf_mean"] = {
                lag: float(np.mean([r["acf"][lag] for r in rows])) for lag in rows[0]["acf"]
            }
        out[family] = family_out
    return out


# --------------------------------------------------------------------- regras pré-registradas
def select_pair(
    grid: Mapping[str, Mapping[str, Any]], generic: Mapping[str, Any], estimators: Sequence[str]
) -> dict[str, Any]:
    """Regra de §4.2 com emendas 1 e 2 (§7.3): pares (N̂, V) admissíveis = nunca
    anticonservadores além de 2 erros-padrão; principal = menor pior |erro|; empate dentro de
    2 erros-padrão → mais estruturas com erro ≥ 0. Sem admissível → em aberto."""
    table: dict[str, list[tuple[str, float, float]]] = {}
    for scenario, families in grid.items():
        for family, family_summary in families.items():
            for name in estimators:
                for label in ("v_cross", "v_h0"):
                    cell = family_summary["pairs"][name].get(label)
                    key = f"{name}+{label}"
                    if cell is None:
                        table.setdefault(key, []).append((f"{scenario}/{family}", math.nan, 0.0))
                    else:
                        table.setdefault(key, []).append(
                            (f"{scenario}/{family}", cell["rel_error"], cell["se"])
                        )
    for structure, s in generic["structures"].items():
        se = s["expected_max_se"] / s["expected_max"]
        for name in estimators:
            e = s["estimators"][name]
            if e.get("below_domain"):
                continue
            table[f"{name}+v_cross"].append(
                (f"generic/{structure}", e["sr0_rel_error_with_v_cross"], se)
            )
            table[f"{name}+v_h0"].append((f"generic/{structure}", e["sr0_rel_error"], se))
    summary: dict[str, Any] = {}
    for key, cells in table.items():
        undefined = [c for c in cells if math.isnan(c[1])]
        worst = max((abs(e) for _, e, _ in cells if not math.isnan(e)), default=math.nan)
        worst_se = max((se for _, e, se in cells if not math.isnan(e)), default=math.nan)
        anti = [name for name, e, se in cells if not math.isnan(e) and e < -2.0 * se]
        summary[key] = {
            "worst_abs_error": worst,
            "worst_se": worst_se,
            "n_structures": len(cells),
            "n_conservative": sum(1 for _, e, _ in cells if not math.isnan(e) and e >= 0.0),
            "anti_conservative_in": anti,
            "undefined_in": [c[0] for c in undefined],
            "admissible": not anti and not undefined,
        }
    admissible = {k: v for k, v in summary.items() if v["admissible"]}
    if not admissible:
        return {"pairs": summary, "selected": None, "reason": "nenhum par admissível"}
    best = min(admissible.values(), key=lambda v: v["worst_abs_error"])
    tied = [
        k
        for k, v in admissible.items()
        if v["worst_abs_error"] - best["worst_abs_error"]
        <= 2.0 * max(v["worst_se"], best["worst_se"])
    ]
    tied.sort(key=lambda k: (-admissible[k]["n_conservative"], admissible[k]["worst_abs_error"]))
    return {"pairs": summary, "selected": tied[0], "tied": tied}


def embargo_decision(grid: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Regra de §7.4 (emenda 2b): adotar só se o efeito da estrutura temporal for detectável
    (|E[φ − φ_permutado]| > 2 erros-padrão) **e** algum embargo reduzir
    |E[φ_emb − φ_permutado]| em mais de 2 erros-padrão."""
    findings = []
    adopt = False
    for scenario, families in grid.items():
        for family, summary in families.items():
            diffs = summary["pbo_differences"]
            leak = diffs["plain_minus_permuted_rows"]["mean"]
            leak_se = diffs["plain_minus_permuted_rows"]["se"]
            detectable = abs(leak) > 2.0 * leak_se
            improvements = {}
            for key, stats in diffs.items():
                if key.startswith("embargo_") and key.endswith("_minus_permuted_rows"):
                    improvements[key] = (abs(leak) - abs(stats["mean"]), stats["se"])
            helps = any(gain > 2.0 * se for gain, se in improvements.values())
            adopt = adopt or (detectable and helps)
            findings.append(
                {
                    "structure": f"{scenario}/{family}",
                    "complementarity_effect": diffs["plain_minus_independent_halves"],
                    "leak": leak,
                    "leak_se": leak_se,
                    "detectable": detectable,
                    "embargo_helps": helps,
                }
            )
    return {"adopt_embargo": adopt, "findings": findings}


# ------------------------------------------------------------------------------ execução
def run_study(
    params: Mapping[str, Any], root_seed: int, *, log: Any = None
) -> dict[str, JsonValue]:
    estimators = list(params["estimators"])
    replicates: dict[str, JsonValue] = {}
    summaries: dict[str, JsonValue] = {}
    for index, scenario in enumerate(sorted(params["scenarios"])):
        records: list[dict[str, JsonValue]] = []
        for rep in range(int(params["replicates"])):
            started = time.perf_counter()
            records.append(replicate_metrics(params, scenario, index, rep, root_seed))
            if log is not None:
                log(f"{scenario} rep {rep}: {time.perf_counter() - started:.1f}s")
        summaries[scenario] = json.loads(json.dumps(aggregate(records, estimators)))
        replicates[scenario] = [dict(r) for r in records]
    return {"summaries": summaries, "replicates": replicates}


def execute(
    params: Mapping[str, Any],
    *,
    project_root: Path,
    ledger_path: Path,
    artifacts_root: Path,
    generic_reference: Mapping[str, Any],
    root_seed: int | None = None,
    log: Any = None,
) -> tuple[str, ArtifactRef, dict[str, Any]]:
    spec = RunSpec(
        experiment_id="sim-grid-neff-embargo",
        run_kind=RunKind.SYNTHETIC,
        parameters={**params, "generic_reference": dict(generic_reference["reference"])},
        random_seeds={"root": new_root_seed() if root_seed is None else root_seed},
        input_data_references=(
            InputReference(
                InputKind.SYNTHETIC,
                "generator://factor-model(mean=r_f)",
                "cenários de experiments.momentum_scenarios; conjuntos A e B por réplica",
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
        study = run_study(params, spec.random_seeds["root"], log=log)
        summaries: Any = study["summaries"]
        decisions: dict[str, Any] = {
            "pair_selection": select_pair(
                summaries, generic_reference["result"], list(params["estimators"])
            ),
            "embargo": embargo_decision(summaries),
        }
        result = {**study, "decisions": decisions}
        ref = ArtifactStore(artifacts_root).put_bytes(
            canonical_bytes(result),
            suffix=".json",
            media_type="application/json",
            description="sim-grid-neff-embargo: réplicas, resumos e aplicação das regras",
        )
        run.set_results([ref])
    return run.run_id, ref, result


def load_generic_reference(artifacts_root: Path, sha256: str, run_id: str) -> dict[str, Any]:
    """Resultado de sim-neff-preliminary (estruturas genéricas), conferido pelo hash."""
    path = artifacts_root / sha256[:2] / f"{sha256}.json"
    data = path.read_bytes()
    if sha256_hex(data) != sha256:
        raise ContractViolationError(f"artefato genérico corrompido: {path}")
    return {"reference": {"run_id": run_id, "sha256": sha256}, "result": json.loads(data)}


GENERIC_RUN = "run-de53b36114814d0398437a14fd8d100f"
GENERIC_SHA256 = "608be6f5fe181768e8a29f7af4d4118d858231f92f897b777317c9303c93a9da"


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    settings = load_settings()
    generic = load_generic_reference(settings.paths.artifacts, GENERIC_SHA256, GENERIC_RUN)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    run_id, ref, result = execute(
        PARAMETERS,
        project_root=settings.paths.root,
        ledger_path=settings.paths.ledger,
        artifacts_root=settings.paths.artifacts,
        generic_reference=generic,
        log=log,
    )
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"run_id": run_id, "artifact": ref.to_json()}, ensure_ascii=False))
    print(
        json.dumps(
            {"summaries": result["summaries"], "decisions": result["decisions"]},
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
