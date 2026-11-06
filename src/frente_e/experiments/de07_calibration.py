"""Calibração da intensidade do sinal de DE-07 (protocolo §7.2, D4), só pelo Sharpe verdadeiro.

**Sharpe verdadeiro** da configuração c: ``SR_c = E[e_c,t] / √Var(e_c,t)``, com e_c,t o
excesso diário da carteira DE-01 sobre o CDI, momentos **incondicionais** do processo
gerador estacionário (escala diária, ŜR na mesma unidade do PSR). Não há forma fechada;
é estimado por oráculo: média e variância **agregadas** (todas as linhas de R réplicas
independentes, combinação exata de grupos). Não é a média dos ŜR de cada réplica.

**Intensidade:** ``δ* = máx_{c ∈ F144} SR_c · √(T − 1)``, T = linhas do calendário comum.

**Procedimento (fixado antes de rodar):**
1. Para cada meia-vida h, avalia δ̂*(s) numa grade geométrica de escalas s (desvio
   estacionário diário das médias específicas), com ``grid_replicates`` réplicas e
   **números aleatórios comuns** entre escalas (mesmo ruído e mesmos choques ξ; muda só
   s). Se a grade não cobrir os alvos, ela é estendida por fatores de 2
   (até ``max_extensions`` vezes de cada lado).
2. δ̂*(s) precisa ser estritamente crescente na grade; caso contrário, erro (sem ajuste).
3. s(alvo) por interpolação linear de log δ̂* em log s.
4. Verificação com fluxo independente (``oracle_replicates`` réplicas): a tabela de SR
   verdadeiros por configuração e o δ̂* verificado. Aceita se |δ̂* − alvo| ≤ ``tolerance``.
   Senão, um único refinamento pela inclinação local em log-log e nova verificação com
   outro fluxo; se falhar de novo, erro explícito (a execução falha no ledger).
5. H₁-com: π = alvo·σ_EW/√(T − 1), com σ_EW exato (``equal_weight_volatility``); a tabela de
   SR verdadeiros vem do oráculo. Sob H₀ os SR verdadeiros são exatamente 0 (analítico).

**Proibido aqui** (testado): DSR, PSR estimado, PBO, CSCV, seleção de configurações e
estimadores de N efetivo. A calibração só olha momentos incondicionais do gerador.

Execução: ``python -m frente_e.experiments.de07_calibration`` (ledger, ``phase =
calibration``). Não é execução confirmatória.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.core.metrics import sharpe_ratio
from frente_e.core.momentum import momentum_family
from frente_e.errors import ContractViolationError
from frente_e.experiments import momentum_scenarios
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import JsonValue, canonical_bytes
from frente_e.experiments.de07_design import (
    equal_weight_volatility,
    map_ordered,
    signal_grid,
)
from frente_e.experiments.identity import InputKind, InputReference, RunKind, RunSpec
from frente_e.experiments.ledger import LedgerWriter
from frente_e.experiments.provenance import code_revision, software_environment
from frente_e.experiments.seeds import new_root_seed
from frente_e.runtime import detect_resources, plan_workers
from frente_e.settings import load_settings

FloatArray = NDArray[np.float64]
EXPERIMENT_ID = "de07-signal-calibration"

# prefixos de caminho dos fluxos (independência entre etapas da calibração)
STAGE_GRID = 0
STAGE_ORACLE = 1
STAGE_REFINE = 2
STAGE_PREMIUM = 3

PARAMETERS: dict[str, JsonValue] = {
    "protocol": "docs/DE07_PROTOCOL.md §7.2 (D4, aprovada em 2026-11-06)",
    "phase": "calibration",
    "scenario": "base",
    "scenarios": {"base": momentum_scenarios.SCENARIOS["base"]},
    "common": momentum_scenarios.COMMON,
    "family": "F144",
    "half_lives": [63.0, 252.0],
    "targets": [1.0, 2.5, 4.0],
    "premium_target": 2.5,
    "scale_grid": [0.00025, 0.0005, 0.001, 0.002, 0.004],
    "max_extensions": 4,
    "grid_replicates": 30,
    "oracle_replicates": 100,
    "tolerance": 0.25,
}

PER_WORKER_BYTES = 300 * 2**20
"""Pico de memória por processo: 247 MiB medidos numa réplica F144 completa (grade, CSCV
S = 16, estimadores; 2026-11-06, Etapa B2), arredondado para cima. Ver DE07_PROTOCOL §8.4."""


# ------------------------------------------------------------------ oráculo (por réplica)
def replicate_moments(args: tuple[Any, ...]) -> tuple[int, FloatArray, FloatArray, FloatArray]:
    """(n linhas, médias, M2 centrado, ŜR) por configuração de uma réplica."""
    scenario, common, signal, root_seed, path, family = args
    x = signal_grid(scenario, common, signal, root_seed, path, family).excess_returns.values
    mean = x.mean(axis=0)
    m2 = np.sum((x - mean) ** 2, axis=0)
    return int(x.shape[0]), mean, m2, sharpe_ratio(x)


def pooled_sharpe(
    parts: Sequence[tuple[int, FloatArray, FloatArray, FloatArray]],
) -> tuple[FloatArray, FloatArray, int]:
    """SR agregado (todas as linhas de todas as réplicas; combinação exata de grupos, em
    ordem fixa), erro-padrão (desvio dos ŜR por réplica / √R) e T por réplica."""
    if not parts:
        raise ContractViolationError("nenhuma réplica")
    n_rows = parts[0][0]
    if any(p[0] != n_rows for p in parts):
        raise ContractViolationError("réplicas com calendários diferentes")
    n, mean, m2 = 0, np.zeros_like(parts[0][1]), np.zeros_like(parts[0][2])
    for count, part_mean, part_m2, _ in parts:
        total = n + count
        delta = part_mean - mean
        mean = mean + delta * (count / total)
        m2 = m2 + part_m2 + delta**2 * (n * count / total)
        n = total
    true_sr: FloatArray = mean / np.sqrt(m2 / (n - 1))
    per_rep = np.array([p[3] for p in parts])
    se: FloatArray = (
        per_rep.std(axis=0, ddof=1) / math.sqrt(len(parts))
        if len(parts) > 1
        else np.full_like(true_sr, math.nan)
    )
    return true_sr, se, n_rows


def truth_table(
    params: Mapping[str, Any],
    signal: Mapping[str, Any] | None,
    root_seed: int,
    prefix: Sequence[int],
    n_reps: int,
    *,
    n_workers: int,
) -> dict[str, Any]:
    """SR verdadeiros (oráculo) da família, δ̂* e a configuração de maior SR verdadeiro."""
    scenario = params["scenarios"][params["scenario"]]
    family = str(params["family"])
    items = [
        (scenario, params["common"], signal, root_seed, (*prefix, rep), family)
        for rep in range(n_reps)
    ]
    parts = map_ordered(replicate_moments, items, n_workers=n_workers)
    true_sr, se, n_rows = pooled_sharpe(parts)
    ids = [c.config_id for c in momentum_family(family)]
    best = int(np.argmax(true_sr))
    return {
        "config_ids": ids,
        "true_sharpe": [float(v) for v in true_sr],
        "true_sharpe_se": [float(v) for v in se],
        "n_rows": n_rows,
        "n_replicates": n_reps,
        "delta_star": float(true_sr[best] * math.sqrt(n_rows - 1)),
        "delta_star_se": float(se[best] * math.sqrt(n_rows - 1)),
        "best_config": ids[best],
    }


# ------------------------------------------------------------------ interpolação (pura)
def interpolate_scale(table: Sequence[tuple[float, float]], target: float) -> float:
    """s com δ̂*(s) = alvo, por interpolação linear em (log s, log δ̂*).

    Exige δ̂* estritamente crescente em s e alvo dentro da faixa avaliada."""
    pts = sorted(table)
    scales = np.array([p[0] for p in pts])
    deltas = np.array([p[1] for p in pts])
    if np.any(deltas <= 0.0) or np.any(np.diff(deltas) <= 0.0):
        raise ContractViolationError(
            "δ̂*(s) não é positivo e estritamente crescente na grade: calibração interrompida"
        )
    if not deltas[0] <= target <= deltas[-1]:
        raise ContractViolationError(
            f"alvo {target} fora da faixa avaliada [{deltas[0]}, {deltas[-1]}]"
        )
    return float(np.exp(np.interp(math.log(target), np.log(deltas), np.log(scales))))


def local_log_slope(table: Sequence[tuple[float, float]], scale: float) -> float:
    """Inclinação d log δ̂* / d log s no segmento da grade que contém ``scale``.

    A escala vem da interpolação dentro da faixa avaliada; nas bordas, o arredondamento
    pode deixá-la um ulp fora, e ela fica presa ao segmento extremo (folga relativa 1e-9)."""
    pts = sorted(table)
    if len(pts) < 2:
        raise ContractViolationError("são necessários dois pontos na grade")
    lo, hi = pts[0][0], pts[-1][0]
    if not lo * (1.0 - 1e-9) <= scale <= hi * (1.0 + 1e-9):
        raise ContractViolationError("escala fora da grade avaliada")
    for (s0, d0), (s1, d1) in itertools.pairwise(pts):
        if scale <= s1 or s1 == hi:
            return math.log(d1 / d0) / math.log(s1 / s0)
    raise ContractViolationError("escala fora da grade avaliada")  # inalcançável


# ------------------------------------------------------------------ calibração
def _signal(half_life: float, scale: float, premium: float = 0.0) -> dict[str, float]:
    return {"half_life": float(half_life), "scale": float(scale), "premium": float(premium)}


def calibrate(
    params: Mapping[str, Any],
    root_seed: int,
    *,
    n_workers: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Procedimento completo do docstring do módulo. Devolve escalas, tabelas e verdades."""
    say = log or (lambda _msg: None)
    tolerance = float(params["tolerance"])
    out_cells: dict[str, Any] = {}
    grids: dict[str, Any] = {}
    for h_index, half_life in enumerate(params["half_lives"]):
        evaluated: dict[float, float] = {}

        def evaluate(
            scale: float,
            h: float = float(half_life),
            hi: int = h_index,
            cache: dict[float, float] = evaluated,
        ) -> float:
            if scale not in cache:
                table = truth_table(
                    params,
                    _signal(h, scale),
                    root_seed,
                    (STAGE_GRID, hi),
                    int(params["grid_replicates"]),
                    n_workers=n_workers,
                )
                cache[scale] = float(table["delta_star"])
                say(f"h={h:g} s={scale:.6g}: δ̂*={cache[scale]:.3f}")
            return cache[scale]

        for scale in params["scale_grid"]:
            evaluate(float(scale))
        for _ in range(int(params["max_extensions"])):
            low = min(evaluated)
            if evaluate(low) <= min(params["targets"]):
                break
            evaluate(low / 2.0)
        for _ in range(int(params["max_extensions"])):
            high = max(evaluated)
            if evaluate(high) >= max(params["targets"]):
                break
            evaluate(high * 2.0)
        table = sorted(evaluated.items())
        grids[f"h{half_life:g}"] = [[s, d] for s, d in table]
        for t_index, target in enumerate(params["targets"]):
            scale = interpolate_scale(table, float(target))
            prefix = (STAGE_ORACLE, h_index, t_index)
            truth = truth_table(
                params,
                _signal(half_life, scale),
                root_seed,
                prefix,
                int(params["oracle_replicates"]),
                n_workers=n_workers,
            )
            attempts = [{"scale": scale, "delta_star": truth["delta_star"]}]
            achieved = float(truth["delta_star"])
            if abs(achieved - target) > tolerance:
                slope = local_log_slope(table, scale)
                scale = scale * (float(target) / achieved) ** (1.0 / slope)
                truth = truth_table(
                    params,
                    _signal(half_life, scale),
                    root_seed,
                    (STAGE_REFINE, h_index, t_index),
                    int(params["oracle_replicates"]),
                    n_workers=n_workers,
                )
                attempts.append({"scale": scale, "delta_star": truth["delta_star"]})
                achieved = float(truth["delta_star"])
                if abs(achieved - target) > tolerance:
                    raise ContractViolationError(
                        f"calibração não convergiu: h={half_life}, alvo {target}, δ̂*={achieved}"
                    )
            say(f"h={half_life:g} alvo={target}: s={scale:.6g}, δ̂*={achieved:.3f}")
            out_cells[f"mom_h{half_life:g}_d{target:g}"] = {
                "hypothesis": "H1-mom",
                "signal": _signal(half_life, scale),
                "target_delta": float(target),
                "attempts": attempts,
                "truth": truth,
            }
    # H₁-com: π analítico, verdades pelo oráculo
    scenario = params["scenarios"][params["scenario"]]
    sigma_ew = equal_weight_volatility(scenario)
    n_rows = int(
        signal_grid(
            scenario, params["common"], None, root_seed, (STAGE_PREMIUM, 0), "F4"
        ).excess_returns.n_observations
    )
    target = float(params["premium_target"])
    premium = target * sigma_ew / math.sqrt(n_rows - 1)
    truth = truth_table(
        params,
        _signal(252.0, 0.0, premium),
        root_seed,
        (STAGE_PREMIUM, 1),
        int(params["oracle_replicates"]),
        n_workers=n_workers,
    )
    say(f"H1-com: π={premium:.6g}/dia, δ̂*(F144)={float(truth['delta_star']):.3f}")
    out_cells[f"com_d{target:g}"] = {
        "hypothesis": "H1-com",
        "signal": _signal(252.0, 0.0, premium),
        "target_delta_equal_weight": target,
        "equal_weight_volatility": sigma_ew,
        "n_rows": n_rows,
        "truth": truth,
    }
    return {"grids": grids, "cells": out_cells}


def execute(
    params: Mapping[str, Any],
    *,
    project_root: Path,
    ledger_path: Path,
    artifacts_root: Path,
    n_workers: int,
    root_seed: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[str, ArtifactRef, dict[str, Any]]:
    if params.get("phase") != "calibration":
        raise ContractViolationError("este executor só registra a fase 'calibration'")
    spec = RunSpec(
        experiment_id=EXPERIMENT_ID,
        run_kind=RunKind.SYNTHETIC,
        parameters=dict(params),
        random_seeds={"root": new_root_seed() if root_seed is None else root_seed},
        input_data_references=(
            InputReference(
                InputKind.SYNTHETIC,
                "generator://factor-model+persistent-specific-means",
                "calibração do sinal de DE-07 pelo Sharpe verdadeiro (oráculo)",
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
        result = calibrate(params, spec.random_seeds["root"], n_workers=n_workers, log=log)
        ref = ArtifactStore(artifacts_root).put_bytes(
            canonical_bytes(result),
            suffix=".json",
            media_type="application/json",
            description="de07-signal-calibration: grades δ̂*(s), escalas e SR verdadeiros",
        )
        run.set_results([ref])
    return run.run_id, ref, result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=None, help="padrão: plano de DA-07")
    args = parser.parse_args(argv)
    settings = load_settings()
    workers = (
        args.workers
        or plan_workers(settings.runtime, detect_resources(), PER_WORKER_BYTES).n_workers
    )

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log(f"processos: {workers}")
    run_id, ref, _ = execute(
        PARAMETERS,
        project_root=settings.paths.root,
        ledger_path=settings.paths.ledger,
        artifacts_root=settings.paths.artifacts,
        n_workers=workers,
        log=log,
    )
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"run_id": run_id, "artifact": ref.to_json()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
