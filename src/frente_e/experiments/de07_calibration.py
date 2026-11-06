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
   verdadeiros por configuração e o δ̂* verificado, com EP **jackknife por réplica**
   aplicado ao próprio estimador (máximo do SR agregado). Classificação (emenda 5,
   :func:`classify`): *inconclusiva* se 2·EP > metade da distância ao nível vizinho mais
   próximo em {0, 1, 2,5, 4}; *calibrada* se |δ̂* − alvo| ≤ 2·EP; senão *reprovada*, com
   um único refinamento pela inclinação local, em outro fluxo, e nova classificação.
   Nenhuma célula é descartada: todas saem com status, e a execução registra o
   resultado inteiro. Histórico: até a emenda 5, a tolerância era 0,25 absoluta, e a
   primeira reprovação interrompia a execução (``run-434c7484…``, falhou).
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
    "tolerance_rule": "emenda 5: |δ̂* − δ*| ≤ 2·EP (jackknife por réplica); inconclusiva se "
    "2·EP > metade da distância ao nível mais próximo em {0, 1, 2,5, 4}",
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
    ordem fixa), seu erro-padrão jackknife por réplica e T por réplica.

    **Unidade amostral independente: a réplica.** Linhas (pregões) e ativos de uma réplica
    são dependentes (fator comum, coortes sobrepostas, médias persistentes) e nunca são
    tratados como observações independentes: o erro-padrão vem do jackknife que retira
    uma réplica inteira de cada vez, aplicado ao **mesmo** estimador (o SR agregado).
    """
    true_sr, _, n_rows, jack = _pooled_with_jackknife(parts)
    return true_sr, jackknife_se(jack), n_rows


def _check_parts(parts: Sequence[tuple[int, FloatArray, FloatArray, FloatArray]]) -> int:
    if not parts:
        raise ContractViolationError("nenhuma réplica")
    n_rows = parts[0][0]
    if any(p[0] != n_rows for p in parts):
        raise ContractViolationError("réplicas com calendários diferentes")
    return n_rows


def _pooled_with_jackknife(
    parts: Sequence[tuple[int, FloatArray, FloatArray, FloatArray]],
) -> tuple[FloatArray, FloatArray, int, FloatArray]:
    """(SR agregado, SR de cada réplica, T, matriz R × C dos SR agregados sem a réplica r)."""
    n_rows = _check_parts(parts)
    n, mean, m2 = 0, np.zeros_like(parts[0][1]), np.zeros_like(parts[0][2])
    for count, part_mean, part_m2, _ in parts:
        total = n + count
        delta = part_mean - mean
        mean = mean + delta * (count / total)
        m2 = m2 + part_m2 + delta**2 * (n * count / total)
        n = total
    true_sr: FloatArray = mean / np.sqrt(m2 / (n - 1))
    jack = np.full((len(parts), true_sr.size), np.nan)
    if len(parts) > 1:
        for r, (count, part_mean, part_m2, _) in enumerate(parts):
            rest = n - count
            mean_r = (mean * n - part_mean * count) / rest
            # remove o grupo r da soma de quadrados centrada (inverso da combinação de grupos)
            m2_r = m2 - part_m2 - (part_mean - mean_r) ** 2 * (rest * count / n)
            jack[r] = mean_r / np.sqrt(m2_r / (rest - 1))
    per_rep = np.array([p[3] for p in parts])
    return true_sr, per_rep, n_rows, jack


def jackknife_se(leave_one_out: FloatArray) -> FloatArray:
    """√((R−1)/R · Σ (θ₍₋ᵣ₎ − θ̄)²) por coluna; NaN com R < 2."""
    values = np.asarray(leave_one_out, dtype=np.float64)
    r = values.shape[0]
    if r < 2:
        return np.full(values.shape[1:], np.nan)
    centered = values - values.mean(axis=0)
    out: FloatArray = np.sqrt((r - 1) / r * np.sum(centered**2, axis=0))
    return out


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
    if n_reps < 2:
        raise ContractViolationError("o erro-padrão por réplica exige ao menos 2 réplicas")
    scenario = params["scenarios"][params["scenario"]]
    family = str(params["family"])
    items = [
        (scenario, params["common"], signal, root_seed, (*prefix, rep), family)
        for rep in range(n_reps)
    ]
    parts = map_ordered(replicate_moments, items, n_workers=n_workers)
    true_sr, _, n_rows, jack = _pooled_with_jackknife(parts)
    se = jackknife_se(jack)
    # δ̂* = máx_c SR_c·√(T−1): o jackknife é aplicado ao máximo (inclui a seleção)
    delta_se = float(jackknife_se(jack.max(axis=1)[:, None])[0]) * math.sqrt(n_rows - 1)
    ids = [c.config_id for c in momentum_family(family)]
    best = int(np.argmax(true_sr))
    return {
        "config_ids": ids,
        "true_sharpe": [float(v) for v in true_sr],
        "true_sharpe_se": [float(v) for v in se],
        "n_rows": n_rows,
        "n_replicates": n_reps,
        "delta_star": float(true_sr[best] * math.sqrt(n_rows - 1)),
        "delta_star_se": delta_se,
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


# ------------------------------------------------------------------ classificação (emenda 5)
CALIBRATED = "calibrada"
REJECTED = "reprovada"
INCONCLUSIVE = "inconclusiva"


def precision_limit(target: float, levels: Sequence[float]) -> float:
    """Maior tolerância admissível para o alvo: metade da distância ao nível mais próximo
    entre ``levels`` (que inclui 0, a nula). Acima disso, um δ aceito poderia estar mais
    perto de outro nível do desenho, e os níveis deixariam de ser distinguíveis."""
    others = [abs(target - level) for level in levels if level != target]
    if not others:
        raise ContractViolationError("são necessários ao menos dois níveis")
    return 0.5 * min(others)


def classify(achieved: float, se: float, target: float, levels: Sequence[float]) -> dict[str, Any]:
    """Regra da emenda 5 (fixada antes da nova calibração), em unidades de δ:

    - tolerância = 2·EP (EP jackknife por réplica de δ̂*);
    - **inconclusiva** se 2·EP > ``precision_limit`` (precisão insuficiente para distinguir
      o alvo dos níveis vizinhos; não é aprovada só porque o EP é grande);
    - **calibrada** se |δ̂* − alvo| ≤ 2·EP; senão **reprovada**.
    """
    if not (math.isfinite(achieved) and math.isfinite(se) and se >= 0.0):
        raise ContractViolationError("δ̂* e EP devem ser finitos, EP >= 0")
    tolerance = 2.0 * se
    limit = precision_limit(target, levels)
    diff = abs(achieved - target)
    if tolerance > limit:
        status = INCONCLUSIVE
    elif diff <= tolerance:
        status = CALIBRATED
    else:
        status = REJECTED
    return {
        "target_delta": float(target),
        "achieved_delta": float(achieved),
        "delta_se": float(se),
        "tolerance": tolerance,
        "precision_limit": limit,
        "abs_difference": diff,
        "status": status,
        "inconclusive": status == INCONCLUSIVE,
    }


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
        levels = [0.0, *(float(t) for t in params["targets"])]
        for t_index, target in enumerate(params["targets"]):
            scale = interpolate_scale(table, float(target))
            truth = truth_table(
                params,
                _signal(half_life, scale),
                root_seed,
                (STAGE_ORACLE, h_index, t_index),
                int(params["oracle_replicates"]),
                n_workers=n_workers,
            )
            verdict = classify(truth["delta_star"], truth["delta_star_se"], target, levels)
            attempts = [{"scale": scale, "stage": "verification", **verdict}]
            if verdict["status"] == REJECTED:
                # um único refinamento, com fluxo independente; inconclusiva não é refinada
                # (mais uma verificação com o mesmo R não ganha precisão)
                slope = local_log_slope(table, scale)
                scale = scale * (float(target) / verdict["achieved_delta"]) ** (1.0 / slope)
                truth = truth_table(
                    params,
                    _signal(half_life, scale),
                    root_seed,
                    (STAGE_REFINE, h_index, t_index),
                    int(params["oracle_replicates"]),
                    n_workers=n_workers,
                )
                verdict = classify(truth["delta_star"], truth["delta_star_se"], target, levels)
                attempts.append({"scale": scale, "stage": "refinement", **verdict})
            say(
                f"h={half_life:g} alvo={target}: s={scale:.6g}, δ̂*={verdict['achieved_delta']:.3f}"
                f" ± {verdict['delta_se']:.3f} -> {verdict['status']}"
            )
            out_cells[f"mom_h{half_life:g}_d{target:g}"] = {
                "hypothesis": "H1-mom",
                "signal": _signal(half_life, scale),
                "calibration": verdict,
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
        "calibration": {
            "status": CALIBRATED,
            "method": "analítico: π = alvo·σ_EW/√(T−1), σ_EW exato; δ_EW alcançado = alvo",
            "target_delta_equal_weight": target,
            "achieved_delta_equal_weight": target,
            "equal_weight_volatility": sigma_ew,
            "f144_delta_star": truth["delta_star"],
            "f144_delta_star_se": truth["delta_star_se"],
            "inconclusive": False,
        },
        "n_rows": n_rows,
        "truth": truth,
    }
    statuses = {k: v["calibration"]["status"] for k, v in out_cells.items()}
    return {
        "grids": grids,
        "cells": out_cells,
        "statuses": statuses,
        "all_calibrated": all(v == CALIBRATED for v in statuses.values()),
    }


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
    parser.add_argument("--seed", type=int, default=None, help="semente raiz fixa (registrada)")
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
        root_seed=args.seed,
        log=log,
    )
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"run_id": run_id, "artifact": ref.to_json()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
