"""Executor das células de DE-07 (protocolo v1 §4–§8). **Não roda nada sozinho.**

Uma célula do manifesto é executada por chamada e vira **uma** execução no ledger.

- ``phase = "development"``: ``RunKind.DEVELOPMENT``, qualquer R. Só para testar código;
  nunca é reportável.
- ``phase = "confirmatory"`` (D7): ``RunKind.SYNTHETIC`` com ``preregistration_reference``
  = protocolo (caminho, versão, SHA-256). Exige manifesto verificado, árvore Git limpa,
  código do executor idêntico ao commit congelado no manifesto e a flag explícita de
  autorização da Etapa C na linha de comando.

Métricas por réplica (nível momentum), por família:
- ŜR diário, g₁ e b₂ (DE-02) de todas as configurações; a escolhida é argmax ŜR;
- Ĉ, ρ̂̄ e os 6 estimadores de N efetivo sobre Ĉ;
- V ∈ {entre tentativas (ddof = 1), V_H0 = 1/(T−1), V_ρ = V_H0·(1 − ρ̂̄)} (V_ρ: candidato
  experimental, D3);
- DSR dos 18 pares (N, V), com os momentos da escolhida; ``None`` se N̂ < 2 (DE-04.c);
- PSR(0) da configuração pré-escolhida (critério 2b), se pertencer à família;
- PBO (CSCV, S do manifesto) e, com sinal, a PBO verdadeira: fração das partições em que a
  escolhida IS tem SR verdadeiro ≤ mediana dos SR verdadeiros da família (D6);
- sobreajuste verdadeiro da escolhida na amostra toda.

Nível gaussiano: ŜR = σ·Z + μ, Z ~ N(0, C), σ = 1/√(T−1), retornos normais (γ₃ = 0,
γ₄ = 3); N̂ sobre a C verdadeira; contagens agregadas por célula.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.core.cscv import block_partition_performance, rank_partitions
from frente_e.core.effective_trials import ESTIMATORS
from frente_e.core.metrics import sharpe_ratio, standardized_moments
from frente_e.core.momentum import momentum_family
from frente_e.core.sharpe_inference import (
    MIN_TRIALS,
    deflated_sharpe_ratio,
    expected_max_factor,
    probabilistic_sharpe_ratio,
)
from frente_e.errors import ContractViolationError, ReproducibilityError, ResourceBudgetError
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import JsonValue, canonical_bytes
from frente_e.experiments.cscv_input import trim_oldest_to_multiple
from frente_e.experiments.de07_calibration import PER_WORKER_BYTES
from frente_e.experiments.de07_design import map_ordered, signal_grid
from frente_e.experiments.de07_manifest import Manifest, load_manifest
from frente_e.experiments.identity import (
    InputKind,
    InputReference,
    PreregistrationReference,
    RunKind,
    RunSpec,
)
from frente_e.experiments.ledger import LedgerWriter
from frente_e.experiments.neff_study import build_correlation
from frente_e.experiments.provenance import code_revision, software_environment
from frente_e.experiments.seeds import generator_for
from frente_e.runtime import detect_resources, plan_workers
from frente_e.settings import load_settings

FloatArray = NDArray[np.float64]
EXPERIMENT_ID = "de07-confirmatory"
V_LABELS = ("v_cross", "v_h0", "v_rho")
PAIRS = tuple(f"{est}+{v}" for est in sorted(ESTIMATORS) for v in V_LABELS)
FROZEN_CODE_PATHS = ("src", "uv.lock", "pyproject.toml")
AUTHORIZATION_FLAG = "--etapa-c-autorizada"


# ------------------------------------------------------------------ peças comuns
def variances(sr: FloatArray, n_obs: int, rho_bar: float) -> dict[str, float]:
    v_h0 = 1.0 / (n_obs - 1)
    return {
        "v_cross": float(np.var(sr, ddof=1)),
        "v_h0": v_h0,
        "v_rho": v_h0 * (1.0 - rho_bar),
    }


def mean_offdiagonal(corr: FloatArray) -> float:
    m = corr.shape[0]
    return float((np.sum(corr) - np.trace(corr)) / (m * (m - 1)))


# ------------------------------------------------------------------ nível momentum
def momentum_replicate(args: tuple[Any, ...]) -> dict[str, JsonValue]:
    """Uma réplica de uma célula de momentum (função de topo: serializável p/ processos)."""
    cell, common, scenarios, truth, root_seed, cell_index, rep, n_blocks, psr_config = args
    families = list(cell["families"])
    largest = max(families, key=lambda f: len(momentum_family(f)))
    grid = signal_grid(
        scenarios[cell["scenario"]],
        common,
        cell.get("signal"),
        root_seed,
        (cell_index, rep),
        largest,
    ).excess_returns
    n_obs = grid.n_observations
    trimmed = trim_oldest_to_multiple(grid, n_blocks)[0] if cell["cscv_families"] else None
    out: dict[str, JsonValue] = {"n_obs": n_obs}
    for family in families:
        ids = [c.config_id for c in momentum_family(family)]
        cols = [grid.config_ids.index(i) for i in ids]
        x = grid.values[:, cols]
        sr = sharpe_ratio(x)
        g1, b2 = standardized_moments(x)
        best = int(np.argmax(sr))
        corr = np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)
        rho_bar = mean_offdiagonal(corr)
        n_hat = {name: float(ESTIMATORS[name](corr)) for name in sorted(ESTIMATORS)}
        v = variances(sr, n_obs, rho_bar)
        dsr: dict[str, JsonValue] = {}
        for pair in PAIRS:
            est, label = pair.split("+")
            if n_hat[est] < MIN_TRIALS:
                dsr[pair] = None
                continue
            dsr[pair] = float(
                deflated_sharpe_ratio(
                    sr[best],
                    n_obs,
                    g1[best],
                    b2[best],
                    n_trials=n_hat[est],
                    sharpe_variance=v[label],
                )
            )
        entry: dict[str, JsonValue] = {
            "max_sharpe": float(sr[best]),
            "best": ids[best],
            "skewness_best": float(g1[best]),
            "kurtosis_best": float(b2[best]),
            "rho_bar": rho_bar,
            "n_hat": dict(n_hat),
            "v": dict(v),
            "dsr": dsr,
            "psr": None,
            "pbo": None,
            "pbo_true": None,
            "overfit_true": None,
            "best_true_sharpe": None,
        }
        if psr_config in ids:
            c = ids.index(psr_config)
            entry["psr"] = float(probabilistic_sharpe_ratio(sr[c], 0.0, n_obs, g1[c], b2[c]))
        true_sr = None if truth is None else np.array([truth[i] for i in ids])
        if true_sr is not None:
            median = float(np.median(true_sr))
            entry["best_true_sharpe"] = float(true_sr[best])
            entry["overfit_true"] = bool(true_sr[best] <= median)
        if trimmed is not None and family in cell["cscv_families"]:
            _, perf_is, perf_oos = block_partition_performance(trimmed.values[:, cols], n_blocks)
            chosen, _, rank = rank_partitions(perf_is, perf_oos)
            entry["pbo"] = float(np.mean(2.0 * rank <= len(ids) + 1))
            if true_sr is not None:
                entry["pbo_true"] = float(np.mean(true_sr[chosen] <= float(np.median(true_sr))))
        out[family] = entry
    return out


# ------------------------------------------------------------------ nível gaussiano
def signal_pattern(pattern: str, m: int) -> FloatArray:
    """Vetor de sinal pré-especificado: nenhum, a primeira configuração, ou o primeiro
    décimo (⌈0,1·M⌉) das configurações."""
    out = np.zeros(m, dtype=np.float64)
    if pattern == "single":
        out[0] = 1.0
    elif pattern == "decile":
        out[: math.ceil(0.1 * m)] = 1.0
    elif pattern != "none":
        raise ContractViolationError(f"padrão de sinal desconhecido: {pattern!r}")
    return out


def gaussian_cell(
    cell: Mapping[str, Any], root_seed: int, cell_index: int, *, chunk: int = 5000
) -> dict[str, JsonValue]:
    corr = build_correlation(cell["structure"])
    m = corr.shape[0]
    n_obs = int(cell["n_obs"])
    sigma = 1.0 / math.sqrt(n_obs - 1)
    mu = float(cell["signal_delta"]) * sigma * signal_pattern(str(cell["signal_pattern"]), m)
    rho_bar = mean_offdiagonal(corr)
    n_hat = {name: float(ESTIMATORS[name](corr)) for name in sorted(ESTIMATORS)}
    lower = np.linalg.cholesky(corr)
    rng = generator_for(root_seed, cell_index)
    n_reps = int(cell["replicates"])
    defined = {p: n_hat[p.split("+")[0]] >= MIN_TRIALS for p in PAIRS}
    rejections = dict.fromkeys(PAIRS, 0)
    true_discoveries = dict.fromkeys(PAIRS, 0)
    gap_sum = dict.fromkeys(PAIRS, 0.0)
    gap_sq = dict.fromkeys(PAIRS, 0.0)
    max_sum = max_sq = v_cross_sum = 0.0
    done = 0
    while done < n_reps:
        n = min(chunk, n_reps - done)
        sr = sigma * (rng.standard_normal((n, m)) @ lower.T) + mu
        best = np.argmax(sr, axis=1)
        mx = sr[np.arange(n), best]
        vc = np.var(sr, axis=1, ddof=1)
        max_sum += float(mx.sum())
        max_sq += float(np.sum(mx**2))
        v_cross_sum += float(vc.sum())
        v_const = {"v_h0": 1.0 / (n_obs - 1), "v_rho": (1.0 - rho_bar) / (n_obs - 1)}
        for pair in PAIRS:
            if not defined[pair]:
                continue
            est, label = pair.split("+")
            variance = vc if label == "v_cross" else np.full(n, v_const[label])
            dsr = deflated_sharpe_ratio(
                mx, n_obs, 0.0, 3.0, n_trials=n_hat[est], sharpe_variance=variance
            )
            reject = dsr > 0.95
            rejections[pair] += int(reject.sum())
            true_discoveries[pair] += int(np.sum(reject & (mu[best] > 0.0)))
            gap = (np.sqrt(variance) * float(expected_max_factor(n_hat[est])) - mx) / sigma
            gap_sum[pair] += float(gap.sum())
            gap_sq[pair] += float(np.sum(gap**2))
        done += n
    pairs: dict[str, JsonValue] = {}
    for pair in PAIRS:
        if not defined[pair]:
            pairs[pair] = None
            continue
        mean_gap = gap_sum[pair] / n_reps
        sd_gap = math.sqrt(max(gap_sq[pair] / n_reps - mean_gap**2, 0.0) * n_reps / (n_reps - 1))
        pairs[pair] = {
            "rejections": rejections[pair],
            "true_discoveries": true_discoveries[pair],
            "z_expected_max_error": mean_gap,
            "z_expected_max_error_se": sd_gap / math.sqrt(n_reps),
        }
    mean_max = max_sum / n_reps
    return {
        "n_replicates": n_reps,
        "n_configs": m,
        "n_obs": n_obs,
        "rho_bar": rho_bar,
        "n_hat": dict(n_hat),
        "expected_max_over_sigma": mean_max / sigma,
        "expected_max_over_sigma_se": math.sqrt(
            max(max_sq / n_reps - mean_max**2, 0.0) / (n_reps - 1)
        )
        / sigma,
        "v_cross_mean": v_cross_sum / n_reps,
        "pairs": pairs,
    }


# ------------------------------------------------------------------ execução de uma célula
def _check_frozen_code(project_root: Path, commit: str) -> None:
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "diff",
            "--quiet",
            commit,
            "HEAD",
            "--",
            *FROZEN_CODE_PATHS,
        ],
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0:
        raise ReproducibilityError(
            f"código do executor mudou desde o commit congelado {commit}: nova versão do manifesto"
        )


def _check_budget_workers(manifest: Manifest, n_workers: int) -> None:
    """O custo estimado do manifesto supõe ``budget.n_workers`` processos. Com menos, o
    orçamento de tempo (D5) seria excedido em silêncio: erro explícito."""
    expected = int(manifest.budget["n_workers"])
    if n_workers != expected:
        raise ResourceBudgetError(
            f"o manifesto foi orçado com {expected} processos; recebido {n_workers}"
        )
    available = plan_workers(
        load_settings().runtime, detect_resources(), PER_WORKER_BYTES
    ).n_workers
    if available < expected:
        raise ResourceBudgetError(
            f"memória disponível permite {available} processo(s), o orçamento exige {expected}: "
            "libere memória ou decida explicitamente uma nova versão do orçamento"
        )


def run_cell(
    manifest: Manifest,
    cell_id: str,
    *,
    project_root: Path,
    ledger_path: Path,
    artifacts_root: Path,
    n_workers: int,
) -> tuple[str, ArtifactRef]:
    cell, cell_index = manifest.cell(cell_id)
    revision = code_revision(project_root)
    confirmatory = manifest.phase == "confirmatory"
    prereg = None
    if confirmatory:
        if revision.get("commit") is None or revision.get("dirty") is not False:
            raise ReproducibilityError("execução confirmatória exige commit e árvore limpa (D7)")
        _check_frozen_code(project_root, manifest.code_commit)
        _check_budget_workers(manifest, n_workers)
        prereg = PreregistrationReference(
            manifest.protocol_path, manifest.protocol_version, manifest.protocol_sha256
        )
    root_seed = manifest.seeds[str(cell["level"])]
    spec = RunSpec(
        experiment_id=EXPERIMENT_ID,
        run_kind=RunKind.SYNTHETIC if confirmatory else RunKind.DEVELOPMENT,
        parameters={
            "phase": manifest.phase,
            "manifest_path": manifest.path,
            "manifest_version": manifest.version,
            "manifest_sha256": manifest.sha256,
            "cell": dict(cell),
            "cell_index": cell_index,
            "evaluation": dict(manifest.evaluation),
            "common": dict(manifest.common),
        },
        random_seeds={"root": root_seed},
        preregistration_reference=prereg,
        input_data_references=(
            InputReference(
                InputKind.SYNTHETIC,
                f"generator://de07/{cell['level']}",
                f"célula {cell_id} do manifesto {manifest.path} v{manifest.version}",
            ),
        ),
    )
    with (
        LedgerWriter(ledger_path) as ledger,
        ledger.run(
            spec, software_environment=software_environment(project_root), code_revision=revision
        ) as run,
    ):
        result: dict[str, JsonValue] = {"cell": cell_id}
        if cell["level"] == "gaussian":
            result["summary"] = gaussian_cell(cell, root_seed, cell_index)
        else:
            truth = manifest.truth_for(cell)
            items = [
                (
                    cell,
                    dict(manifest.common),
                    dict(manifest.scenarios),
                    truth,
                    root_seed,
                    cell_index,
                    rep,
                    int(manifest.evaluation["n_blocks"]),
                    str(manifest.evaluation["psr_config"]),
                )
                for rep in range(int(cell["replicates"]))
            ]
            result["replicates"] = list(map_ordered(momentum_replicate, items, n_workers=n_workers))
        ref = ArtifactStore(artifacts_root).put_bytes(
            canonical_bytes(result),
            suffix=".json",
            media_type="application/json",
            description=f"{EXPERIMENT_ID}: célula {cell_id} ({manifest.phase})",
        )
        run.set_results([ref])
    return run.run_id, ref


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        AUTHORIZATION_FLAG,
        dest="authorized",
        action="store_true",
        help="obrigatória para a fase confirmatória (Etapa C, autorização do desenvolvedor)",
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    manifest = load_manifest(args.manifest, project_root=settings.paths.root)
    if manifest.phase == "confirmatory" and not args.authorized:
        raise ReproducibilityError(
            f"Etapa C bloqueada: a fase confirmatória exige {AUTHORIZATION_FLAG}"
        )
    run_id, ref = run_cell(
        manifest,
        args.cell,
        project_root=settings.paths.root,
        ledger_path=settings.paths.ledger,
        artifacts_root=settings.paths.artifacts,
        n_workers=args.workers,
    )
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"run_id": run_id, "artifact": ref.to_json()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
