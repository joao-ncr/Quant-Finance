"""Gera o manifesto congelado de DE-07 (Etapa B3) a partir da calibração registrada.

Entradas: ``run_id`` da calibração (o artefato é localizado e conferido pelo ledger e pelo
SHA-256), o protocolo (SHA-256 do arquivo), o commit do código e a matriz de células
definida em :data:`MATRIX`. Saídas: ``de07_signal_truth_v<n>.json`` (SR verdadeiros
congelados) e ``de07_confirmatory_v<n>.toml``. Com as mesmas entradas, inclusive as
sementes, os bytes são idênticos.

Matriz (protocolo §7.5): ver os ``justification`` de cada célula e :data:`EXCLUDED`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from frente_e.errors import ContractViolationError
from frente_e.experiments import momentum_scenarios
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import canonical_bytes
from frente_e.experiments.de07_confirmatory import PAIRS, V_LABELS
from frente_e.experiments.de07_manifest import load_manifest, sha256_file, to_toml
from frente_e.experiments.ledger import verify_ledger
from frente_e.experiments.neff_study import STRUCTURES
from frente_e.experiments.provenance import software_environment
from frente_e.experiments.seeds import new_root_seed
from frente_e.settings import load_settings

# custo sequencial por réplica (s), medido na Etapa B2 (protocolo §8.2.1), e sobrecarga
# por célula (início do processo, importações, criação do pool), estimada com folga
SECONDS_F144_FULL = 5.52
SECONDS_F16_FULL = 0.79
SECONDS_GAUSSIAN_CELL = 0.5
OVERHEAD_PER_CELL = 10.0
PARALLEL_SPEEDUP = 1.5  # medido 1,58 (F144, regime estável); arredondado para baixo
N_WORKERS = 2
MAX_HOURS = 2.0
SIGNAL_DELTA_GAUSSIAN = 2.5

EXTREME = {
    f"equi_{rho:g}_m{m}": {"kind": "equicorrelation", "m": m, "rho": rho}
    for rho in (0.95, 0.98, 0.99)
    for m in (4, 16, 144)
}

GAUSSIAN_METRICS = ["dsr_rejection_18_pairs", "true_discovery", "z_expected_max_error", "n_hat"]
MOMENTUM_METRICS = [
    "dsr_18_pairs",
    "psr_j06_k06",
    "pbo_cscv_s16",
    "n_hat",
    "v_candidates",
    "rho_bar",
]
SIGNAL_METRICS = [*MOMENTUM_METRICS, "true_discovery", "pbo_true", "overfit_true"]

EXCLUDED: list[dict[str, str]] = [
    {
        "cell": "H₁ (qualquer) com F144",
        "reason": "custa ≈ 1840 s de relógio por célula (500 × 5,52 s / 1,5) e levaria o total a "
        "≈ 2,3 h; poder e PBO sob sinal ficam medidos em F4 e F16 (limitação declarada)",
    },
    {
        "cell": "P0-low_common e P0-heterogeneous (F4/F16)",
        "reason": "regimes de correlação 0,93–0,97 já cobertos pelas células gaussianas "
        "equi 0,95 e pela P0-base; a mecânica DE-01 é a mesma; +≈ 530 s cada, sem folga",
    },
    {
        "cell": "P0-vol_clustering com F144",
        "reason": "+≈ 3150 s; o objetivo da célula (PSR sob aglomeração de volatilidade e "
        "nível do DSR com outra estrutura temporal) é atendido com F4/F16",
    },
    {
        "cell": "seis células low_common com sinal",
        "reason": "excluídas pela decisão D5 (≈ 4 h adicionais)",
    },
    {
        "cell": "padrão gaussiano 'decile' com M = 4",
        "reason": "⌈0,1·4⌉ = 1: idêntico ao padrão 'single' (redundante)",
    },
]


def _gaussian_cells(n_obs: int) -> list[dict[str, Any]]:
    structures: dict[str, Any] = {**STRUCTURES, **EXTREME}
    cells: list[dict[str, Any]] = []
    for name, spec in structures.items():
        cells.append(
            {
                "id": f"G0-{name}",
                "level": "gaussian",
                "hypothesis": "H0",
                "replicates": 20_000,
                "metrics": GAUSSIAN_METRICS,
                "justification": "nível do DSR (2a) e erro de E[máx] numa estrutura de "
                "dependência; nível Z, N̂ sobre a C verdadeira",
                "estimated_seconds": SECONDS_GAUSSIAN_CELL + OVERHEAD_PER_CELL,
                "structure": dict(spec),
                "n_obs": n_obs,
                "signal_pattern": "none",
                "signal_delta": 0.0,
            }
        )
    for name, spec in structures.items():
        m = _size(spec)
        for pattern in ("single", "decile"):
            if pattern == "decile" and math.ceil(0.1 * m) == 1:
                continue
            cells.append(
                {
                    "id": f"G1-{name}-{pattern}",
                    "level": "gaussian",
                    "hypothesis": "H1-gauss",
                    "replicates": 20_000,
                    "metrics": GAUSSIAN_METRICS,
                    "justification": "poder complementar (não decisório) do DSR com sinal "
                    f"δ = 2,5 em '{pattern}' nesta estrutura",
                    "estimated_seconds": SECONDS_GAUSSIAN_CELL + OVERHEAD_PER_CELL,
                    "structure": dict(spec),
                    "n_obs": n_obs,
                    "signal_pattern": pattern,
                    "signal_delta": SIGNAL_DELTA_GAUSSIAN,
                }
            )
    return cells


def _size(spec: Mapping[str, Any]) -> int:
    if "m" in spec:
        return int(spec["m"])
    if spec["kind"] == "blocks":
        return int(spec["n_blocks"]) * int(spec["block_size"])
    if spec["kind"] == "grid":
        return int(spec["rows"]) * int(spec["cols"])
    raise ContractViolationError(f"estrutura sem tamanho: {spec['kind']}")


def _momentum_cells(calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    small = ["F4", "F16"]
    cells: list[dict[str, Any]] = [
        {
            "id": "P0-base",
            "level": "momentum",
            "hypothesis": "H0",
            "replicates": 1000,
            "metrics": MOMENTUM_METRICS,
            "justification": "nível do DSR (2a) na estrutura real da grade, com N = 4, 16, 144; "
            "PSR (2b); distribuição nula da PBO por família; referência nula das AUC; N̂ e V",
            "estimated_seconds": 1000 * SECONDS_F144_FULL + OVERHEAD_PER_CELL,
            "scenario": "base",
            "families": ["F4", "F16", "F144"],
            "cscv_families": ["F4", "F16", "F144"],
        },
        {
            "id": "P0-vol_clustering",
            "level": "momentum",
            "hypothesis": "H0",
            "replicates": 1000,
            "metrics": MOMENTUM_METRICS,
            "justification": "PSR (2b) e nível do DSR sob aglomeração de volatilidade (curtose, "
            "dependência temporal na variância); PBO nula com outra estrutura temporal",
            "estimated_seconds": 1000 * SECONDS_F16_FULL + OVERHEAD_PER_CELL,
            "scenario": "vol_clustering",
            "families": small,
            "cscv_families": small,
        },
    ]
    for half_life in (63, 252):
        for delta in (1, 2.5, 4):
            key = f"mom_h{half_life}_d{delta:g}"
            entry = calibration["cells"][key]
            cells.append(
                {
                    "id": f"P1-mom-h{half_life}-d{delta:g}",
                    "level": "momentum",
                    "hypothesis": "H1-mom",
                    "replicates": 500,
                    "metrics": SIGNAL_METRICS,
                    "justification": f"poder (2c) com δ* = {delta:g} e meia-vida {half_life}; "
                    "PBO sob sinal, AUC contra P0-base e PBO verdadeira"
                    + ("; célula de poder da regra D3" if delta == 2.5 else ""),
                    "estimated_seconds": 500 * SECONDS_F16_FULL + OVERHEAD_PER_CELL,
                    "scenario": "base",
                    "families": small,
                    "cscv_families": small,
                    "signal": dict(entry["signal"]),
                    "truth_key": key,
                }
            )
    entry = calibration["cells"]["com_d2.5"]
    cells.append(
        {
            "id": "P1-com-d2.5",
            "level": "momentum",
            "hypothesis": "H1-com",
            "replicates": 500,
            "metrics": SIGNAL_METRICS,
            "justification": "prêmio de mercado sem momentum (P8): quanto o DSR rejeita numa "
            "família só comprada sem sinal transversal",
            "estimated_seconds": 500 * SECONDS_F16_FULL + OVERHEAD_PER_CELL,
            "scenario": "base",
            "families": small,
            "cscv_families": small,
            "signal": dict(entry["signal"]),
            "truth_key": "com_d2.5",
        }
    )
    return cells


def _scenario(name: str) -> dict[str, Any]:
    spec: Any = momentum_scenarios.SCENARIOS[name]
    return dict(spec)


def calibration_inputs(
    ledger_path: Path, artifacts_root: Path, run_id: str
) -> tuple[dict[str, Any], int, str]:
    """(resultado, semente raiz, SHA-256 do artefato) da calibração concluída ``run_id``."""
    verify_ledger(ledger_path)
    records = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines() if x]
    started = [r for r in records if r["run_id"] == run_id and r["status"] == "started"]
    done = [r for r in records if r["run_id"] == run_id and r["status"] == "completed"]
    if len(started) != 1 or len(done) != 1:
        raise ContractViolationError(f"{run_id}: calibração não encontrada ou não concluída")
    if started[0]["parameters"].get("phase") != "calibration":
        raise ContractViolationError(f"{run_id}: não é uma execução de calibração")
    ref = ArtifactRef.from_json(done[0]["results_reference"][0])
    store = ArtifactStore(artifacts_root)
    store.verify(ref)
    result = json.loads(store.path_of(ref).read_bytes())
    return result, int(started[0]["random_seeds"]["root"]), ref.sha256


def build(
    *,
    project_root: Path,
    calibration: Mapping[str, Any],
    calibration_run_id: str,
    calibration_seed: int,
    calibration_sha256: str,
    code_commit: str,
    seeds: Mapping[str, int],
    version: int,
    out_dir: Path,
    protocol_path: str = "docs/DE07_PROTOCOL.md",
) -> tuple[Path, Path]:
    statuses = {k: v["calibration"]["status"] for k, v in calibration["cells"].items()}
    pending = {k: v for k, v in statuses.items() if v != "calibrada"}
    if pending:
        raise ContractViolationError(
            f"células não calibradas (emenda 5): {pending}; o manifesto não pode ser congelado"
        )
    truth_rel = (out_dir / f"de07_signal_truth_v{version}.json").relative_to(project_root)
    truth = {
        "calibration_run_id": calibration_run_id,
        "calibration_artifact_sha256": calibration_sha256,
        "cells": {k: dict(v) for k, v in calibration["cells"].items()},
    }
    (project_root / truth_rel).write_bytes(canonical_bytes(truth) + b"\n")
    n_obs = {int(v["truth"]["n_rows"]) for v in calibration["cells"].values()}
    if len(n_obs) != 1:
        raise ContractViolationError("calibração com calendários diferentes")
    cells = [*_gaussian_cells(n_obs.pop()), *_momentum_cells(calibration)]
    for order, cell in enumerate(cells, start=1):
        cell["order"] = order
    env = software_environment(project_root)
    packages: Any = env["packages"]
    estimated = sum(
        float(c["estimated_seconds"]) / (PARALLEL_SPEEDUP if c["level"] == "momentum" else 1.0)
        for c in cells
    )
    data = {
        "manifest": {
            "name": "de07-confirmatory",
            "version": version,
            "phase": "confirmatory",
            "created": "2026-11-06",
            "protocol_path": protocol_path,
            "protocol_version": 1,
            "protocol_sha256": sha256_file(project_root / protocol_path),
            "truth_path": truth_rel.as_posix(),
            "truth_sha256": sha256_file(project_root / truth_rel),
            "calibration_run_id": calibration_run_id,
            "calibration_artifact_sha256": calibration_sha256,
            "code_commit": code_commit,
            "uv_lock_sha256": str(env["uv_lock_sha256"]),
            "python": str(env["python"]),
            "numpy": str(packages["numpy"]),
            "scipy": str(packages["scipy"]),
            "execution_order": [c["id"] for c in cells],
            "notes": "preparação confirmatória congelada na Etapa B3; a Etapa C exige "
            "autorização explícita do desenvolvedor (--etapa-c-autorizada). Exclusões: "
            + " | ".join(f"{e['cell']}: {e['reason']}" for e in EXCLUDED),
        },
        "seeds": {
            "gaussian": int(seeds["gaussian"]),
            "momentum": int(seeds["momentum"]),
            "calibration": int(calibration_seed),
        },
        "budget": {
            "max_hours": MAX_HOURS,
            "n_workers": N_WORKERS,
            "parallel_speedup": PARALLEL_SPEEDUP,
            "seconds_per_replicate": {
                "momentum_F4_F16_F144_cscv": SECONDS_F144_FULL,
                "momentum_F4_F16_cscv": SECONDS_F16_FULL,
                "gaussian_cell_20000": SECONDS_GAUSSIAN_CELL,
                "overhead_per_cell": OVERHEAD_PER_CELL,
            },
            "estimated_hours": round(estimated / 3600.0, 3),
            "basis": "medições da Etapa B2 (protocolo §8.2.1): sequencial por réplica; "
            "momentum em 2 processos com ganho 1,5 (medido 1,58); gaussiano sequencial",
        },
        "evaluation": {
            "alpha": 0.05,
            "fwer": 0.05,
            "ci_level": 0.95,
            "psr_config": "j06_k06",
            "n_blocks": 16,
            "pairs": list(PAIRS),
            "v_candidates": list(V_LABELS),
            "power_cells": ["P1-mom-h63-d2.5", "P1-mom-h252-d2.5"],
            "pbo_quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            "pbo_tail_points": [0.5, 0.75, 0.9],
            "decision_rule": "D3: domínio (N̂ ≥ 2 em todas as réplicas) → validade (2a: "
            "binomial exato unilateral H0 nível ≤ 0,05, Holm por par, FWER 5%) → maior poder "
            "mínimo nas power_cells (F4 e F16); empate ≤ 2 EP pareados → poder médio → "
            "menor máx |z_E| gaussiano; sem admissível, nenhum (de07_evaluation.evaluate_study)",
        },
        "common": dict(momentum_scenarios.COMMON),
        "scenarios": {name: _scenario(name) for name in ("base", "vol_clustering")},
        "cells": cells,
    }
    manifest_rel = (out_dir / f"de07_confirmatory_v{version}.toml").relative_to(project_root)
    (project_root / manifest_rel).write_text(to_toml(data), encoding="utf-8", newline="\n")
    load_manifest(manifest_rel, project_root=project_root)  # verificação imediata
    return project_root / manifest_rel, project_root / truth_rel


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration-run", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--gaussian-seed", type=int, default=None)
    parser.add_argument("--momentum-seed", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    result, cal_seed, cal_sha = calibration_inputs(
        settings.paths.ledger, settings.paths.artifacts, args.calibration_run
    )
    seeds = {
        "gaussian": args.gaussian_seed if args.gaussian_seed is not None else new_root_seed(),
        "momentum": args.momentum_seed if args.momentum_seed is not None else new_root_seed(),
    }
    manifest, truth = build(
        project_root=settings.paths.root,
        calibration=result,
        calibration_run_id=args.calibration_run,
        calibration_seed=cal_seed,
        calibration_sha256=cal_sha,
        code_commit=args.code_commit,
        seeds=seeds,
        version=args.version,
        out_dir=settings.paths.root / "configs" / "studies",
    )
    print(json.dumps({"manifest": str(manifest), "truth": str(truth), "seeds": seeds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
