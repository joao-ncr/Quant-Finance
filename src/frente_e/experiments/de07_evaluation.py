"""Avaliação das células de DE-07 (protocolo v1 §4–§6, D1, D3, D6). Fixada antes da Etapa C.

Recebe os artefatos das células (``de07_confirmatory``) e devolve tabelas e a aplicação da
regra de seleção de (N, V). Nada aqui ajusta parâmetros: só conta, testa e resume.

Regras (D3), por par (N, V):
1. **domínio:** N̂ < 2 em alguma réplica de alguma célula → inadmissível;
2. **validade (2a):** em cada célula nula (gaussiana e de momentum, por família), valor-p
   exato de H₀: nível ≤ alfa; Holm entre as células nulas do par (FWER); alguma rejeição →
   inadmissível;
3. **poder no pior caso:** entre os admissíveis, maior poder mínimo nas células de poder
   (``power_cells``: H₁-mom com δ* = 2,5), por família. Empate quando a diferença em
   relação ao líder, na célula em que o líder atinge o mínimo, é ≤ 2 erros-padrão pareados;
   desempate pelo poder médio em todas as células H₁-mom e depois pelo menor máx |z_E|
   nas células gaussianas nulas;
4. **sem admissível → nenhum par selecionado** (não há "menos ruim").
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from frente_e.core.mc_inference import (
    binomial_excess_pvalue,
    binomial_two_sided_pvalue,
    clopper_pearson,
    holm_reject,
    mann_whitney_auc,
    quantile_with_ci,
)
from frente_e.errors import ContractViolationError

JsonDict = dict[str, Any]


# ------------------------------------------------------------------ resumos de célula
def rate(k: int, n: int, level: float) -> JsonDict:
    lo, hi = clopper_pearson(k, n, level)
    return {"k": k, "n": n, "rate": k / n, "ci": [lo, hi]}


def momentum_family_summary(
    replicates: Sequence[Mapping[str, Any]],
    family: str,
    pairs: Sequence[str],
    ev: Mapping[str, Any],
) -> JsonDict:
    """Contagens e distribuições de uma família numa célula de momentum."""
    rows = [r[family] for r in replicates]
    n = len(rows)
    threshold = 1.0 - float(ev["alpha"])
    level = float(ev["ci_level"])
    out_pairs: JsonDict = {}
    for pair in pairs:
        values = [row["dsr"][pair] for row in rows]
        undefined = sum(v is None for v in values)
        reject = np.array([v is not None and v > threshold for v in values])
        true_sr = [row["best_true_sharpe"] for row in rows]
        true_disc = np.array(
            [bool(r) and (t is not None and t > 0.0) for r, t in zip(reject, true_sr, strict=True)]
        )
        out_pairs[pair] = {
            "undefined": undefined,
            "rejection": rate(int(reject.sum()), n, level),
            "true_discovery": rate(int(true_disc.sum()), n, level),
            "indicators": [bool(v) for v in true_disc]
            if any(t is not None for t in true_sr)
            else [bool(v) for v in reject],
        }
    psr = [row["psr"] for row in rows]
    summary: JsonDict = {
        "n": n,
        "pairs": out_pairs,
        "rho_bar_mean": float(np.mean([row["rho_bar"] for row in rows])),
        "max_sharpe_mean": float(np.mean([row["max_sharpe"] for row in rows])),
        "v_mean": {k: float(np.mean([row["v"][k] for row in rows])) for k in rows[0]["v"]},
        "n_hat": {
            name: {
                "mean": float(np.mean([row["n_hat"][name] for row in rows])),
                "sd": float(np.std([row["n_hat"][name] for row in rows], ddof=1)) if n > 1 else 0.0,
                "below_2": float(np.mean([row["n_hat"][name] < 2.0 for row in rows])),
            }
            for name in rows[0]["n_hat"]
        },
    }
    if all(p is not None for p in psr):
        k = int(np.sum(np.array(psr) > threshold))
        summary["psr"] = {
            **rate(k, n, level),
            "pvalue_two_sided": binomial_two_sided_pvalue(k, n, float(ev["alpha"])),
        }
    pbo = [row["pbo"] for row in rows]
    if all(p is not None for p in pbo):
        summary["pbo"] = pbo_distribution(pbo, ev)
        summary["pbo_values"] = [float(p) for p in pbo]
        pbo_true = [row["pbo_true"] for row in rows]
        if all(p is not None for p in pbo_true):
            summary["pbo_vs_truth"] = paired_mean_difference(pbo, pbo_true)
    overfit = [row["overfit_true"] for row in rows]
    if all(o is not None for o in overfit):
        summary["overfit_true"] = rate(int(sum(bool(o) for o in overfit)), n, level)
    return summary


def pbo_distribution(values: Sequence[float], ev: Mapping[str, Any]) -> JsonDict:
    x = np.asarray(values, dtype=np.float64)
    level = float(ev["ci_level"])
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "quantiles": {
            str(q): list(quantile_with_ci(x, float(q), level)) for q in ev["pbo_quantiles"]
        },
        "tail": {str(t): float(np.mean(x >= float(t))) for t in ev["pbo_tail_points"]},
    }


def paired_mean_difference(a: Sequence[float], b: Sequence[float]) -> JsonDict:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else math.nan
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff": float(d.mean()),
        "se": se,
    }


def gaussian_summary(summary: Mapping[str, Any], ev: Mapping[str, Any]) -> JsonDict:
    n = int(summary["n_replicates"])
    level = float(ev["ci_level"])
    pairs: JsonDict = {}
    for pair, cell in summary["pairs"].items():
        if cell is None:
            pairs[pair] = None
            continue
        pairs[pair] = {
            "rejection": rate(int(cell["rejections"]), n, level),
            "true_discovery": rate(int(cell["true_discoveries"]), n, level),
            "z_expected_max_error": cell["z_expected_max_error"],
            "z_expected_max_error_se": cell["z_expected_max_error_se"],
        }
    return {"n": n, "pairs": pairs, "n_hat": summary["n_hat"], "rho_bar": summary["rho_bar"]}


# ------------------------------------------------------------------ regra D3
def select_pair(
    cells: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, str],
    pairs: Sequence[str],
    ev: Mapping[str, Any],
) -> JsonDict:
    """Aplica D3. ``cells[cell_id][family]`` são resumos (``momentum_family_summary``) e
    ``cells[cell_id]["gaussian"]`` resumos gaussianos; ``hypotheses[cell_id]`` ∈ {H0, ...}."""
    alpha, fwer = float(ev["alpha"]), float(ev["fwer"])
    power_cells = set(ev["power_cells"])
    report: JsonDict = {}
    for pair in pairs:
        undefined_in: list[str] = []
        null_tests: list[tuple[str, float]] = []
        power: list[tuple[str, float, list[bool]]] = []
        all_h1_power: list[float] = []
        z_gauss: list[float] = []
        for cell_id, families in cells.items():
            for family, s in families.items():
                where = f"{cell_id}/{family}"
                entry = s["pairs"][pair]
                if entry is None or entry.get("undefined", 0) > 0:
                    undefined_in.append(where)
                    continue
                k, n = entry["rejection"]["k"], entry["rejection"]["n"]
                if hypotheses[cell_id] == "H0":
                    null_tests.append((where, binomial_excess_pvalue(k, n, alpha)))
                    if family == "gaussian":
                        z_gauss.append(abs(float(entry["z_expected_max_error"])))
                elif hypotheses[cell_id] == "H1-mom":
                    p = float(entry["true_discovery"]["rate"])
                    all_h1_power.append(p)
                    if cell_id in power_cells:
                        power.append((where, p, list(entry["indicators"])))
        rejects = holm_reject([p for _, p in null_tests], fwer) if null_tests else []
        violations = [w for (w, _), r in zip(null_tests, rejects, strict=True) if r]
        worst = min(power, key=lambda x: x[1]) if power else None
        report[pair] = {
            "undefined_in": undefined_in,
            "size_violations": violations,
            "n_null_tests": len(null_tests),
            "admissible": not undefined_in and not violations and worst is not None,
            "min_power": None if worst is None else worst[1],
            "min_power_cell": None if worst is None else worst[0],
            "mean_power": float(np.mean(all_h1_power)) if all_h1_power else None,
            "max_abs_z_gaussian_null": max(z_gauss) if z_gauss else None,
            "_power": {w: ind for w, _, ind in power},
        }
    admissible = [p for p in pairs if report[p]["admissible"]]
    if not admissible:
        _strip(report)
        return {"pairs": report, "selected": None, "reason": "nenhum par admissível (D3)"}
    leader = max(admissible, key=lambda name: report[name]["min_power"])
    lead_cell = report[leader]["min_power_cell"]
    lead_ind = np.array(report[leader]["_power"][lead_cell], dtype=np.float64)
    tied = []
    for name in admissible:
        other = np.array(report[name]["_power"][lead_cell], dtype=np.float64)
        diff = lead_ind - other
        se = float(diff.std(ddof=1) / math.sqrt(diff.size)) if diff.size > 1 else 0.0
        if report[leader]["min_power"] - report[name]["min_power"] <= 2.0 * se:
            tied.append(name)
    tied.sort(
        key=lambda name: (
            -(report[name]["mean_power"] or 0.0),
            report[name]["max_abs_z_gaussian_null"] or 0.0,
        )
    )
    _strip(report)
    return {"pairs": report, "selected": tied[0], "tied": tied}


def _strip(report: JsonDict) -> None:
    for entry in report.values():
        entry.pop("_power", None)


# ------------------------------------------------------------------ PSR (2b) e PBO (D6)
def psr_level_tests(psr_cells: Mapping[str, Mapping[str, Any]], fwer: float) -> JsonDict:
    """Critério 2b: nível do PSR(0) da configuração pré-escolhida por célula nula, teste
    bilateral exato, Bonferroni entre células."""
    if not psr_cells:
        raise ContractViolationError("nenhuma célula com PSR")
    m = len(psr_cells)
    return {
        cell: {**s, "bonferroni_reject": bool(s["pvalue_two_sided"] <= fwer / m)}
        for cell, s in psr_cells.items()
    }


def pbo_auc(signal_values: ArrayLike, null_values: ArrayLike) -> float:
    """P(φ_H₁ < φ_H₀) + ½·P(empate): separação nula × sinal, sem limiar (D6)."""
    return mann_whitney_auc(signal_values, null_values)


# ------------------------------------------------- estudo completo (pré-especificado)
def evaluate_study(
    cells: Sequence[Mapping[str, Any]],
    ev_config: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[str],
) -> JsonDict:
    """Avaliação completa, fixada antes da Etapa C.

    - resumos por célula e família;
    - **D3** sobre todas as células (nulas: gaussianas e momentum; poder: ``power_cells``);
    - **2b**: PSR(0) da configuração pré-escolhida, uma vez por célula de momentum H₀ (a
      série é a mesma em todas as famílias que a contêm; usa a primeira família listada);
    - **D6**: para cada célula com sinal e família com PBO, a AUC contra a célula H₀ de
      momentum com o **mesmo cenário e família**, e a comparação PBO × PBO verdadeira.
    """
    summaries: dict[str, JsonDict] = {}
    hypotheses: dict[str, str] = {}
    for cell in cells:
        cid = str(cell["id"])
        hypotheses[cid] = str(cell["hypothesis"])
        if cid not in results:
            raise ContractViolationError(f"resultado ausente da célula {cid!r}")
        if cell["level"] == "gaussian":
            summaries[cid] = {"gaussian": gaussian_summary(results[cid]["summary"], ev_config)}
        else:
            summaries[cid] = {
                f: momentum_family_summary(results[cid]["replicates"], f, pairs, ev_config)
                for f in cell["families"]
            }
    selection = select_pair(summaries, hypotheses, pairs, ev_config)
    psr_cells: JsonDict = {}
    null_pbo: dict[tuple[str, str], list[float]] = {}
    for cell in cells:
        if cell["level"] != "momentum" or cell["hypothesis"] != "H0":
            continue
        for family in cell["families"]:
            s = summaries[str(cell["id"])][family]
            if "psr" in s and str(cell["id"]) not in psr_cells:
                psr_cells[str(cell["id"])] = s["psr"]
            if "pbo_values" in s:
                null_pbo[(str(cell["scenario"]), family)] = s["pbo_values"]
    pbo_signal: JsonDict = {}
    for cell in cells:
        if cell["level"] != "momentum" or cell["hypothesis"] == "H0":
            continue
        for family in cell["families"]:
            s = summaries[str(cell["id"])][family]
            if "pbo_values" not in s:
                continue
            reference = null_pbo.get((str(cell["scenario"]), family))
            pbo_signal[f"{cell['id']}/{family}"] = {
                "auc_vs_null": None if reference is None else pbo_auc(s["pbo_values"], reference),
                "pbo_vs_truth": s.get("pbo_vs_truth"),
                "overfit_true": s.get("overfit_true"),
            }
    for families in summaries.values():
        for s in families.values():
            s.pop("pbo_values", None)
            for entry in (s.get("pairs") or {}).values():
                if isinstance(entry, dict):
                    entry.pop("indicators", None)
    return {
        "summaries": summaries,
        "pair_selection": selection,
        "psr_level": psr_level_tests(psr_cells, float(ev_config["fwer"])) if psr_cells else None,
        "pbo_signal": pbo_signal,
    }
