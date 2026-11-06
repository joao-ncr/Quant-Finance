"""DE-07, Etapa B: gerador com sinal pela regra DE-01, calibração, manifesto e executor.

Tudo com parâmetros reduzidos, ledger e artefatos temporários e manifesto de fase
``development``. Nenhuma execução confirmatória é feita aqui.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frente_e.core.cscv import cscv_pbo
from frente_e.core.momentum import momentum_family, valuation_returns
from frente_e.core.null_models import dsr_null_rejection_rate_equicorrelated
from frente_e.core.sharpe_inference import expected_max_factor
from frente_e.core.signal_models import mean_shift
from frente_e.errors import ContractViolationError, ReproducibilityError
from frente_e.experiments import de07_calibration as cal
from frente_e.experiments import de07_confirmatory as conf
from frente_e.experiments import de07_design as design
from frente_e.experiments import de07_evaluation as ev
from frente_e.experiments import de07_manifest as man
from frente_e.experiments import momentum_scenarios as ms
from frente_e.experiments.cscv_input import trim_oldest_to_multiple
from frente_e.experiments.ledger import verify_ledger
from frente_e.experiments.seeds import generator_for

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE: dict[str, Any] = dict(ms.SCENARIOS["base"])  # type: ignore[arg-type]
COMMON: dict[str, Any] = dict(ms.COMMON)
SMALL_SCENARIO: dict[str, Any] = {**BASE, "n_assets": 12}
SMALL_COMMON: dict[str, Any] = {**COMMON, "n_obs": 900}
F144 = [c.config_id for c in momentum_family("F144")]
NULL_BASE_DIGEST = "af168b2ecf34fdbef6ba0fa9e4737f1378e4e0824993b9c9b2f5b650d299e179"
MOM_SIGNAL = {"half_life": 252.0, "scale": 0.004, "premium": 0.0}


# ------------------------------------------------------------------ gerador e DE-01
def test_null_control_is_bit_identical_to_the_original_scenario() -> None:
    """Digest do painel `base` com generator_for(7, 1), calculado com o código anterior à
    Etapa B2 (commit 3d2d3e8): os controles nulos continuam exatamente os mesmos."""
    panel, _ = design.signal_panel(BASE, COMMON, None, 7, (1,))
    old, _ = ms.scenario_panel(BASE, COMMON, generator_for(7, 1, 0))
    assert np.array_equal(panel.total_return_index, old.total_return_index)
    direct, _ = ms.scenario_panel(BASE, COMMON, generator_for(7, 1))
    digest = hashlib.sha256(direct.total_return_index.tobytes()).hexdigest()
    assert digest == NULL_BASE_DIGEST


def test_signal_changes_only_the_mean_term() -> None:
    null, _ = design.signal_panel(SMALL_SCENARIO, SMALL_COMMON, None, 11, (0, 0))
    sig, _ = design.signal_panel(SMALL_SCENARIO, SMALL_COMMON, MOM_SIGNAL, 11, (0, 0))
    shift = valuation_returns(sig.total_return_index) - valuation_returns(null.total_return_index)
    expected = mean_shift(
        900,
        12,
        half_life=252.0,
        scale=0.004,
        premium=0.0,
        rng=generator_for(11, 0, 0, design.STREAM_SIGNAL),
    )
    assert np.allclose(shift[1:], expected[1:], rtol=0.0, atol=1e-12)
    assert np.max(np.abs(shift[1:].mean(axis=1))) < 1e-12


def test_signal_goes_through_the_de01_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = getattr(design, "build_momentum_returns")  # noqa: B009

    def spy(panel: Any, rf: Any, configs: Any) -> Any:
        calls.append(len(configs))
        return original(panel, rf, configs)

    monkeypatch.setattr(design, "build_momentum_returns", spy)
    grid = design.signal_grid(SMALL_SCENARIO, SMALL_COMMON, MOM_SIGNAL, 3, (0,), "F16")
    assert calls == [16]
    assert grid.excess_returns.n_configs == 16


def _truth(signal: dict[str, float] | None, family: str, reps: int = 6) -> dict[str, Any]:
    params = {
        "scenario": "base",
        "scenarios": {"base": SMALL_SCENARIO},
        "common": SMALL_COMMON,
        "family": family,
    }
    return cal.truth_table(params, signal, 5, (9,), reps, n_workers=1)


def test_momentum_signal_makes_winners_profitable_and_premium_is_common() -> None:
    """H₁-mom: SR verdadeiros > 0 (médias específicas persistentes chegam à carteira só
    pelos preços). H₁-com: SR verdadeiros > 0 e quase iguais entre configurações. H₀: ≈ 0."""
    mom = _truth(MOM_SIGNAL, "F4")
    assert all(
        sr > 3 * se for sr, se in zip(mom["true_sharpe"], mom["true_sharpe_se"], strict=True)
    )
    sigma_ew = design.equal_weight_volatility(SMALL_SCENARIO)
    premium = 0.5 * sigma_ew  # forte, só para o teste
    com = _truth({"half_life": 252.0, "scale": 0.0, "premium": premium}, "F4")
    sr = np.array(com["true_sharpe"])
    assert np.all(sr > 0.3)
    assert float(sr.std() / sr.mean()) < 0.1
    null = _truth(None, "F4")
    assert all(
        abs(s) < 4 * se for s, se in zip(null["true_sharpe"], null["true_sharpe_se"], strict=True)
    )


def test_equal_weight_volatility_matches_simulation() -> None:
    panel, _ = design.signal_panel(BASE, COMMON, None, 13, (0,))
    ew = valuation_returns(panel.total_return_index)[1:].mean(axis=1)
    assert float(ew.std(ddof=1)) == pytest.approx(
        design.equal_weight_volatility(BASE),
        rel=0.03,
    )


def test_truth_table_is_identical_for_one_or_two_processes() -> None:
    params = {
        "scenario": "base",
        "scenarios": {"base": SMALL_SCENARIO},
        "common": SMALL_COMMON,
        "family": "F4",
    }
    one = cal.truth_table(params, MOM_SIGNAL, 21, (1,), 4, n_workers=1)
    two = cal.truth_table(params, MOM_SIGNAL, 21, (1,), 4, n_workers=2)
    assert one == two


# --------------------------------------------------- separação calibração × avaliação
FORBIDDEN_FOR_CALIBRATION = {
    "frente_e.core.cscv",
    "frente_e.core.sharpe_inference",
    "frente_e.core.effective_trials",
    "frente_e.core.mc_inference",
    "frente_e.experiments.cscv_input",
    "frente_e.experiments.de07_confirmatory",
    "frente_e.experiments.de07_evaluation",
}


@pytest.mark.parametrize(
    "module",
    [
        "src/frente_e/experiments/de07_calibration.py",
        "src/frente_e/experiments/de07_design.py",
        "src/frente_e/core/signal_models.py",
    ],
)
def test_calibration_and_generation_do_not_import_evaluation_code(module: str) -> None:
    tree = ast.parse((REPO_ROOT / module).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imported & FORBIDDEN_FOR_CALIBRATION


# ------------------------------------------------------------------ calibração (procedimento)
def test_interpolation_recovers_a_power_law_and_rejects_bad_grids() -> None:
    table = [(s, 50.0 * s**0.8) for s in (0.001, 0.002, 0.004, 0.008)]
    target = 50.0 * 0.003**0.8
    assert cal.interpolate_scale(table, target) == pytest.approx(0.003, rel=1e-12)
    assert cal.local_log_slope(table, 0.003) == pytest.approx(0.8, rel=1e-12)
    assert cal.local_log_slope(table, 0.008 * (1 + 1e-15)) == pytest.approx(0.8, rel=1e-12)
    with pytest.raises(ContractViolationError):
        cal.local_log_slope(table, 0.02)
    with pytest.raises(ContractViolationError):
        cal.interpolate_scale([(0.001, 2.0), (0.002, 1.0)], 1.5)
    with pytest.raises(ContractViolationError):
        cal.interpolate_scale(table, 1000.0)


def test_pooled_sharpe_equals_direct_concatenation() -> None:
    rng = generator_for(1, 2)
    blocks = [rng.normal(0.01, 1.0, size=(50, 3)) for _ in range(4)]
    parts = [
        (
            b.shape[0],
            b.mean(axis=0),
            np.sum((b - b.mean(axis=0)) ** 2, axis=0),
            b.mean(axis=0) / b.std(axis=0, ddof=1),
        )
        for b in blocks
    ]
    sr, _, n_rows = cal.pooled_sharpe(parts)
    allx = np.vstack(blocks)
    assert n_rows == 50
    assert np.allclose(sr, allx.mean(axis=0) / allx.std(axis=0, ddof=1), rtol=1e-12)


def _fake_truth(delta_of_scale: Any) -> Any:
    def fake(params: Any, signal: Any, root: int, prefix: Any, n: int, *, n_workers: int) -> Any:
        delta = delta_of_scale(signal["scale"], tuple(prefix)) if signal else 0.0
        return {
            "config_ids": F144,
            "true_sharpe": [delta / math.sqrt(399)] * 144,
            "true_sharpe_se": [0.0] * 144,
            "n_rows": 400,
            "n_replicates": n,
            "delta_star": delta,
            "delta_star_se": 0.0,
            "best_config": F144[0],
        }

    return fake


CAL_SMALL = {
    **cal.PARAMETERS,
    "scenarios": {"base": SMALL_SCENARIO},
    "common": SMALL_COMMON,
    "scale_grid": [0.001, 0.002],
    "grid_replicates": 2,
    "oracle_replicates": 2,
}


def test_calibration_extends_grid_interpolates_and_refines(monkeypatch: pytest.MonkeyPatch) -> None:
    # δ*(s) = 1000·s: exige estender a grade para baixo (alvo 1) e para cima (alvo 4)
    monkeypatch.setattr(cal, "truth_table", _fake_truth(lambda s, prefix: 1000.0 * s))
    result = cal.calibrate(CAL_SMALL, 1, n_workers=1)
    cells = result["cells"]
    assert cells["mom_h63_d2.5"]["signal"]["scale"] == pytest.approx(0.0025)
    assert cells["mom_h252_d1"]["signal"]["scale"] == pytest.approx(0.001)
    assert cells["mom_h252_d4"]["signal"]["scale"] == pytest.approx(0.004)
    assert "com_d2.5" in cells

    # oráculo com viés de 0,8 em relação à grade: a verificação falha, um refinamento pela
    # inclinação local (fluxo próprio, mesmo viés) acerta o alvo
    def biased(s: float, prefix: tuple[int, ...]) -> float:
        oracle = prefix[0] in (cal.STAGE_ORACLE, cal.STAGE_REFINE)
        return 1000.0 * s * (0.8 if oracle else 1.0)

    monkeypatch.setattr(cal, "truth_table", _fake_truth(biased))
    refined = cal.calibrate(CAL_SMALL, 1, n_workers=1)["cells"]["mom_h63_d4"]
    assert len(refined["attempts"]) == 2
    assert refined["truth"]["delta_star"] == pytest.approx(4.0)

    # o refinamento também erra: erro explícito, sem segunda tentativa silenciosa
    def drifting(s: float, prefix: tuple[int, ...]) -> float:
        factor = {cal.STAGE_ORACLE: 0.5, cal.STAGE_REFINE: 0.25}.get(prefix[0], 1.0)
        return 1000.0 * s * factor

    monkeypatch.setattr(cal, "truth_table", _fake_truth(drifting))
    with pytest.raises(ContractViolationError, match="não convergiu"):
        cal.calibrate(CAL_SMALL, 1, n_workers=1)


def test_calibration_run_is_recorded_as_calibration_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cal, "truth_table", _fake_truth(lambda s, prefix: 1000.0 * s))
    ledger = tmp_path / "ledger.jsonl"
    _run_id, _ref, _ = cal.execute(
        CAL_SMALL,
        project_root=REPO_ROOT,
        ledger_path=ledger,
        artifacts_root=tmp_path / "a",
        n_workers=1,
        root_seed=3,
    )
    verify_ledger(ledger)
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert records[0]["parameters"]["phase"] == "calibration"
    assert records[0]["run_kind"] == "synthetic"
    assert records[-1]["status"] == "completed"
    with pytest.raises(ContractViolationError):
        cal.execute(
            {**CAL_SMALL, "phase": "confirmatory"},
            project_root=REPO_ROOT,
            ledger_path=ledger,
            artifacts_root=tmp_path / "a",
            n_workers=1,
        )


# ------------------------------------------------------------------ manifesto e executor
def _write_project(tmp_path: Path, *, phase: str = "development", **overrides: Any) -> Path:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "proto.md").write_text("protocolo de teste\n", encoding="utf-8")
    truth = {
        "cells": {
            "mom": {
                "hypothesis": "H1-mom",
                "signal": MOM_SIGNAL,
                "truth": {"config_ids": F144, "true_sharpe": list(np.linspace(0.0, 0.02, 144))},
            }
        }
    }
    (tmp_path / "truth.json").write_text(json.dumps(truth), encoding="utf-8")

    def sha(rel: str) -> str:
        return hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()

    reps = {"G0": 2000, "G1": 2000, "P0": 3, "P1": 3}
    if phase == "confirmatory":
        reps = {"G0": 20_000, "G1": 20_000, "P0": 1000, "P1": 500}
    reps.update(overrides.pop("reps", {}))
    common_cell = {"metrics": ["dsr"], "justification": "teste", "estimated_seconds": 1.0}
    cells = [
        {
            "id": "G0",
            "order": 1,
            "level": "gaussian",
            "hypothesis": "H0",
            "replicates": reps["G0"],
            **common_cell,
            "structure": {"kind": "equicorrelation", "m": 8, "rho": 0.5},
            "n_obs": 500,
            "signal_pattern": "none",
            "signal_delta": 0.0,
        },
        {
            "id": "G1",
            "order": 2,
            "level": "gaussian",
            "hypothesis": "H1-gauss",
            "replicates": reps["G1"],
            **common_cell,
            "structure": {"kind": "equicorrelation", "m": 8, "rho": 0.5},
            "n_obs": 500,
            "signal_pattern": "single",
            "signal_delta": 2.5,
        },
        {
            "id": "P0",
            "order": 3,
            "level": "momentum",
            "hypothesis": "H0",
            "replicates": reps["P0"],
            **common_cell,
            "scenario": "base",
            "families": ["F4", "F16"],
            "cscv_families": ["F4", "F16"],
        },
        {
            "id": "P1",
            "order": 4,
            "level": "momentum",
            "hypothesis": "H1-mom",
            "replicates": reps["P1"],
            **common_cell,
            "scenario": "base",
            "families": ["F4", "F16"],
            "cscv_families": ["F4", "F16"],
            "signal": MOM_SIGNAL,
            "truth_key": "mom",
        },
    ]
    data = {
        "manifest": {
            "name": "teste",
            "version": 1,
            "phase": phase,
            "created": "2026-11-06",
            "protocol_path": "docs/proto.md",
            "protocol_version": 1,
            "protocol_sha256": sha("docs/proto.md"),
            "truth_path": "truth.json",
            "truth_sha256": sha("truth.json"),
            "calibration_run_id": "run-x",
            "calibration_artifact_sha256": "0" * 64,
            "code_commit": "abc",
            "uv_lock_sha256": "0",
            "python": "3.13.5",
            "numpy": "x",
            "scipy": "x",
            "execution_order": ["G0", "G1", "P0", "P1"],
            "notes": "teste",
        },
        "seeds": {"gaussian": 101, "momentum": 202, "calibration": 303},
        "budget": {
            "max_hours": 2.0,
            "n_workers": 2,
            "parallel_speedup": 1.5,
            "seconds_per_replicate": {"x": 1.0},
            "estimated_hours": 0.0,
            "basis": "teste",
        },
        "evaluation": {
            "alpha": 0.05,
            "fwer": 0.05,
            "ci_level": 0.95,
            "psr_config": "j06_k06",
            "n_blocks": 4,
            "pairs": list(conf.PAIRS),
            "v_candidates": list(conf.V_LABELS),
            "power_cells": ["P1"],
            "pbo_quantiles": [0.05, 0.5, 0.95],
            "pbo_tail_points": [0.5, 0.9],
            "decision_rule": "D3",
        },
        "common": SMALL_COMMON,
        "scenarios": {"base": SMALL_SCENARIO},
        "cells": cells,
    }
    for key, value in overrides.items():
        data[key] = value
    path = tmp_path / "manifest.toml"
    path.write_text(man.to_toml(data), encoding="utf-8")
    return path


def test_to_toml_round_trips(tmp_path: Path) -> None:
    path = _write_project(tmp_path)
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["cells"][3]["signal"] == MOM_SIGNAL
    assert parsed["scenarios"]["base"] == SMALL_SCENARIO
    assert man.to_toml(parsed) == path.read_text(encoding="utf-8")


def test_manifest_verification_failures(tmp_path: Path) -> None:
    path = _write_project(tmp_path)
    manifest = man.load_manifest(path, project_root=tmp_path)
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "docs" / "proto.md").write_text("alterado\n", encoding="utf-8")
    with pytest.raises(ContractViolationError, match="SHA-256"):
        man.load_manifest(path, project_root=tmp_path)
    path = _write_project(tmp_path, seeds={"gaussian": 1, "momentum": 2, "calibration": 1})
    with pytest.raises(ContractViolationError, match="distintas"):
        man.load_manifest(path, project_root=tmp_path)
    path = _write_project(tmp_path, phase="confirmatory", reps={"P1": 400})
    with pytest.raises(ContractViolationError, match="D5"):
        man.load_manifest(path, project_root=tmp_path)
    raw = tomllib.loads(_write_project(tmp_path).read_text(encoding="utf-8"))
    raw["budget"]["extra"] = 1
    path.write_text(man.to_toml(raw), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="desconhecidas"):
        man.load_manifest(path, project_root=tmp_path)


def test_confirmatory_phase_is_blocked_without_clean_commit(tmp_path: Path) -> None:
    path = _write_project(tmp_path, phase="confirmatory")
    manifest = man.load_manifest(path, project_root=tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(ReproducibilityError):
        conf.run_cell(
            manifest,
            "G0",
            project_root=tmp_path,
            ledger_path=ledger,
            artifacts_root=tmp_path / "a",
            n_workers=2,
        )
    assert not ledger.exists()  # nada registrado


def test_main_requires_explicit_authorization_for_confirmatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = man.load_manifest(
        _write_project(tmp_path, phase="confirmatory"), project_root=tmp_path
    )
    monkeypatch.setattr(conf, "load_manifest", lambda *a, **k: manifest)
    called: list[str] = []
    monkeypatch.setattr(conf, "run_cell", lambda *a, **k: called.append("run"))
    with pytest.raises(ReproducibilityError, match="Etapa C bloqueada"):
        conf.main(["--manifest", "x.toml", "--cell", "G0"])
    assert not called


def test_development_cells_run_record_and_evaluate(tmp_path: Path) -> None:
    manifest = man.load_manifest(_write_project(tmp_path), project_root=tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    artifacts = tmp_path / "a"
    results: dict[str, Any] = {}
    for cell in manifest.raw["manifest"]["execution_order"]:
        _, ref = conf.run_cell(
            manifest,
            cell,
            project_root=REPO_ROOT,
            ledger_path=ledger,
            artifacts_root=artifacts,
            n_workers=1,
        )
        results[cell] = json.loads((artifacts / ref.relative_path).read_bytes())
    state = verify_ledger(ledger)
    assert len(state.runs) == 4
    assert not state.incomplete_runs
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert {r["run_kind"] for r in records if r["status"] == "started"} == {"development"}
    assert all(
        r["parameters"]["manifest_sha256"] == manifest.sha256
        for r in records
        if r["status"] == "started"
    )
    # avaliação completa sobre os resumos (só estrutura; R pequeno)
    evalcfg = manifest.evaluation
    summaries: dict[str, Any] = {
        "G0": {"gaussian": ev.gaussian_summary(results["G0"]["summary"], evalcfg)},
        "G1": {"gaussian": ev.gaussian_summary(results["G1"]["summary"], evalcfg)},
    }
    for cell in ("P0", "P1"):
        summaries[cell] = {
            f: ev.momentum_family_summary(results[cell]["replicates"], f, conf.PAIRS, evalcfg)
            for f in ("F4", "F16")
        }
    hyp = {"G0": "H0", "G1": "H1-gauss", "P0": "H0", "P1": "H1-mom"}
    decision = ev.select_pair(summaries, hyp, conf.PAIRS, evalcfg)
    assert set(decision["pairs"]) == set(conf.PAIRS)
    assert "pbo_vs_truth" in summaries["P1"]["F16"]
    assert "pbo_vs_truth" not in summaries["P0"]["F16"]
    assert "psr" in summaries["P0"]["F4"]


def test_momentum_replicate_metrics_match_direct_computation() -> None:
    cell = {
        "scenario": "base",
        "families": ["F4", "F16"],
        "cscv_families": ["F16"],
        "signal": MOM_SIGNAL,
    }
    truth = dict(zip(F144, np.linspace(0.0, 0.02, 144), strict=True))
    args = (cell, SMALL_COMMON, {"base": SMALL_SCENARIO}, truth, 77, 3, 0, 4, "j06_k06")
    out: Any = conf.momentum_replicate(args)
    grid = design.signal_grid(SMALL_SCENARIO, SMALL_COMMON, MOM_SIGNAL, 77, (3, 0), "F16")
    trimmed = trim_oldest_to_multiple(grid.excess_returns, 4)[0]
    assert out["F16"]["pbo"] == pytest.approx(cscv_pbo(trimmed, 4).pbo, abs=1e-15)
    assert out["F4"]["pbo"] is None
    assert out["F16"]["psr"] is not None
    assert out["F4"]["psr"] is not None
    assert out == conf.momentum_replicate(args)  # determinístico


def test_momentum_cell_results_do_not_depend_on_worker_count() -> None:
    cell = {"scenario": "base", "families": ["F4"], "cscv_families": ["F4"], "signal": MOM_SIGNAL}
    items = [
        (cell, SMALL_COMMON, {"base": SMALL_SCENARIO}, None, 5, 1, r, 4, "j06_k06")
        for r in range(4)
    ]
    one = design.map_ordered(conf.momentum_replicate, items, n_workers=1)
    two = design.map_ordered(conf.momentum_replicate, items, n_workers=2)
    assert one == two


def test_gaussian_cell_matches_the_analytic_rejection_rate() -> None:
    """Com V fixo (V_H0) e N = M na equicorrelação, a taxa tem forma exata (§2.2)."""
    cell = {
        "structure": {"kind": "equicorrelation", "m": 8, "rho": 0.5},
        "n_obs": 500,
        "signal_pattern": "none",
        "signal_delta": 0.0,
        "replicates": 40_000,
    }
    out: Any = conf.gaussian_cell(cell, 9, 1)
    k = out["pairs"]["raw+v_h0"]["rejections"]
    p = dsr_null_rejection_rate_equicorrelated(8, 0.5, 500, 0.95, float(expected_max_factor(8)))
    assert abs(k / 40_000 - p) < 4 * math.sqrt(p * (1 - p) / 40_000)
    assert out["pairs"]["raw+v_h0"]["true_discoveries"] == 0
