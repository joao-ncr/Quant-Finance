"""Estudo da grade: regras pré-registradas e reprodutibilidade (parâmetros reduzidos)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from frente_e.experiments import grid_study as gs
from frente_e.experiments import momentum_scenarios as ms
from frente_e.experiments.ledger import verify_ledger

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIOS: dict[str, Any] = ms.SCENARIOS
ESTIMATORS = ["raw", "galwey"]

SMALL: dict[str, Any] = {
    **gs.PARAMETERS,
    "scenarios": {"base": {**_SCENARIOS["base"], "n_assets": 12}},
    "common": {**ms.COMMON, "n_obs": 800},
    "replicates": 3,
    "n_blocks": 4,
    "embargoes": [2],
    "estimators": ESTIMATORS,
}


def _generic(error_cross: float, error_h0: float) -> dict[str, Any]:
    cell = {"below_domain": False, "sr0_rel_error": error_h0}
    cell["sr0_rel_error_with_v_cross"] = error_cross
    return {
        "structures": {
            "g": {
                "expected_max": 2.0,
                "expected_max_se": 0.002,
                "estimators": {name: dict(cell) for name in ESTIMATORS},
            }
        }
    }


def _grid(errors: dict[str, tuple[float, float]], se: float = 0.01) -> dict[str, Any]:
    pairs = {
        name: {
            "v_cross": {"rel_error": cross, "se": se},
            "v_h0": {"rel_error": h0, "se": se},
        }
        for name, (cross, h0) in errors.items()
    }
    return {"s": {"F16": {"pairs": pairs}}}


def test_run_is_recorded_reproducible_and_complete(tmp_path: Path) -> None:
    run_id, _, result = gs.execute(
        SMALL,
        project_root=REPO_ROOT,
        ledger_path=tmp_path / "ledger.jsonl",
        artifacts_root=tmp_path / "artifacts",
        generic_reference={
            "reference": {"run_id": "x", "sha256": "0" * 64},
            "result": _generic(0.0, 0.0),
        },
        root_seed=11,
    )
    assert verify_ledger(tmp_path / "ledger.jsonl").count_by_status()["completed"] == 1
    started = json.loads((tmp_path / "ledger.jsonl").read_text("utf-8").splitlines()[0])
    assert started["run_id"] == run_id
    params = {k: v for k, v in started["parameters"].items() if k != "generic_reference"}
    rebuilt = gs.run_study(params, started["random_seeds"]["root"])
    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(
        {"summaries": result["summaries"], "replicates": result["replicates"]}, sort_keys=True
    )
    summary: Any = result["summaries"]["base"]["F144"]
    assert summary["n_configs"] == 144
    assert summary["v_h0"] == pytest.approx(1 / (summary_obs(result) - 1))
    assert set(summary["pbo_mean_se"]) == {
        "plain",
        "independent_halves",
        "permuted_rows",
        "embargo_2",
    }
    assert "embargo" in result["decisions"]
    assert "pair_selection" in result["decisions"]


def summary_obs(result: dict[str, Any]) -> int:
    return int(result["replicates"]["base"][0]["n_obs"])


def test_pair_rule_excludes_anti_conservative_and_picks_smallest_worst_error() -> None:
    grid = _grid({"raw": (0.30, 0.50), "galwey": (-0.20, 0.10)})
    decision = gs.select_pair(grid, _generic(0.05, 0.08), ESTIMATORS)
    pairs = decision["pairs"]
    assert not pairs["galwey+v_cross"]["admissible"]  # −20% < −2 erros-padrão
    assert pairs["galwey+v_cross"]["anti_conservative_in"] == ["s/F16"]
    assert decision["selected"] == "galwey+v_h0"  # pior |erro| 10% entre os admissíveis


def test_pair_rule_tie_prefers_more_conservative_structures() -> None:
    se = 0.01
    pairs: dict[str, Any] = {
        "raw": {"v_cross": [0.100, 0.010], "v_h0": [0.5, 0.5]},
        "galwey": {"v_cross": [0.099, -0.010], "v_h0": [0.5, 0.5]},  # −1% > −2 erros-padrão
    }
    grid = {
        f"s{i}": {
            "F16": {
                "pairs": {
                    name: {
                        label: {"rel_error": values[label][i], "se": se}
                        for label in ("v_cross", "v_h0")
                    }
                    for name, values in pairs.items()
                }
            }
        }
        for i in range(2)
    }
    decision = gs.select_pair(grid, _generic(0.0, 0.0), ESTIMATORS)
    # piores erros 10,0% e 9,9%: empate (diferença < 2 erros-padrão); raw+v_cross tem 3
    # estruturas conservadoras e galwey+v_cross tem 2 -> raw+v_cross
    assert set(decision["tied"]) >= {"raw+v_cross", "galwey+v_cross"}
    assert decision["selected"] == "raw+v_cross"


def test_pair_rule_keeps_decision_open_without_admissible_pair() -> None:
    grid = _grid({"raw": (-0.3, -0.3), "galwey": (-0.3, -0.3)})
    decision = gs.select_pair(grid, _generic(0.0, 0.0), ESTIMATORS)
    assert decision["selected"] is None


def _embargo_grid(leak: float, leak_se: float, emb: float, emb_se: float) -> dict[str, Any]:
    diffs = {
        "plain_minus_permuted_rows": {"mean": leak, "se": leak_se},
        "plain_minus_independent_halves": {"mean": 0.2, "se": 0.01},
        "embargo_21_minus_permuted_rows": {"mean": emb, "se": emb_se},
    }
    return {"s": {"F144": {"pbo_differences": diffs}}}


@pytest.mark.parametrize(
    ("leak", "emb", "adopt"),
    [
        (0.10, 0.01, True),  # efeito temporal detectável e embargo o reduz
        (0.10, 0.10, False),  # detectável, mas o embargo não ajuda
        (0.01, 0.00, False),  # não detectável
    ],
)
def test_embargo_rule(leak: float, emb: float, adopt: bool) -> None:
    decision = gs.embargo_decision(_embargo_grid(leak, 0.02, emb, 0.02))
    assert decision["adopt_embargo"] is adopt
