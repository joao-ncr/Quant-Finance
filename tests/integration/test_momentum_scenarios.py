"""Cenários de momentum sintéticos: registro no ledger e reconstrução (parâmetros reduzidos)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frente_e.experiments import momentum_scenarios as ms
from frente_e.experiments.ledger import verify_ledger
from frente_e.experiments.seeds import generator_for

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]

_SCENARIOS: dict[str, Any] = ms.SCENARIOS
SMALL: dict[str, Any] = {
    **ms.PARAMETERS,
    "scenarios": {
        name: {**spec, "n_assets": 12}
        for name, spec in _SCENARIOS.items()
        if name in ("base", "vol_clustering")
    },
    "common": {**ms.COMMON, "n_obs": 800},
    "family": "F16",
}


def test_scenarios_fix_mean_equal_to_risk_free() -> None:
    panel, rf = ms.scenario_panel(_SCENARIOS["base"], ms.COMMON, generator_for(1, 0))
    assert panel.total_return_index.shape == (6550, 60)
    assert np.all(rf == ms.COMMON["risk_free_daily"])
    short = {**ms.COMMON, "n_obs": 50}
    a, _ = ms.scenario_panel(_SCENARIOS["base"], short, generator_for(1, 0))
    again, _ = ms.scenario_panel(_SCENARIOS["base"], short, generator_for(1, 0))
    other, _ = ms.scenario_panel(_SCENARIOS["base"], short, generator_for(2, 0))
    np.testing.assert_array_equal(a.total_return_index, again.total_return_index)
    assert not np.array_equal(a.total_return_index, other.total_return_index)


def test_mechanics_run_is_recorded_and_reconstructible(tmp_path: Path) -> None:
    run_id, _, result = ms.execute(
        SMALL,
        project_root=REPO_ROOT,
        ledger_path=tmp_path / "ledger.jsonl",
        artifacts_root=tmp_path / "artifacts",
        root_seed=7,
    )
    verification = verify_ledger(tmp_path / "ledger.jsonl")
    assert verification.count_by_status()["completed"] == 1
    started = json.loads((tmp_path / "ledger.jsonl").read_text("utf-8").splitlines()[0])
    assert started["run_id"] == run_id
    rebuilt = ms.run_mechanics(started["parameters"], started["random_seeds"]["root"])
    assert rebuilt == result
    summary: Any = result["base"]
    assert summary["n_configs"] == 16
    assert summary["deferred_entries"] == 0  # painel completo
    assert -1.0 <= summary["config_correlation_min"] <= summary["config_correlation_max"] <= 1.0
