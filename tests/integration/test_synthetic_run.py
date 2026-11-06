"""Ponta a ponta: execução sintética registrada e reconstruída só a partir do ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frente_e.core.contracts import ReturnsMatrix
from frente_e.core.metrics import sharpe_ratio
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.identity import InputKind, InputReference, RunKind, RunSpec
from frente_e.experiments.ledger import LedgerWriter, verify_ledger
from frente_e.experiments.provenance import code_revision, software_environment
from frente_e.experiments.seeds import generator_for, new_root_seed

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_experiment(params: dict[str, int], root_seed: int) -> np.ndarray:
    """Experimento de desenvolvimento: matriz sem sinal e Sharpe diário por coluna."""
    rng = generator_for(root_seed, 0)
    values = rng.normal(0.0, 0.01, size=(params["t"], params["n"]))
    matrix = ReturnsMatrix(
        values, np.arange(params["t"]), tuple(f"c{i}" for i in range(params["n"]))
    )
    return sharpe_ratio(matrix.values)


def test_synthetic_run_is_recorded_and_reproducible(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    store = ArtifactStore(tmp_path / "artifacts")
    params = {"t": 252, "n": 10}
    spec = RunSpec(
        experiment_id="dev-synthetic-sharpe",
        run_kind=RunKind.DEVELOPMENT,
        parameters={"kind": "noise", **params, "metric": "sharpe_daily_ddof1"},
        random_seeds={"root": new_root_seed()},
        input_data_references=(
            InputReference(InputKind.SYNTHETIC, "generator://normal(0,0.01)", "ruído i.i.d."),
        ),
    )

    with (
        LedgerWriter(ledger_path) as ledger,
        ledger.run(
            spec,
            software_environment=software_environment(REPO_ROOT),
            code_revision=code_revision(REPO_ROOT),
        ) as run,
    ):
        result = _synthetic_experiment(params, spec.random_seeds["root"])
        run.set_results([store.put_array(result, description="Sharpe diário por coluna")])

    # Reconstrução usando apenas o que está registrado.
    verification = verify_ledger(ledger_path)
    assert verification.count_by_status()["completed"] == 1
    started, completed = (json.loads(x) for x in ledger_path.read_text("utf-8").splitlines())
    assert started["software_environment"]["packages"]["numpy"] == np.__version__
    ref = ArtifactRef.from_json(completed["results_reference"][0])
    recorded = store.load_array(ref)

    params_from_ledger = {k: started["parameters"][k] for k in ("t", "n")}
    rerun = _synthetic_experiment(params_from_ledger, started["random_seeds"]["root"])
    np.testing.assert_array_equal(rerun, recorded)  # mesmo ambiente: igualdade bit a bit
    assert store.put_array(rerun, description="x").sha256 == ref.sha256
