"""CSCV/PBO de ponta a ponta: matriz sintética -> descarte DE-05.b -> núcleo -> ledger.

Execução de desenvolvimento num ledger temporário (nunca o do projeto).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frente_e.core.contracts import ReturnsMatrix
from frente_e.core.cscv import cscv_pbo
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import JsonValue, canonical_bytes
from frente_e.experiments.cscv_input import cscv_result_to_json, trim_oldest_to_multiple
from frente_e.experiments.identity import InputKind, InputReference, RunKind, RunSpec
from frente_e.experiments.ledger import LedgerWriter, verify_ledger
from frente_e.experiments.provenance import code_revision, software_environment
from frente_e.experiments.seeds import generator_for

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


def _synthetic(params: dict[str, Any], root_seed: int) -> ReturnsMatrix:
    t, n = params["t"], params["n"]
    values = generator_for(root_seed, 0).normal(0.0, 0.01, size=(t, n))
    dates = np.datetime64("2020-01-01") + np.arange(t)
    return ReturnsMatrix(values, dates, tuple(f"cfg{j:02d}" for j in range(n)))


def _pipeline(params: dict[str, Any], root_seed: int) -> dict[str, JsonValue]:
    trimmed, dropped = trim_oldest_to_multiple(_synthetic(params, root_seed), params["n_blocks"])
    return cscv_result_to_json(cscv_pbo(trimmed, params["n_blocks"]), trimmed, dropped)


def test_trim_drops_only_the_oldest_rows() -> None:
    m = _synthetic({"t": 43, "n": 3}, 1)
    trimmed, dropped = trim_oldest_to_multiple(m, 4)
    assert trimmed.n_observations == 40
    assert dropped == ["2020-01-01", "2020-01-02", "2020-01-03"]
    np.testing.assert_array_equal(trimmed.values, m.values[3:])
    same, none = trim_oldest_to_multiple(trimmed, 4)
    assert same is trimmed
    assert none == []


def test_pbo_run_is_recorded_and_reconstructible(tmp_path: Path) -> None:
    params: dict[str, Any] = {"t": 243, "n": 7, "n_blocks": 8, "metric": "sharpe_daily_ddof1"}
    spec = RunSpec(
        experiment_id="dev-cscv-pbo",
        run_kind=RunKind.DEVELOPMENT,
        parameters=params,
        random_seeds={"root": 99},
        input_data_references=(
            InputReference(InputKind.SYNTHETIC, "generator://normal(0,0.01)", "ruído i.i.d."),
        ),
    )
    store = ArtifactStore(tmp_path / "artifacts")
    with (
        LedgerWriter(tmp_path / "ledger.jsonl") as ledger,
        ledger.run(
            spec,
            software_environment=software_environment(REPO_ROOT),
            code_revision=code_revision(REPO_ROOT),
        ) as run,
    ):
        result = _pipeline(params, spec.random_seeds["root"])
        run.set_results(
            [
                store.put_bytes(
                    canonical_bytes(result),
                    suffix=".json",
                    media_type="application/json",
                    description="CSCV/PBO por partição",
                )
            ]
        )

    assert verify_ledger(tmp_path / "ledger.jsonl").count_by_status()["completed"] == 1
    started, completed = (
        json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text("utf-8").splitlines()
    )
    ref = ArtifactRef.from_json(completed["results_reference"][0])
    stored: Any = json.loads(store.path_of(ref).read_bytes())
    assert stored["dropped_oldest_index"] == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert stored["n_observations"] == 240
    assert stored["n_partitions"] == 70
    assert stored["convention"].startswith("rank/(N+1)")

    rebuilt = _pipeline(started["parameters"], started["random_seeds"]["root"])
    assert canonical_bytes(rebuilt) == canonical_bytes(result)
