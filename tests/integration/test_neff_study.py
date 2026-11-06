"""Estudo preliminar de N efetivo: registro no ledger e reprodutibilidade.

Usa parâmetros reduzidos (rotulados como teste), nunca o ledger do projeto.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from frente_e.experiments import neff_study
from frente_e.experiments.artifacts import ArtifactStore
from frente_e.experiments.ledger import verify_ledger

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]

SMALL: dict[str, Any] = {
    **neff_study.PARAMETERS,
    "structures": {
        "independent": {"kind": "equicorrelation", "m": 8, "rho": 0.0},
        "blocks": {"kind": "blocks", "n_blocks": 2, "block_size": 4, "rho": 0.9},
    },
    "n_reps": 2_000,
    "chunk": 500,
    "n_obs": 60,
    "n_correlation_draws": 3,
    "approximation_table_n": [2, 10],
}


def test_run_is_recorded_and_reproducible(tmp_path: Path) -> None:
    run_id, ref, result = neff_study.execute(
        SMALL,
        project_root=REPO_ROOT,
        ledger_path=tmp_path / "ledger.jsonl",
        artifacts_root=tmp_path / "artifacts",
        root_seed=123,
    )
    verification = verify_ledger(tmp_path / "ledger.jsonl")
    assert verification.count_by_status()["completed"] == 1
    assert verification.incomplete_runs == ()

    started = json.loads((tmp_path / "ledger.jsonl").read_text("utf-8").splitlines()[0])
    assert started["run_id"] == run_id
    assert started["run_kind"] == "synthetic"
    assert started["random_seeds"] == {"root": 123}

    stored = json.loads(ArtifactStore(tmp_path / "artifacts").path_of(ref).read_bytes())
    assert stored == json.loads(json.dumps(result))

    # Reconstrução só com o que está no ledger: mesmo resultado, bit a bit.
    rerun = neff_study.run_study(started["parameters"], started["random_seeds"]["root"])
    assert rerun == result


def test_result_contents_are_internally_consistent() -> None:
    result: Any = json.loads(json.dumps(neff_study.run_study(SMALL, 7)))
    independent = result["structures"]["independent"]
    assert independent["estimators"]["raw"]["n_hat"] == 8.0
    blocks = result["structures"]["blocks"]
    # correlação encolhe a variância entre tentativas (DE-04.b)
    assert blocks["mean_cross_variance"] < independent["mean_cross_variance"]
    for row in result["approximation"]:
        assert row["rel_error"] == pytest.approx(row["approx"] / row["exact"] - 1)


def test_unknown_structure_is_rejected() -> None:
    with pytest.raises(ValueError, match="desconhecida"):
        neff_study.build_correlation({"kind": "nope"})
