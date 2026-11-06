"""Artefatos endereçados por conteúdo e fluxos aleatórios reproduzíveis."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from frente_e.errors import ContractViolationError, DataValidationError
from frente_e.experiments.artifacts import ArtifactRef, ArtifactStore
from frente_e.experiments.canonical import sha256_hex
from frente_e.experiments.seeds import generator_for, new_root_seed


# ------------------------------------------------------------------- artefatos
def test_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    a = store.put_bytes(b"abc", suffix=".txt", media_type="text/plain", description="x")
    b = store.put_bytes(b"abc", suffix=".txt", media_type="text/plain", description="x")
    assert a == b
    assert a.sha256 == sha256_hex(b"abc")
    assert a.relative_path == f"{a.sha256[:2]}/{a.sha256}.txt"
    store.verify(a)


def test_array_roundtrip_without_pickle(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    matrix = generator_for(3, 0).normal(size=(50, 4))
    ref = store.put_array(matrix, description="T×N")
    np.testing.assert_array_equal(store.load_array(ref), matrix)
    with pytest.raises(ContractViolationError):
        store.put_array(np.array([object()]), description="obj")


def test_tampered_artifact_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"abc", suffix=".txt", media_type="text/plain", description="x")
    store.path_of(ref).write_bytes(b"abd")
    with pytest.raises(DataValidationError, match="não confere"):
        store.verify(ref)
    with pytest.raises(DataValidationError, match="corrompido"):
        store.put_bytes(b"abc", suffix=".txt", media_type="text/plain", description="x")


def test_missing_artifact_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"abc", suffix=".txt", media_type="text/plain", description="x")
    store.path_of(ref).unlink()
    with pytest.raises(DataValidationError, match="ausente"):
        store.verify(ref)


@pytest.mark.parametrize("path", ["../x", "/abs/x", "a/../../x"])
def test_unsafe_relative_paths_rejected(path: str) -> None:
    with pytest.raises(ContractViolationError):
        ArtifactRef(path, "a" * 64, 1, "text/plain", "x")


# ------------------------------------------------------------------- sementes
def test_same_seed_and_path_reproduce_exactly() -> None:
    a = generator_for(42, 3, 1).normal(size=1000)
    b = generator_for(42, 3, 1).normal(size=1000)
    np.testing.assert_array_equal(a, b)


def test_different_paths_and_seeds_differ() -> None:
    base = generator_for(42, 0).normal(size=100)
    assert not np.array_equal(base, generator_for(42, 1).normal(size=100))
    assert not np.array_equal(base, generator_for(43, 0).normal(size=100))


def test_stream_does_not_depend_on_creation_order() -> None:
    forward = [generator_for(9, i).normal(size=5) for i in range(4)]
    backward = [generator_for(9, i).normal(size=5) for i in reversed(range(4))][::-1]
    for f, b in zip(forward, backward, strict=True):
        np.testing.assert_array_equal(f, b)


def _replica(i: int) -> list[float]:
    values: list[float] = generator_for(2026, i).normal(size=3).tolist()
    return values


def test_results_identical_across_process_counts() -> None:
    sequential = [_replica(i) for i in range(4)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(_replica, range(4)))
    assert sequential == parallel


def test_new_root_seed_within_bounds() -> None:
    seeds = {new_root_seed() for _ in range(100)}
    assert all(0 <= s < 2**63 for s in seeds)
    assert len(seeds) == 100


@pytest.mark.parametrize(("root", "path"), [(-1, ()), (2**63, ()), (1, (-1,)), (True, ())])
def test_invalid_seed_inputs_rejected(root: int, path: tuple[int, ...]) -> None:
    with pytest.raises(ContractViolationError):
        generator_for(root, *path)
