from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from frente_e.core.contracts import ReturnsMatrix
from frente_e.errors import ContractViolationError


def _valid(t: int = 5, n: int = 3) -> dict[str, Any]:
    return {
        "values": np.zeros((t, n), dtype=np.float64),
        "index": np.arange(t, dtype=np.int64),
        "config_ids": tuple(f"c{i}" for i in range(n)),
    }


def test_valid_matrix_exposes_dimensions() -> None:
    m = ReturnsMatrix(**_valid(t=7, n=4))
    assert (m.n_observations, m.n_configs) == (7, 4)


def test_inputs_are_copied_and_read_only() -> None:
    kwargs = _valid()
    m = ReturnsMatrix(**kwargs)
    kwargs["values"][0, 0] = 99.0  # mutar a origem não afeta o contrato
    assert m.values[0, 0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        m.values[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        m.index[0] = 1


def test_datetime_index_is_accepted() -> None:
    kwargs = _valid(t=3)
    kwargs["index"] = np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[D]")
    assert ReturnsMatrix(**kwargs).n_observations == 3


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(bad: float) -> None:
    kwargs = _valid()
    kwargs["values"][2, 1] = bad
    with pytest.raises(ContractViolationError, match="não finitos"):
        ReturnsMatrix(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("values", np.zeros((5, 3), dtype=np.float32), "float64"),
        ("values", np.zeros(5, dtype=np.float64), "2-D"),
        ("values", np.zeros((1, 3), dtype=np.float64), "T >= 2"),
        ("values", np.zeros((5, 0), dtype=np.float64), "N >= 1"),
        ("index", np.arange(4), "1-D com 5"),
        ("index", np.array([0, 1, 1, 2, 3]), "estritamente crescente"),
        ("index", np.array([4, 3, 2, 1, 0]), "estritamente crescente"),
        ("index", np.linspace(0, 1, 5), "inteiro ou datetime64"),
        ("config_ids", ("a", "b"), "esperado 3"),
        ("config_ids", ("a", "a", "b"), "únicos"),
        ("config_ids", ("a", "", "b"), "não vazias"),
    ],
)
def test_contract_violations(field: str, value: object, message: str) -> None:
    kwargs = _valid()
    kwargs[field] = value
    with pytest.raises(ContractViolationError, match=message):
        ReturnsMatrix(**kwargs)


def test_nat_in_index_is_rejected() -> None:
    kwargs = _valid(t=3)
    kwargs["index"] = np.array(["2020-01-02", "NaT", "2020-01-06"], dtype="datetime64[D]")
    with pytest.raises(ContractViolationError, match="estritamente crescente"):
        ReturnsMatrix(**kwargs)


@given(
    values=hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=20),
        elements=st.floats(-1.0, 1.0, allow_nan=False),
    )
)
def test_any_finite_matrix_roundtrips_exactly(
    values: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    t, n = values.shape
    m = ReturnsMatrix(values=values, index=np.arange(t), config_ids=tuple(map(str, range(n))))
    np.testing.assert_array_equal(m.values, values)
