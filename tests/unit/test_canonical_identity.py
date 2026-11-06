"""Serialização canônica, fingerprint e identificadores (DA-03)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from frente_e.errors import ContractViolationError
from frente_e.experiments.canonical import canonical_json, sha256_hex, to_canonical
from frente_e.experiments.identity import (
    RUN_ID_PATTERN,
    InputKind,
    InputReference,
    PreregistrationReference,
    RunKind,
    RunSpec,
    configuration_fingerprint,
    new_run_id,
)

HASH_A = "a" * 64

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(), children, max_size=4)
    ),
    max_leaves=20,
)


def _spec(**overrides: Any) -> RunSpec:
    base: dict[str, Any] = {
        "experiment_id": "momentum-grid-v1",
        "run_kind": RunKind.SYNTHETIC,
        "parameters": {"family": "momentum", "formation": [21, 63], "holding": 21},
        "random_seeds": {"root": 12345},
        "universe_reference": "universe/liquid-v1",
    }
    base.update(overrides)
    return RunSpec(**base)


# ------------------------------------------------------------------- forma canônica
def test_canonical_json_is_compact_sorted_and_utf8() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "ç"], "c": None}) == '{"a":[1.5,"ç"],"b":1,"c":null}'


@given(json_values)
def test_canonical_json_roundtrips(value: Any) -> None:
    text = canonical_json(value)
    assert canonical_json(json.loads(text)) == text


@given(st.dictionaries(st.text(), st.integers(), min_size=2, max_size=8))
def test_key_insertion_order_does_not_matter(mapping: dict[str, int]) -> None:
    reversed_mapping = dict(reversed(list(mapping.items())))
    assert canonical_json(mapping) == canonical_json(reversed_mapping)


def test_int_and_float_are_distinct_values() -> None:
    assert canonical_json({"x": 1}) != canonical_json({"x": 1.0})


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), {1: "x"}, {"a": {1, 2}}, np.int64(3), np.bool_(True), b"bytes"],
)
def test_non_canonicalizable_values_are_rejected(bad: object) -> None:
    with pytest.raises(ContractViolationError):
        to_canonical(bad)


def test_sha256_known_vector() -> None:
    assert sha256_hex(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# ------------------------------------------------------------------- fingerprint
def test_fingerprint_is_deterministic_and_order_independent() -> None:
    a = _spec(parameters={"family": "momentum", "holding": 21, "formation": [21, 63]})
    b = _spec(parameters={"formation": [21, 63], "holding": 21, "family": "momentum"})
    assert configuration_fingerprint(a) == configuration_fingerprint(b)
    assert configuration_fingerprint(a) == configuration_fingerprint(
        _spec(parameters={"family": "momentum", "holding": 21, "formation": [21, 63]})
    )


@pytest.mark.parametrize(
    "change",
    [
        {"parameters": {"family": "momentum", "formation": [21, 63], "holding": 42}},
        {"parameters": {"family": "momentum", "formation": [63, 21], "holding": 21}},
        {"random_seeds": {"root": 12346}},
        {"random_seeds": {"root": 12345, "bootstrap": 1}},
        {"universe_reference": "universe/liquid-v2"},
        {"universe_reference": None},
        {
            "preregistration_reference": PreregistrationReference(
                "configs/studies/a.toml", 1, HASH_A
            )
        },
    ],
)
def test_relevant_changes_produce_new_fingerprint(change: dict[str, Any]) -> None:
    assert configuration_fingerprint(_spec()) != configuration_fingerprint(_spec(**change))


def test_preregistration_version_changes_fingerprint() -> None:
    v1 = PreregistrationReference("configs/studies/a.toml", 1, HASH_A)
    v2 = PreregistrationReference("configs/studies/a.toml", 2, HASH_A)
    assert configuration_fingerprint(_spec(preregistration_reference=v1)) != (
        configuration_fingerprint(_spec(preregistration_reference=v2))
    )


@pytest.mark.parametrize(
    "change",
    [
        {"experiment_id": "another-experiment"},
        {"run_kind": RunKind.DEVELOPMENT},
        {"input_data_references": (InputReference(InputKind.SYNTHETIC, "gen://x", "x"),)},
    ],
)
def test_non_scientific_fields_do_not_change_fingerprint(change: dict[str, Any]) -> None:
    assert configuration_fingerprint(_spec()) == configuration_fingerprint(_spec(**change))


def test_run_ids_are_unique_and_well_formed() -> None:
    ids = {new_run_id() for _ in range(2000)}
    assert len(ids) == 2000
    assert all(RUN_ID_PATTERN.match(i) for i in ids)


# ------------------------------------------------------------------- validação do RunSpec
@pytest.mark.parametrize(
    "change",
    [
        {"experiment_id": "Bad ID"},
        {"experiment_id": "ab"},
        {"parameters": {}},
        {"parameters": {"x": float("nan")}},
        {"random_seeds": {"root": -1}},
        {"random_seeds": {"root": 2**63}},
        {"random_seeds": {"root": True}},
        {"random_seeds": {"root": 1.0}},
        {"universe_reference": "  "},
    ],
)
def test_invalid_specs_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ContractViolationError):
        _spec(**change)


def test_file_inputs_require_content_hash() -> None:
    with pytest.raises(ContractViolationError, match="sha256"):
        InputReference(InputKind.FILE, "data/raw/x.zip", "x")
    assert InputReference(InputKind.FILE, "data/raw/x.zip", "x", HASH_A).sha256 == HASH_A
