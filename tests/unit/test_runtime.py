from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from frente_e.errors import ContractViolationError, ResourceBudgetError
from frente_e.runtime import (
    NATIVE_THREAD_VARIABLES,
    MachineResources,
    detect_resources,
    native_thread_env,
    plan_workers,
)
from frente_e.settings import RuntimeBudget

MIB = 2**20
GIB = 2**30


def _machine(cpus: int = 4, available: int | None = 4 * GIB) -> MachineResources:
    return MachineResources(
        platform="test",
        logical_cpus=cpus,
        total_memory_bytes=8 * GIB,
        available_memory_bytes=available,
    )


def test_detect_resources_reports_plausible_values() -> None:
    resources = detect_resources()
    assert resources.logical_cpus >= 1
    if resources.total_memory_bytes is not None and resources.available_memory_bytes is not None:
        assert 0 < resources.available_memory_bytes <= resources.total_memory_bytes


def test_memory_is_the_binding_limit() -> None:
    budget = RuntimeBudget(memory_fraction=0.5, max_workers=8, native_threads_per_worker=1)
    plan = plan_workers(budget, _machine(cpus=8, available=1 * GIB), per_worker_bytes=200 * MIB)
    assert plan.memory_budget_bytes == 512 * MIB
    assert plan.n_workers == 2  # floor(512 / 200)
    assert plan.limited_by == "memory"


def test_max_workers_caps_plan() -> None:
    budget = RuntimeBudget(memory_fraction=0.6, max_workers=2, native_threads_per_worker=1)
    plan = plan_workers(budget, _machine(cpus=4, available=8 * GIB), per_worker_bytes=100 * MIB)
    assert (plan.n_workers, plan.limited_by) == (2, "max_workers")


def test_cpu_count_caps_plan() -> None:
    budget = RuntimeBudget(memory_fraction=0.6, max_workers=16, native_threads_per_worker=1)
    plan = plan_workers(budget, _machine(cpus=3, available=64 * GIB), per_worker_bytes=100 * MIB)
    assert (plan.n_workers, plan.limited_by) == (3, "cpus")


def test_insufficient_memory_fails_explicitly() -> None:
    budget = RuntimeBudget(memory_fraction=0.6, max_workers=2, native_threads_per_worker=1)
    with pytest.raises(ResourceBudgetError):
        plan_workers(budget, _machine(available=100 * MIB), per_worker_bytes=200 * MIB)


def test_unknown_memory_fails_explicitly() -> None:
    budget = RuntimeBudget(memory_fraction=0.6, max_workers=2, native_threads_per_worker=1)
    with pytest.raises(ResourceBudgetError):
        plan_workers(budget, _machine(available=None), per_worker_bytes=MIB)


def test_non_positive_worker_size_is_rejected() -> None:
    budget = RuntimeBudget(memory_fraction=0.6, max_workers=2, native_threads_per_worker=1)
    with pytest.raises(ContractViolationError):
        plan_workers(budget, _machine(), per_worker_bytes=0)


@given(
    fraction=st.floats(min_value=0.01, max_value=1.0),
    max_workers=st.integers(1, 64),
    cpus=st.integers(1, 64),
    available=st.integers(1 * MIB, 256 * GIB),
    per_worker=st.integers(1 * MIB, 8 * GIB),
)
def test_plan_never_exceeds_any_limit(
    fraction: float, max_workers: int, cpus: int, available: int, per_worker: int
) -> None:
    budget = RuntimeBudget(
        memory_fraction=fraction, max_workers=max_workers, native_threads_per_worker=1
    )
    try:
        plan = plan_workers(budget, _machine(cpus=cpus, available=available), per_worker)
    except ResourceBudgetError:
        assert per_worker > fraction * available  # só falha quando nem 1 processo cabe
        return
    assert 1 <= plan.n_workers <= min(max_workers, cpus)
    assert plan.n_workers * per_worker <= plan.memory_budget_bytes <= fraction * available


def test_native_thread_env_covers_all_variables() -> None:
    env = native_thread_env(1)
    assert set(env) == set(NATIVE_THREAD_VARIABLES)
    assert set(env.values()) == {"1"}


def test_native_thread_env_rejects_zero() -> None:
    with pytest.raises(ContractViolationError):
        native_thread_env(0)
