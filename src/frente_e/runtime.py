"""Recursos da máquina e planejamento de paralelismo dentro do orçamento (DA-07).

Política (pedido do desenvolvedor, 2026-11-06):
- começar sequencial; paralelizar só com benefício medido;
- nº de processos limitado por ``max_workers``, CPUs e memória
  (``memory_fraction`` × RAM disponível);
- uma thread nativa por processo para evitar oversubscription.

Só stdlib: a memória é lida por ``GlobalMemoryStatusEx`` no Windows e por
``sysconf`` em POSIX.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

from frente_e.errors import ContractViolationError, ResourceBudgetError
from frente_e.settings import RuntimeBudget

# Variáveis que limitam pools de threads de BLAS/OpenMP/Polars. Precisam estar no
# ambiente ANTES de importar numpy/polars no processo de destino.
NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
)


@dataclass(frozen=True)
class MachineResources:
    platform: str
    logical_cpus: int
    total_memory_bytes: int | None
    available_memory_bytes: int | None


@dataclass(frozen=True)
class WorkerPlan:
    n_workers: int
    memory_budget_bytes: int
    per_worker_bytes: int
    limited_by: str  # "max_workers", "cpus" ou "memory"


def detect_resources() -> MachineResources:
    total, available = _memory_bytes()
    return MachineResources(
        platform=sys.platform,
        logical_cpus=os.cpu_count() or 1,
        total_memory_bytes=total,
        available_memory_bytes=available,
    )


def plan_workers(
    budget: RuntimeBudget, resources: MachineResources, per_worker_bytes: int
) -> WorkerPlan:
    """Número de processos que cabe no orçamento.

    ``per_worker_bytes`` é a memória de pico estimada por processo; deve vir de
    medição (benchmarks B2–B5), não de palpite.
    """
    if per_worker_bytes <= 0:
        raise ContractViolationError("per_worker_bytes deve ser positivo")
    if resources.available_memory_bytes is None:
        raise ResourceBudgetError(
            "Memória disponível desconhecida nesta plataforma; impossível respeitar o orçamento."
        )
    memory_budget = int(budget.memory_fraction * resources.available_memory_bytes)
    limits = {
        "max_workers": budget.max_workers,
        "cpus": resources.logical_cpus,
        "memory": memory_budget // per_worker_bytes,
    }
    limited_by = min(limits, key=lambda name: limits[name])
    n_workers = limits[limited_by]
    if n_workers < 1:
        raise ResourceBudgetError(
            f"Um processo exige ~{per_worker_bytes / 2**20:.0f} MiB, mas o orçamento é "
            f"{memory_budget / 2**20:.0f} MiB ({budget.memory_fraction:.0%} da RAM disponível)."
        )
    return WorkerPlan(
        n_workers=n_workers,
        memory_budget_bytes=memory_budget,
        per_worker_bytes=per_worker_bytes,
        limited_by=limited_by,
    )


def native_thread_env(n_threads: int) -> dict[str, str]:
    """Variáveis de ambiente que limitam threads nativas em processos filhos."""
    if n_threads < 1:
        raise ContractViolationError("n_threads deve ser >= 1")
    return dict.fromkeys(NATIVE_THREAD_VARIABLES, str(n_threads))


def _memory_bytes() -> tuple[int | None, int | None]:
    if sys.platform == "win32":
        return _memory_windows()
    return _memory_posix()


def _memory_windows() -> tuple[int | None, int | None]:
    if sys.platform != "win32":  # mantém o mypy coerente em outras plataformas
        return None, None

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _memory_posix() -> tuple[int | None, int | None]:
    if sys.platform == "win32":  # os.sysconf não existe no Windows
        return None, None
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None, None
    try:
        available: int | None = page * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        available = None  # ex.: macOS não expõe SC_AVPHYS_PAGES
    return total, available
