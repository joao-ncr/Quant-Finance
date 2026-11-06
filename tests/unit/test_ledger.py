"""Ledger de tentativas: registro, encadeamento, corrupção, interrupção e recuperação."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from frente_e.errors import (
    ContractViolationError,
    LedgerIntegrityError,
    LedgerLockedError,
    ReproducibilityError,
    TornTailError,
)
from frente_e.experiments.artifacts import ArtifactStore
from frente_e.experiments.canonical import canonical_bytes
from frente_e.experiments.identity import (
    PreregistrationReference,
    RunKind,
    RunSpec,
    configuration_fingerprint,
)
from frente_e.experiments.ledger import (
    LedgerHead,
    LedgerWriter,
    recover_torn_tail,
    verify_ledger,
)
from frente_e.experiments.records import GENESIS_HASH, RunStatus, compute_record_hash

ENV = {"python": "3.13-test", "packages": {"numpy": "test"}}
NO_GIT = {"vcs": "git", "commit": None, "dirty": None, "detail": "teste"}


class StepClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


def _spec(**overrides: Any) -> RunSpec:
    base: dict[str, Any] = {
        "experiment_id": "ledger-test",
        "run_kind": RunKind.SYNTHETIC,
        "parameters": {"family": "momentum", "formation": 63},
        "random_seeds": {"root": 7},
    }
    base.update(overrides)
    return RunSpec(**base)


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "experiments" / "ledger.jsonl"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def _writer(path: Path) -> LedgerWriter:
    return LedgerWriter(path, clock=StepClock())


def _complete_runs(path: Path, store: ArtifactStore, n: int) -> list[str]:
    ids = []
    with _writer(path) as ledger:
        for i in range(n):
            with ledger.run(
                _spec(random_seeds={"root": i}), software_environment=ENV, code_revision=NO_GIT
            ) as run:
                run.set_results(
                    [
                        store.put_bytes(
                            f"r{i}".encode(),
                            suffix=".txt",
                            media_type="text/plain",
                            description="t",
                        )
                    ]
                )
            ids.append(run.run_id)
    return ids


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().split(b"\n")[:-1]


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"".join(line + b"\n" for line in lines))


def _rechain(records: list[dict[str, Any]]) -> list[bytes]:
    """Reescreve a cadeia inteira de forma consistente (o 'ataque' que o hash não impede)."""
    previous = GENESIS_HASH
    out = []
    for i, record in enumerate(records):
        record = {**record, "sequence": i, "previous_record_hash": previous}
        record["record_hash"] = compute_record_hash(record)
        previous = record["record_hash"]
        out.append(canonical_bytes(record))
    return out


# ------------------------------------------------------------------- caminho feliz
def test_valid_ledger_verifies(ledger_path: Path, store: ArtifactStore) -> None:
    ids = _complete_runs(ledger_path, store, 3)
    result = verify_ledger(ledger_path)
    assert result.head.n_records == 6
    assert result.incomplete_runs == ()
    assert result.count_by_status()["completed"] == 3
    assert set(result.runs) == set(ids)


def test_empty_or_missing_ledger_is_valid(ledger_path: Path) -> None:
    assert verify_ledger(ledger_path).head == LedgerHead(0, GENESIS_HASH)


def test_records_hold_full_configuration_once(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    started, completed = (json.loads(line) for line in _lines(ledger_path))
    assert started["status"] == "started"
    assert started["parameters"] == {"family": "momentum", "formation": 63}
    assert started["random_seeds"] == {"root": 0}
    assert started["code_revision"] == NO_GIT  # ausência de commit registrada, não inventada
    assert started["previous_record_hash"] == GENESIS_HASH
    assert completed["status"] == "completed"
    assert completed["parameters"] is None
    assert completed["results_reference"][0]["sha256"]
    assert completed["configuration_fingerprint"] == started["configuration_fingerprint"]
    assert completed["previous_record_hash"] == started["record_hash"]


def test_same_configuration_repeated_gets_new_run_id(ledger_path: Path) -> None:
    with _writer(ledger_path) as ledger:
        a = ledger.start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)
        b = ledger.start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)
        runs = ledger.runs
    assert a != b
    assert runs[a].configuration_fingerprint == runs[b].configuration_fingerprint


def test_different_seeds_get_different_fingerprints(ledger_path: Path) -> None:
    with _writer(ledger_path) as ledger:
        a = ledger.start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)
        b = ledger.start_run(
            _spec(random_seeds={"root": 8}), software_environment=ENV, code_revision=NO_GIT
        )
        runs = ledger.runs
    assert runs[a].configuration_fingerprint != runs[b].configuration_fingerprint
    assert runs[a].configuration_fingerprint == configuration_fingerprint(_spec())


def test_appending_after_reopen_continues_chain(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    _complete_runs(ledger_path, store, 1)
    assert verify_ledger(ledger_path).head.n_records == 4


# ------------------------------------------------------------------- falhas e interrupções
def test_exception_is_recorded_as_failed_and_reraised(ledger_path: Path) -> None:
    with (
        _writer(ledger_path) as ledger,
        pytest.raises(ZeroDivisionError),
        ledger.run(_spec(), software_environment=ENV, code_revision=NO_GIT),
    ):
        _ = 1 / 0
    last = json.loads(_lines(ledger_path)[-1])
    assert last["status"] == "failed"
    assert last["error_information"]["type"] == "ZeroDivisionError"
    assert "Traceback" in last["error_information"]["traceback"]
    assert verify_ledger(ledger_path).count_by_status()["failed"] == 1


def test_keyboard_interrupt_is_recorded_as_aborted(ledger_path: Path) -> None:
    with (
        _writer(ledger_path) as ledger,
        pytest.raises(KeyboardInterrupt),
        ledger.run(_spec(), software_environment=ENV, code_revision=NO_GIT),
    ):
        raise KeyboardInterrupt
    assert json.loads(_lines(ledger_path)[-1])["status"] == "aborted"


def test_missing_results_is_failure_not_success(ledger_path: Path) -> None:
    with (
        _writer(ledger_path) as ledger,
        pytest.raises(ContractViolationError),
        ledger.run(_spec(), software_environment=ENV, code_revision=NO_GIT),
    ):
        pass
    assert json.loads(_lines(ledger_path)[-1])["error_information"]["type"] == "MissingResults"


def test_interrupted_execution_is_detected_and_can_be_aborted(ledger_path: Path) -> None:
    with _writer(ledger_path) as ledger:  # processo "morre" sem registro terminal
        run_id = ledger.start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)
    result = verify_ledger(ledger_path)
    assert result.incomplete_runs == (run_id,)
    with _writer(ledger_path) as ledger:
        assert ledger.incomplete_runs() == (run_id,)
        ledger.abort_run(run_id, "processo interrompido; encontrado na verificação")
    assert verify_ledger(ledger_path).incomplete_runs == ()


def test_terminal_twice_is_rejected(ledger_path: Path) -> None:
    with _writer(ledger_path) as ledger:
        run_id = ledger.start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)
        ledger.complete_run(run_id, [])
        with pytest.raises(ContractViolationError, match="já está"):
            ledger.fail_run(run_id, {"type": "X", "message": "y"})


# ------------------------------------------------------------------- corrupção
def test_modified_line_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 2)
    lines = _lines(ledger_path)
    lines[0] = lines[0].replace(b'"formation":63', b'"formation":64')
    _write_lines(ledger_path, lines)
    with pytest.raises(LedgerIntegrityError, match="linha 1: record_hash"):
        verify_ledger(ledger_path)


def test_modified_line_with_recomputed_hash_breaks_chain(
    ledger_path: Path, store: ArtifactStore
) -> None:
    _complete_runs(ledger_path, store, 2)
    lines = _lines(ledger_path)
    record = json.loads(lines[0])
    record["parameters"]["formation"] = 64
    record["record_hash"] = compute_record_hash(record)
    lines[0] = canonical_bytes(record)
    _write_lines(ledger_path, lines)
    with pytest.raises(LedgerIntegrityError, match="linha 2: previous_record_hash"):
        verify_ledger(ledger_path)


def test_invalid_previous_hash_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    lines = _lines(ledger_path)
    record = json.loads(lines[1])
    record["previous_record_hash"] = "f" * 64
    record["record_hash"] = compute_record_hash(record)
    lines[1] = canonical_bytes(record)
    _write_lines(ledger_path, lines)
    with pytest.raises(LedgerIntegrityError, match="previous_record_hash"):
        verify_ledger(ledger_path)


def test_removed_middle_line_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 2)
    lines = _lines(ledger_path)
    _write_lines(ledger_path, lines[:1] + lines[2:])
    with pytest.raises(LedgerIntegrityError, match="sequence"):
        verify_ledger(ledger_path)


def test_reordered_lines_are_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 2)
    lines = _lines(ledger_path)
    lines[1], lines[2] = lines[2], lines[1]
    _write_lines(ledger_path, lines)
    with pytest.raises(LedgerIntegrityError):
        verify_ledger(ledger_path)


def test_duplicate_run_id_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 2)
    records = [json.loads(line) for line in _lines(ledger_path)]
    records[2]["run_id"] = records[0]["run_id"]  # segundo started reusa o run_id
    records[3]["run_id"] = records[0]["run_id"]
    _write_lines(ledger_path, _rechain(records))
    with pytest.raises(LedgerIntegrityError, match="run_id duplicado"):
        verify_ledger(ledger_path)


def test_terminal_without_started_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    records = [json.loads(line) for line in _lines(ledger_path)]
    _write_lines(ledger_path, _rechain(records[1:]))
    with pytest.raises(LedgerIntegrityError, match="sem started"):
        verify_ledger(ledger_path)


def test_unsupported_schema_version_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    records = [json.loads(line) for line in _lines(ledger_path)]
    records[0]["schema_version"] = 99
    _write_lines(ledger_path, _rechain(records))
    with pytest.raises(LedgerIntegrityError, match="schema_version não suportada"):
        verify_ledger(ledger_path)


def test_non_canonical_formatting_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    lines = _lines(ledger_path)
    lines[0] = json.dumps(json.loads(lines[0]), indent=1).replace("\n", " ").encode()
    _write_lines(ledger_path, lines)
    with pytest.raises(LedgerIntegrityError, match="forma canônica"):
        verify_ledger(ledger_path)


def test_garbage_line_is_detected(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    _write_lines(ledger_path, [*_lines(ledger_path), b"not json"])
    with pytest.raises(LedgerIntegrityError, match="JSON inválido"):
        verify_ledger(ledger_path)


def test_writer_refuses_corrupted_ledger(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 2)
    lines = _lines(ledger_path)
    _write_lines(ledger_path, lines[:1] + lines[2:])
    with pytest.raises(LedgerIntegrityError), _writer(ledger_path):
        pytest.fail("o escritor não deveria abrir um ledger corrompido")


# ------------------------------------------------------------------- limitações e âncora
def test_full_rewrite_passes_verification_but_fails_anchor(
    ledger_path: Path, store: ArtifactStore
) -> None:
    _complete_runs(ledger_path, store, 2)
    anchor = verify_ledger(ledger_path).head
    records = [json.loads(line) for line in _lines(ledger_path)]
    records[0]["parameters"]["formation"] = 999
    _write_lines(ledger_path, _rechain(records))
    verify_ledger(ledger_path)  # limitação documentada: a cadeia recalculada é consistente
    with pytest.raises(LedgerIntegrityError, match="âncora"):
        verify_ledger(ledger_path, expected_head=anchor)


def test_removed_last_records_detected_only_by_anchor(
    ledger_path: Path, store: ArtifactStore
) -> None:
    _complete_runs(ledger_path, store, 2)
    anchor = verify_ledger(ledger_path).head
    _write_lines(ledger_path, _lines(ledger_path)[:-1])
    verify_ledger(ledger_path)  # limitação documentada
    with pytest.raises(LedgerIntegrityError, match="removidos"):
        verify_ledger(ledger_path, expected_head=anchor)


def test_anchor_on_prefix_accepts_later_appends(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    anchor = verify_ledger(ledger_path).head
    _complete_runs(ledger_path, store, 1)
    verify_ledger(ledger_path, expected_head=anchor)


def test_head_text_roundtrip() -> None:
    head = LedgerHead(5, "ab" * 32)
    assert LedgerHead.parse(str(head)) == head
    with pytest.raises(ContractViolationError):
        LedgerHead.parse("5-abc")


# ------------------------------------------------------------------- cauda truncada
def test_truncated_last_line_blocks_writer_until_recovery(
    ledger_path: Path, store: ArtifactStore
) -> None:
    _complete_runs(ledger_path, store, 1)
    data = ledger_path.read_bytes()
    ledger_path.write_bytes(data + b'{"schema_version":1,"seq')  # escrita interrompida
    with pytest.raises(TornTailError):
        verify_ledger(ledger_path)
    with pytest.raises(TornTailError), _writer(ledger_path):
        pytest.fail("o escritor não deveria abrir com cauda truncada")

    quarantine = recover_torn_tail(ledger_path, clock=StepClock())
    assert quarantine is not None
    assert quarantine.read_bytes() == b'{"schema_version":1,"seq'
    assert ledger_path.read_bytes() == data

    _complete_runs(ledger_path, store, 1)  # continua a cadeia normalmente
    assert verify_ledger(ledger_path).head.n_records == 4


def test_recovery_is_noop_on_clean_ledger(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 1)
    before = ledger_path.read_bytes()
    assert recover_torn_tail(ledger_path) is None
    assert ledger_path.read_bytes() == before


def test_recovery_refuses_when_prefix_is_corrupt(ledger_path: Path, store: ArtifactStore) -> None:
    _complete_runs(ledger_path, store, 2)
    lines = _lines(ledger_path)
    corrupted = b"".join(line + b"\n" for line in [lines[0], lines[2]]) + b'{"partial'
    ledger_path.write_bytes(corrupted)
    with pytest.raises(LedgerIntegrityError):
        recover_torn_tail(ledger_path)
    assert ledger_path.read_bytes() == corrupted  # nada alterado


# ------------------------------------------------------------------- escritor único
def test_second_writer_is_refused(ledger_path: Path) -> None:
    with _writer(ledger_path), pytest.raises(LedgerLockedError), _writer(ledger_path):
        pytest.fail("dois escritores simultâneos")


def test_lock_is_released_after_exit(ledger_path: Path) -> None:
    with _writer(ledger_path):
        pass
    with _writer(ledger_path) as ledger:
        assert ledger.head.n_records == 0


def test_external_modification_between_appends_is_detected(ledger_path: Path) -> None:
    with _writer(ledger_path) as ledger:
        ledger.start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)
        with ledger_path.open("ab") as fh:
            fh.write(b"intruso\n")
        with pytest.raises(LedgerIntegrityError, match="alterado fora"):
            ledger.start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)


def test_writer_outside_context_is_rejected(ledger_path: Path) -> None:
    with pytest.raises(ContractViolationError, match="fora do contexto"):
        _writer(ledger_path).start_run(_spec(), software_environment=ENV, code_revision=NO_GIT)


# ------------------------------------------------------------------- reprodutibilidade
PREREG = PreregistrationReference("configs/studies/x.toml", 1, "c" * 64)


@pytest.mark.parametrize(
    ("revision", "prereg", "message"),
    [
        (NO_GIT, PREREG, "sem commit"),
        ({"vcs": "git", "commit": "a" * 40, "dirty": True, "detail": "1"}, PREREG, "alterações"),
        (
            {"vcs": "git", "commit": "a" * 40, "dirty": False, "detail": "limpo"},
            None,
            "pré-registro",
        ),
    ],
)
def test_real_data_study_requires_reproducibility(
    ledger_path: Path,
    revision: dict[str, Any],
    prereg: PreregistrationReference | None,
    message: str,
) -> None:
    spec = _spec(run_kind=RunKind.STUDY, preregistration_reference=prereg)
    with _writer(ledger_path) as ledger, pytest.raises(ReproducibilityError, match=message):
        ledger.start_run(spec, software_environment=ENV, code_revision=revision)
    assert verify_ledger(ledger_path).head.n_records == 0  # nada registrado


def test_real_data_study_allowed_when_reproducible(ledger_path: Path) -> None:
    spec = _spec(run_kind=RunKind.STUDY, preregistration_reference=PREREG)
    clean = {"vcs": "git", "commit": "a" * 40, "dirty": False, "detail": "limpo"}
    with _writer(ledger_path) as ledger:
        run_id = ledger.start_run(spec, software_environment=ENV, code_revision=clean)
        assert ledger.runs[run_id].status is RunStatus.STARTED
