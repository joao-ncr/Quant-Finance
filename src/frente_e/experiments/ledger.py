"""Ledger de tentativas: JSONL append-only encadeado por SHA-256 (DA-03).

Garantias:
- cada linha é a forma canônica de um registro (``records.validate_record``) + ``\\n``;
- a cadeia (``previous_record_hash``), a numeração (``sequence``) e o hash de cada linha
  detectam alteração, remoção, inserção e reordenação **no meio** do arquivo;
- uma última linha sem ``\\n`` é uma escrita interrompida (:class:`TornTailError`): o
  escritor se recusa a continuar até a recuperação explícita (:func:`recover_torn_tail`);
- um único escritor por vez (trava de arquivo do SO, liberada se o processo morrer);
- falhas e interrupções viram registros (``failed``/``aborted``); nada é descartado.

Limitações (documentadas em docs/ARCHITECTURE_DECISIONS.md, DA-03):
- quem reescreve o arquivo inteiro e recalcula toda a cadeia não é detectado pela
  verificação isolada; a remoção das últimas linhas também não. As duas são detectadas
  comparando com uma âncora externa (:class:`LedgerHead`), por exemplo o ledger
  versionado no Git ou o head citado num relatório;
- a durabilidade depende de ``fsync`` e do sistema de arquivos.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

from frente_e.errors import (
    ContractViolationError,
    LedgerIntegrityError,
    LedgerLockedError,
    TornTailError,
)
from frente_e.experiments.artifacts import ArtifactRef
from frente_e.experiments.canonical import JsonValue, canonical_bytes, to_canonical
from frente_e.experiments.identity import (
    RunKind,
    RunSpec,
    configuration_fingerprint,
    new_run_id,
)
from frente_e.experiments.provenance import require_reproducible
from frente_e.experiments.records import (
    GENESIS_HASH,
    SCHEMA_VERSION,
    TERMINAL_STATUSES,
    RecordShapeError,
    RunStatus,
    compute_record_hash,
    validate_record,
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class LedgerHead:
    """Âncora externa: nº de registros e hash do último (GENESIS se vazio)."""

    n_records: int
    last_record_hash: str

    def __str__(self) -> str:
        return f"{self.n_records}:{self.last_record_hash}"

    @classmethod
    def parse(cls, text: str) -> LedgerHead:
        count, sep, digest = text.partition(":")
        if not sep or not count.isdigit() or len(digest) != 64:
            raise ContractViolationError(f"head inválido {text!r}; formato N:SHA256")
        return cls(int(count), digest)


@dataclass(frozen=True)
class RunState:
    run_id: str
    experiment_id: str
    run_kind: str
    configuration_fingerprint: str
    status: RunStatus
    started_sequence: int


@dataclass(frozen=True)
class VerificationResult:
    head: LedgerHead
    size_bytes: int
    runs: Mapping[str, RunState]

    @property
    def incomplete_runs(self) -> tuple[str, ...]:
        return tuple(r.run_id for r in self.runs.values() if r.status is RunStatus.STARTED)

    def count_by_status(self) -> dict[str, int]:
        counts = {s.value: 0 for s in RunStatus}
        for run in self.runs.values():
            counts[run.status.value] += 1
        return counts


# --------------------------------------------------------------------------- leitura


def verify_ledger(path: Path, expected_head: LedgerHead | None = None) -> VerificationResult:
    """Verifica o arquivo inteiro. Arquivo inexistente equivale a ledger vazio."""
    data = path.read_bytes() if path.exists() else b""
    result = _scan(data)
    if expected_head is not None:
        _check_anchor(data, result, expected_head)
    return result


def _scan(data: bytes) -> VerificationResult:
    if data and not data.endswith(b"\n"):
        last_newline = data.rfind(b"\n")
        line_no = data.count(b"\n") + 1
        raise TornTailError(
            f"última linha incompleta ({len(data) - last_newline - 1} bytes sem '\\n'); "
            "use 'frente-e ledger recover-tail'",
            line=line_no,
        )
    runs: dict[str, RunState] = {}
    previous = GENESIS_HASH
    lines = data.split(b"\n")[:-1] if data else []
    for index, raw in enumerate(lines):
        record = _parse_line(raw, index + 1)
        runs[record["run_id"]] = _transition(runs, record, index, previous, index + 1)
        previous = record["record_hash"]
    return VerificationResult(LedgerHead(len(lines), previous), len(data), runs)


def _parse_line(raw: bytes, line_no: int) -> dict[str, Any]:
    try:
        record = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LedgerIntegrityError(f"JSON inválido ({exc})", line=line_no) from exc
    if not isinstance(record, dict):
        raise LedgerIntegrityError("registro não é um objeto JSON", line=line_no)
    try:
        validate_record(record)
        canonical = canonical_bytes(record)
    except (RecordShapeError, ContractViolationError) as exc:
        raise LedgerIntegrityError(str(exc), line=line_no) from exc
    if canonical != raw:
        raise LedgerIntegrityError("linha não está em forma canônica", line=line_no)
    return record


def _transition(
    runs: Mapping[str, RunState],
    record: Mapping[str, Any],
    expected_sequence: int,
    expected_previous: str,
    line_no: int,
) -> RunState:
    """Valida cadeia e transição de estado; devolve o novo estado da execução."""
    if record["sequence"] != expected_sequence:
        raise LedgerIntegrityError(
            f"sequence {record['sequence']} != {expected_sequence} (linha removida, "
            "inserida ou reordenada)",
            line=line_no,
        )
    if record["previous_record_hash"] != expected_previous:
        raise LedgerIntegrityError("previous_record_hash não encadeia", line=line_no)
    if compute_record_hash(record) != record["record_hash"]:
        raise LedgerIntegrityError("record_hash não confere com o conteúdo", line=line_no)

    status = RunStatus(record["status"])
    run_id = record["run_id"]
    existing = runs.get(run_id)
    if status is RunStatus.STARTED:
        if existing is not None:
            raise LedgerIntegrityError(f"run_id duplicado: {run_id}", line=line_no)
        return RunState(
            run_id=run_id,
            experiment_id=record["experiment_id"],
            run_kind=record["run_kind"],
            configuration_fingerprint=record["configuration_fingerprint"],
            status=status,
            started_sequence=record["sequence"],
        )
    if existing is None:
        raise LedgerIntegrityError(f"registro {status} sem started: {run_id}", line=line_no)
    if existing.status in TERMINAL_STATUSES:
        raise LedgerIntegrityError(f"execução {run_id} já encerrada", line=line_no)
    for name in ("experiment_id", "run_kind", "configuration_fingerprint"):
        if record[name] != getattr(existing, name):
            raise LedgerIntegrityError(f"{name} difere do registro started", line=line_no)
    return RunState(
        run_id=run_id,
        experiment_id=existing.experiment_id,
        run_kind=existing.run_kind,
        configuration_fingerprint=existing.configuration_fingerprint,
        status=status,
        started_sequence=existing.started_sequence,
    )


def _check_anchor(data: bytes, result: VerificationResult, anchor: LedgerHead) -> None:
    if anchor.n_records > result.head.n_records:
        raise LedgerIntegrityError(
            f"ledger tem {result.head.n_records} registros, a âncora exige ao menos "
            f"{anchor.n_records} (registros finais removidos?)"
        )
    if anchor.n_records == 0:
        if anchor.last_record_hash != GENESIS_HASH:
            raise LedgerIntegrityError("âncora vazia deve usar o hash GENESIS")
        return
    line = data.split(b"\n")[anchor.n_records - 1]
    if json.loads(line)["record_hash"] != anchor.last_record_hash:
        raise LedgerIntegrityError(
            f"registro {anchor.n_records} não confere com a âncora (cadeia reescrita?)"
        )


def _reject_constant(name: str) -> None:
    raise ValueError(f"constante JSON não permitida: {name}")


# --------------------------------------------------------------------------- recuperação


def recover_torn_tail(path: Path, clock: Clock = utc_now) -> Path | None:
    """Move a última linha incompleta para um arquivo de quarentena e trunca o ledger.

    Só age se o restante do ledger for íntegro. Devolve o caminho da quarentena,
    ou ``None`` se não havia cauda incompleta. Nunca mexe em linhas completas.
    """
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    _scan(data[:cut])  # o prefixo precisa ser íntegro; caso contrário, nada é alterado
    stamp = clock().astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine = path.with_name(f"{path.name}.torn-{stamp}")
    quarantine.write_bytes(data[cut:])
    with path.open("r+b") as fh:
        fh.truncate(cut)
        fh.flush()
        os.fsync(fh.fileno())
    return quarantine


# --------------------------------------------------------------------------- escrita


class _ExclusiveLock:
    """Trava não bloqueante do SO num arquivo auxiliar; liberada ao fechar o handle."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: IO[bytes] | None = None

    def acquire(self) -> None:
        fh = self._path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise LedgerLockedError(f"Ledger em uso por outro escritor: {self._path}") from exc
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class RunHandle:
    """Entregue pelo contexto :meth:`LedgerWriter.run`; recebe os resultados."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._results: tuple[ArtifactRef, ...] | None = None

    def set_results(self, results: Sequence[ArtifactRef]) -> None:
        if self._results is not None:
            raise ContractViolationError("resultados já registrados para esta execução")
        self._results = tuple(results)

    @property
    def results(self) -> tuple[ArtifactRef, ...] | None:
        return self._results


class LedgerWriter:
    """Único escritor do ledger. Use como context manager."""

    def __init__(self, path: Path, clock: Clock = utc_now) -> None:
        self.path = path
        self._clock = clock
        self._lock = _ExclusiveLock(path.with_name(path.name + ".lock"))
        self._state: VerificationResult | None = None
        self._runs: dict[str, RunState] = {}

    # -- ciclo de vida
    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        try:
            self.path.touch(exist_ok=True)
            self._state = verify_ledger(self.path)
            self._runs = dict(self._state.runs)
        except BaseException:
            self._lock.release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._lock.release()
        self._state = None

    @property
    def head(self) -> LedgerHead:
        return self._require_open().head

    @property
    def runs(self) -> Mapping[str, RunState]:
        return dict(self._runs)

    def incomplete_runs(self) -> tuple[str, ...]:
        return tuple(r.run_id for r in self._runs.values() if r.status is RunStatus.STARTED)

    # -- operações
    def start_run(
        self,
        spec: RunSpec,
        *,
        software_environment: Mapping[str, Any],
        code_revision: Mapping[str, Any],
    ) -> str:
        revision = to_canonical(code_revision, "code_revision")
        assert isinstance(revision, dict)
        require_reproducible(spec, revision)
        run_id = new_run_id()
        if run_id in self._runs:  # colisão de UUID4: não deve ocorrer; falha explícita
            raise LedgerIntegrityError(f"run_id gerado já existe: {run_id}")
        self._append(
            {
                "experiment_id": spec.experiment_id,
                "run_id": run_id,
                "run_kind": str(spec.run_kind),
                "configuration_fingerprint": configuration_fingerprint(spec),
                "status": RunStatus.STARTED.value,
                "parameters": spec.parameters,
                "universe_reference": spec.universe_reference,
                "preregistration_reference": (
                    None
                    if spec.preregistration_reference is None
                    else spec.preregistration_reference.to_json()
                ),
                "random_seeds": dict(spec.random_seeds),
                "software_environment": software_environment,
                "code_revision": revision,
                "input_data_references": [ref.to_json() for ref in spec.input_data_references],
                "results_reference": None,
                "error_information": None,
            }
        )
        return run_id

    def complete_run(self, run_id: str, results: Sequence[ArtifactRef]) -> None:
        self._terminal(run_id, RunStatus.COMPLETED, [r.to_json() for r in results], None)

    def fail_run(self, run_id: str, error: Mapping[str, Any]) -> None:
        self._terminal(run_id, RunStatus.FAILED, None, error)

    def abort_run(self, run_id: str, reason: str) -> None:
        self._terminal(run_id, RunStatus.ABORTED, None, {"type": "Aborted", "message": reason})

    @contextmanager
    def run(
        self,
        spec: RunSpec,
        *,
        software_environment: Mapping[str, Any],
        code_revision: Mapping[str, Any],
    ) -> Iterator[RunHandle]:
        """Registra started; ao sair, completed/failed/aborted. Exceções são repassadas."""
        run_id = self.start_run(
            spec, software_environment=software_environment, code_revision=code_revision
        )
        handle = RunHandle(run_id)
        try:
            yield handle
        except Exception as exc:
            self.fail_run(run_id, error_information(exc))
            raise
        except BaseException as exc:  # KeyboardInterrupt, SystemExit
            self.abort_run(run_id, f"interrompido por {type(exc).__name__}")
            raise
        if handle.results is None:
            message = "a execução terminou sem registrar resultados (set_results)"
            self.fail_run(run_id, {"type": "MissingResults", "message": message})
            raise ContractViolationError(message)
        self.complete_run(run_id, handle.results)

    # -- internos
    def _terminal(
        self,
        run_id: str,
        status: RunStatus,
        results: list[dict[str, JsonValue]] | None,
        error: Mapping[str, Any] | None,
    ) -> None:
        state = self._runs.get(run_id)
        if state is None:
            raise ContractViolationError(f"run_id desconhecido: {run_id}")
        if state.status is not RunStatus.STARTED:
            raise ContractViolationError(f"execução {run_id} já está {state.status}")
        self._append(
            {
                "experiment_id": state.experiment_id,
                "run_id": run_id,
                "run_kind": state.run_kind,
                "configuration_fingerprint": state.configuration_fingerprint,
                "status": status.value,
                "parameters": None,
                "universe_reference": None,
                "preregistration_reference": None,
                "random_seeds": None,
                "software_environment": None,
                "code_revision": None,
                "input_data_references": None,
                "results_reference": results,
                "error_information": error,
            }
        )

    def _append(self, fields: Mapping[str, Any]) -> None:
        state = self._require_open()
        size_now = self.path.stat().st_size
        if size_now != state.size_bytes:
            raise LedgerIntegrityError(
                f"ledger alterado fora deste escritor ({state.size_bytes} -> {size_now} bytes)"
            )
        canonical_fields = to_canonical(fields, "record")
        assert isinstance(canonical_fields, dict)
        record: dict[str, Any] = {
            **canonical_fields,
            "schema_version": SCHEMA_VERSION,
            "sequence": state.head.n_records,
            "timestamp_utc": _format_timestamp(self._clock()),
            "previous_record_hash": state.head.last_record_hash,
        }
        record["record_hash"] = compute_record_hash(record)
        line_no = state.head.n_records + 1
        try:
            validate_record(record)
        except RecordShapeError as exc:
            raise ContractViolationError(f"registro inválido: {exc}") from exc
        new_run_state = _transition(
            self._runs, record, state.head.n_records, state.head.last_record_hash, line_no
        )
        line = canonical_bytes(record) + b"\n"
        with self.path.open("ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._runs[new_run_state.run_id] = new_run_state
        self._state = VerificationResult(
            LedgerHead(line_no, record["record_hash"]),
            state.size_bytes + len(line),
            self._runs,
        )

    def _require_open(self) -> VerificationResult:
        if self._state is None:
            raise ContractViolationError("LedgerWriter fora do contexto 'with'")
        return self._state


def error_information(exc: BaseException) -> dict[str, JsonValue]:
    return {
        "type": type(exc).__qualname__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def _format_timestamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ContractViolationError("o relógio do ledger deve devolver datetime com fuso")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


__all__ = [
    "LedgerHead",
    "LedgerWriter",
    "RunHandle",
    "RunKind",
    "RunState",
    "VerificationResult",
    "error_information",
    "recover_torn_tail",
    "verify_ledger",
]
