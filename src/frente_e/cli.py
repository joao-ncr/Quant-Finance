"""Ponto de entrada único: ``python -m frente_e <comando>``."""

from __future__ import annotations

import argparse
import io
import logging
import platform
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from frente_e import __version__
from frente_e.errors import ContractViolationError, FrenteEError
from frente_e.experiments.ledger import LedgerHead, LedgerWriter, recover_torn_tail, verify_ledger
from frente_e.logging_setup import configure_logging
from frente_e.runtime import detect_resources
from frente_e.settings import load_settings

logger = logging.getLogger(__name__)


def _force_utf8_streams() -> None:
    """Saída sempre em UTF-8.

    No Windows com Python 3.13, stdout/stderr redirecionados usam a codificação do
    sistema (cp1252), o que corrompe acentos para quem lê a saída como UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_streams()
    parser = argparse.ArgumentParser(
        prog="frente-e", description="Frente E: pesquisa DSR/PBO na B3."
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info", help="Ambiente, recursos da máquina e orçamento de execução.")
    info.add_argument("--root", type=Path, default=None, help="Raiz do projeto (padrão: busca).")

    ledger = sub.add_parser("ledger", help="Verificação e manutenção do ledger de tentativas.")
    ledger.add_argument("--root", type=Path, default=None)
    ledger_sub = ledger.add_subparsers(dest="ledger_command", required=True)
    verify = ledger_sub.add_parser("verify", help="Verifica cadeia, schema e transições.")
    verify.add_argument("--expected-head", default=None, help="Âncora N:SHA256 a conferir.")
    ledger_sub.add_parser("head", help="Imprime a âncora atual (N:SHA256).")
    ledger_sub.add_parser("incomplete", help="Lista execuções sem registro terminal.")
    ledger_sub.add_parser("recover-tail", help="Coloca em quarentena uma última linha truncada.")
    abort = ledger_sub.add_parser("abort", help="Registra 'aborted' para uma execução incompleta.")
    abort.add_argument("run_id")
    abort.add_argument("--reason", required=True)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    try:
        if args.command == "info":
            return _cmd_info(args.root)
        if args.command == "ledger":
            return _cmd_ledger(args)
    except FrenteEError as exc:
        logger.error("%s", exc)
        return 2
    parser.error(f"comando desconhecido: {args.command}")
    return 2  # inalcançável: parser.error encerra o processo


def _cmd_ledger(args: argparse.Namespace) -> int:
    path = load_settings(args.root).paths.ledger
    command = args.ledger_command
    if command == "verify":
        anchor = None if args.expected_head is None else LedgerHead.parse(args.expected_head)
        result = verify_ledger(path, expected_head=anchor)
        counts = ", ".join(f"{k}={v}" for k, v in result.count_by_status().items())
        print(f"OK: {result.head.n_records} registros; execuções: {counts}")
        print(f"head: {result.head}")
        if result.incomplete_runs:
            print(f"ATENÇÃO: {len(result.incomplete_runs)} execução(ões) incompleta(s)")
        return 0
    if command == "head":
        print(verify_ledger(path).head)
        return 0
    if command == "incomplete":
        for run_id in verify_ledger(path).incomplete_runs:
            print(run_id)
        return 0
    if command == "recover-tail":
        quarantine = recover_torn_tail(path)
        print("nenhuma cauda incompleta" if quarantine is None else f"quarentena: {quarantine}")
        return 0
    if command == "abort":
        with LedgerWriter(path) as writer:
            writer.abort_run(args.run_id, args.reason)
        print(f"abortada: {args.run_id}")
        return 0
    raise ContractViolationError(f"subcomando desconhecido: {command}")


def _cmd_info(root: Path | None) -> int:
    settings = load_settings(root)
    resources = detect_resources()
    budget = settings.runtime

    def gib(n: int | None) -> str:
        return "desconhecida" if n is None else f"{n / 2**30:.2f} GiB"

    memory_budget = (
        None
        if resources.available_memory_bytes is None
        else int(budget.memory_fraction * resources.available_memory_bytes)
    )
    lines = [
        f"frente-e {__version__}",
        f"python {sys.version.split()[0]} ({platform.platform()})",
        f"numpy {version('numpy')}",
        f"raiz do projeto: {settings.paths.root}",
        f"CPUs lógicas: {resources.logical_cpus}",
        f"RAM total: {gib(resources.total_memory_bytes)}",
        f"RAM disponível agora: {gib(resources.available_memory_bytes)}",
        f"orçamento de memória ({budget.memory_fraction:.0%} da disponível): {gib(memory_budget)}",
        f"max_workers: {budget.max_workers} | threads nativas por processo: "
        f"{budget.native_threads_per_worker}",
    ]
    print("\n".join(lines))
    return 0
