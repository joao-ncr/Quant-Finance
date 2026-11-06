"""Configuração de logging (stdlib).

Os módulos usam ``logging.getLogger(__name__)``; só os pontos de entrada (CLI,
runner) chamam :func:`configure_logging`. Os horários são em UTC (ISO 8601).
"""

from __future__ import annotations

import logging
import time
from typing import TextIO

ROOT_LOGGER_NAME = "frente_e"
_HANDLER_MARK = "_frente_e_handler"
_FORMAT = "%(asctime)s %(levelname)-8s %(processName)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_formatter() -> logging.Formatter:
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    formatter.converter = time.gmtime
    return formatter


def configure_logging(
    level: int | str = logging.INFO, stream: TextIO | None = None
) -> logging.Logger:
    """Configura o logger raiz do pacote. Idempotente: não duplica handlers."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_utc_formatter())
    setattr(handler, _HANDLER_MARK, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
