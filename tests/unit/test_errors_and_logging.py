from __future__ import annotations

import io
import logging
import re

import pytest

from frente_e.errors import ContractViolationError, FrenteEError, PendingDecisionError
from frente_e.logging_setup import ROOT_LOGGER_NAME, configure_logging


def test_pending_decision_error_names_the_decision() -> None:
    err = PendingDecisionError("DE-03", "método de N efetivo")
    assert err.decision_id == "DE-03"
    assert "DE-03" in str(err)
    assert "PROJECT_DECISIONS.md" in str(err)
    assert isinstance(err, FrenteEError)
    assert isinstance(err, NotImplementedError)


def test_contract_violation_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="x"):
        raise ContractViolationError("x")


def test_configure_logging_is_idempotent_and_utc() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream)
    logger = configure_logging("INFO", stream)
    own = [h for h in logger.handlers if getattr(h, "_frente_e_handler", False)]
    assert len(own) == 1
    logging.getLogger(f"{ROOT_LOGGER_NAME}.test").info("mensagem")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z INFO ", lines[0])
    assert lines[0].endswith("frente_e.test: mensagem")
