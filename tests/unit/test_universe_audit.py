"""Auditoria de elegibilidade: seleção de ações, painel, ranking sem look-ahead, U1/U2."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from frente_e.data import universe_audit as ua
from frente_e.data.cotahist import SPOT_MARKET
from frente_e.errors import DataValidationError


def _projection(rows: list[tuple[str, str, str, str, str]]) -> dict[str, np.ndarray]:
    """(tpmerc, codbdi, especi, isin, ticker) com data e volume fixos."""
    return {
        "date": np.full(len(rows), np.datetime64("2020-01-02")),
        "tpmerc": np.array([r[0].encode() for r in rows], dtype="S3"),
        "codbdi": np.array([r[1].encode() for r in rows], dtype="S2"),
        "especi": np.array([r[2].encode() for r in rows], dtype="S10"),
        "codisi": np.array([r[3].encode() for r in rows], dtype="S12"),
        "codneg": np.array([r[4].encode() for r in rows], dtype="S12"),
        "totneg": np.ones(len(rows), dtype=np.int64),
        "voltot": np.ones(len(rows)),
    }


def test_share_selection_rules() -> None:
    rows = [
        ("010", "02", "PN  N2", "BRPETRACNPR6", "PETR4"),  # ação
        ("010", "02", "ON *EJ", "BRVALEACNOR0", "VALE3"),  # '*' e sufixo
        ("010", "02", "UNT N2", "BRSANBCDAM13", "SANB11"),  # unit
        ("010", "08", "ON", "BROIBRACNOR1", "OIBR3"),  # recuperação judicial
        ("010", "02", "ON REC", "BRXXXXR01XX1", "XXXX9"),  # recibo de subscrição
        ("010", "02", "DRN", "BRAAPLBDR004", "AAPL34"),  # BDR
        ("010", "12", "CI", "BRHGLGCTF004", "HGLG11"),  # fundo
        ("020", "96", "PN", "BRPETRACNPR6", "PETR4F"),  # fracionário
    ]
    mask, reasons = ua.select_shares(_projection(rows), ua.REGULAR_LOT)
    assert mask.tolist() == [True, True, True, False, False, False, False, False]
    assert reasons["classe_fora:REC"] == 1
    assert reasons["bdi_fora"] == 2  # BDI 08 e 12 no mercado à vista
    assert reasons["classe_fora:DRN"] == 1
    assert reasons["fora_do_mercado_a_vista"] == 1
    wide, _ = ua.select_shares(_projection(rows), ua.REGULAR_LOT | ua.SPECIAL_REGIMES)
    assert wide[3]  # regime especial incluído na variante ampla
    assert SPOT_MARKET == b"010"


def test_share_token_normalization() -> None:
    assert ua.share_token(b"ON REC") == b"ON"  # excluído à parte pelo token REC
    assert ua.share_token(b"PN *EDJ") == b"PN"
    assert ua.share_token(b"") == b""


def _records(
    presence: np.ndarray, volume: np.ndarray, isins: list[str], tickers: list[str] | None = None
) -> tuple[ua.ShareRecords, np.ndarray]:
    sessions = np.datetime64("2000-01-03") + np.arange(presence.shape[0])
    t_idx, i_idx = np.nonzero(presence)
    tick = tickers or [i[2:6] + "3" for i in isins]
    recs = ua.ShareRecords(
        date=sessions[t_idx],
        codisi=np.array([isins[i].encode() for i in i_idx], dtype="S12"),
        codneg=np.array([tick[i].encode() for i in i_idx], dtype="S12"),
        codbdi=np.full(t_idx.size, b"02", dtype="S2"),
        traded=np.ones(t_idx.size, dtype=bool),
        volume=volume[t_idx, i_idx],
    )
    return recs, sessions


def test_presence_threshold_window_and_issuer_consolidation() -> None:
    n = 300
    isins = ["BRAAAAACNOR1", "BRAAAAACNPR1", "BRBBBBACNOR1", "BRCCCCACNOR1"]
    presence = np.ones((n, 4), dtype=bool)
    presence[n - 13 :, 3] = False  # CCCC: 239 de 252 na última janela (< 95%)
    volume = np.tile(np.array([1.0, 5.0, 3.0, 9.0]), (n, 1))
    records, sessions = _records(presence, volume, isins)
    months = ua.month_end_rankings(ua.build_panel(records, sessions))
    last = months[-1]
    assert last.ranked_issuers == ("AAAA", "BBBB")  # CCCC fora pela presença
    assert last.chosen[0] == "BRAAAAACNPR1"  # classe mais negociada do emissor
    assert last.n_eligible_instruments == 3


def test_presence_at_exactly_95_percent_is_eligible() -> None:
    n = 252
    presence = np.ones((n, 2), dtype=bool)
    presence[:12, 1] = False  # 240/252 = 95,24% ≥ 95%
    records, sessions = _records(presence, np.ones((n, 2)), ["BRAAAAACNOR1", "BRBBBBACNOR1"])
    months = ua.month_end_rankings(ua.build_panel(records, sessions))
    assert set(months[-1].ranked_issuers) == {"AAAA", "BBBB"}


def test_ranking_uses_only_sessions_up_to_t() -> None:
    n = 400
    isins = ["BRAAAAACNOR1", "BRBBBBACNOR1"]
    presence = np.ones((n, 2), dtype=bool)
    volume = np.tile(np.array([2.0, 1.0]), (n, 1))
    base, sessions = _records(presence, volume, isins)
    before = ua.month_end_rankings(ua.build_panel(base, sessions))
    future = volume.copy()
    future[-20:, 1] = 1e9  # mudança só no fim
    changed, _ = _records(presence, future, isins)
    after = ua.month_end_rankings(ua.build_panel(changed, sessions))
    cutoff = str(sessions[-21])
    for a, b in zip(before, after, strict=True):
        if a.date <= cutoff:
            assert a.ranked_issuers == b.ranked_issuers
    assert after[-1].ranked_issuers[0] == "BBBB"


def test_volume_ties_are_broken_by_issuer_code_and_universes_slice() -> None:
    n = 260
    isins = [f"BR{chr(65 + i) * 4}ACNOR1" for i in range(5)]
    presence = np.ones((n, 5), dtype=bool)
    records, sessions = _records(presence, np.ones((n, 5)), isins)
    month = ua.month_end_rankings(ua.build_panel(records, sessions))[-1]
    assert month.ranked_issuers == ("AAAA", "BBBB", "CCCC", "DDDD", "EEEE")
    assert ua.universe(month, "U1") == month.ranked_issuers  # menos de 30
    assert ua.universe(month, "U2") == ()


def test_sessions_must_cover_records() -> None:
    records, sessions = _records(np.ones((5, 1), dtype=bool), np.ones((5, 1)), ["BRAAAAACNOR1"])
    with pytest.raises(DataValidationError):
        ua.build_panel(records, sessions[1:])


def test_survivorship_and_identity_diagnostics() -> None:
    n = 300
    isins = ["BRAAAAACNOR1", "BRBBBBACNOR1"]
    presence = np.ones((n, 2), dtype=bool)
    presence[260:, 1] = False  # BBBB para no pregão 259: ainda elegível no fim de mês 271
    records, sessions = _records(presence, np.ones((n, 2)), isins, ["AAAA3", "BBBB3"])
    months = ua.month_end_rankings(ua.build_panel(records, sessions))
    report = ua.survivorship(records, months)
    assert [s["isin"] for s in report["stopped_before_end"]] == ["BRBBBBACNOR1"]
    identity = ua.identity_changes(records)
    assert identity["isin_with_multiple_tickers"] == 0


def test_summary_flags_months_below_threshold() -> None:
    months = [
        ua.MonthEnd("2001-01-31", 3, ("A", "B", "C"), ("x", "y", "z")),
        ua.MonthEnd("2001-02-28", 1, ("A",), ("x",)),
    ]
    summary = ua.summarize(months, threshold=2)
    assert summary["month_ends_below_60"] == ["2001-02-28"]
    assert summary["eligible_issuers_min"] == 1
    assert Counter(summary["min_eligible_by_year"]) == Counter({"2001": 1})
