"""Auditoria de elegibilidade histórica do COTAHIST (DD-01/DD-02): presença e volume.

**Escopo restrito por construção:** recebe só :func:`frente_e.data.cotahist.audit_projection`
(sem preços). Nada aqui calcula retorno, sinal, Sharpe ou PBO.

Regras da proposta DD-01 (METHODOLOGY_CONVENTIONS.md §6.6), com a leitura operacional
documentada em docs/COTAHIST_AUDIT.md:

- mercado à vista (TPMERC 010); BDI em ``bdi_codes`` (variantes comparadas: só 02, ou 02 +
  regimes especiais 05–09 e 11 da tabela CODBDI);
- ação = primeiro token de ESPECI, sem '*', em :data:`SHARE_TOKENS`, e sem o token ``REC``
  (recibos de subscrição); o resto é contado;
- instrumento = ISIN; emissor = caracteres 3–6 do ISIN;
- presença no pregão t = registro com TOTNEG > 0; pregões = datas com algum registro à vista;
- janela = os 252 pregões até t, inclusive (dados conhecidos no fechamento de t);
- elegível = presença ≥ 95% na janela; por emissor, a classe de maior volume médio;
- U1 = postos 1–30, U2 = postos 31–60 (desempate pelo código do emissor).

Execução: ``python -m frente_e.data.universe_audit --first-year 1999 --last-year 2025``.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.data import cotahist
from frente_e.errors import DataValidationError
from frente_e.settings import load_settings

SHARE_TOKENS = frozenset(
    {b"ON", b"PN", b"PNA", b"PNB", b"PNC", b"PND", b"PNE", b"PNF", b"PNG", b"PNH", b"UNT"}
)
REGULAR_LOT = frozenset({b"02"})
SPECIAL_REGIMES = frozenset({b"05", b"06", b"07", b"08", b"09", b"11"})
WINDOW = 252
MIN_PRESENCE = 0.95
UNIVERSES = {"U1": (1, 30), "U2": (31, 60)}


def share_token(especi: bytes) -> bytes:
    tokens = especi.replace(b"*", b" ").split()
    return tokens[0] if tokens else b""


@dataclass(frozen=True)
class ShareRecords:
    """Registros de ações selecionados (uma variante de BDI), só campos sem preço."""

    date: NDArray[np.datetime64]
    codisi: NDArray[np.bytes_]
    codneg: NDArray[np.bytes_]
    codbdi: NDArray[np.bytes_]
    traded: NDArray[np.bool_]  # TOTNEG > 0
    volume: NDArray[np.float64]


def select_shares(
    projection: dict[str, NDArray[np.generic]], bdi_codes: frozenset[bytes]
) -> tuple[NDArray[np.bool_], Counter[str]]:
    spot = projection["tpmerc"] == cotahist.SPOT_MARKET
    in_bdi = np.isin(projection["codbdi"], sorted(bdi_codes))
    especi = projection["especi"].tolist()
    tokens = np.array([share_token(e) for e in especi], dtype="S10")
    receipt = np.array([b"REC" in e.replace(b"*", b" ").split() for e in especi], dtype=bool)
    is_share = np.isin(tokens, sorted(SHARE_TOKENS)) & ~receipt
    isin_ok = np.char.startswith(projection["codisi"].astype("S12"), b"BR")
    reasons: Counter[str] = Counter()
    reasons["fora_do_mercado_a_vista"] += int(np.count_nonzero(~spot))
    reasons["bdi_fora"] += int(np.count_nonzero(spot & ~in_bdi))
    for token, is_receipt in zip(
        tokens[spot & in_bdi & ~is_share].tolist(),
        receipt[spot & in_bdi & ~is_share].tolist(),
        strict=True,
    ):
        label = "REC" if is_receipt else token.decode("latin-1")
        reasons[f"classe_fora:{label}"] += 1
    reasons["isin_invalido"] += int(np.count_nonzero(spot & in_bdi & is_share & ~isin_ok))
    return spot & in_bdi & is_share & isin_ok, reasons


def collect(
    paths: Iterable[Path], bdi_codes: frozenset[bytes]
) -> tuple[ShareRecords, Counter[str], list[dict[str, Any]]]:
    parts: list[dict[str, NDArray[np.generic]]] = []
    reasons: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    for path in paths:
        check = cotahist.FileCheck()
        n_selected = 0
        for block in cotahist.iter_quotes(path, check):
            spot = block[block["tpmerc"] == cotahist.SPOT_MARKET]
            reasons["fora_do_mercado_a_vista"] += block.size - spot.size
            if not spot.size:
                continue
            projection = cotahist.audit_projection(spot)
            mask, why = select_shares(projection, bdi_codes)
            reasons.update(why)
            n_selected += int(mask.sum())
            parts.append({k: projection[k][mask] for k in projection})
        if not check.complete:
            raise DataValidationError(f"{path.name}: leitura incompleta")
        files.append(
            {
                "file": path.name,
                "header": check.header,
                "records_declared": check.trailer_total,
                "trailer_convention": check.trailer_convention,
                "quotes": check.n_quotes,
                "share_records": n_selected,
            }
        )
    joined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    records = ShareRecords(
        date=joined["date"].astype("datetime64[D]"),
        codisi=joined["codisi"].astype("S12"),
        codneg=joined["codneg"].astype("S12"),
        codbdi=joined["codbdi"].astype("S2"),
        traded=joined["totneg"] > 0,
        volume=joined["voltot"].astype(np.float64),
    )
    return records, reasons, files


@dataclass(frozen=True)
class Panel:
    sessions: NDArray[np.datetime64]
    instruments: NDArray[np.bytes_]  # ISIN
    issuers: NDArray[np.bytes_]
    presence: NDArray[np.bool_]  # (pregões, instrumentos)
    volume: NDArray[np.float64]


def build_panel(records: ShareRecords, sessions: NDArray[np.datetime64]) -> Panel:
    """Painel pregão × ISIN. ``sessions`` = calendário do mercado à vista (todas as datas)."""
    s_idx = np.searchsorted(sessions, records.date)
    if np.any(sessions[np.minimum(s_idx, sessions.size - 1)] != records.date):
        raise DataValidationError("registro com data fora do calendário de pregões")
    instruments, i_idx = np.unique(records.codisi, return_inverse=True)
    presence = np.zeros((sessions.size, instruments.size), dtype=bool)
    volume = np.zeros((sessions.size, instruments.size), dtype=np.float64)
    traded = records.traded
    presence[s_idx[traded], i_idx[traded]] = True
    np.add.at(volume, (s_idx[traded], i_idx[traded]), records.volume[traded])
    issuers = np.array([isin[2:6] for isin in instruments.tolist()], dtype="S4")
    return Panel(sessions, instruments, issuers, presence, volume)


@dataclass(frozen=True)
class MonthEnd:
    date: str
    n_eligible_instruments: int
    ranked_issuers: tuple[str, ...]
    chosen: tuple[str, ...]  # ISIN escolhido para cada emissor ranqueado


def month_end_rankings(
    panel: Panel, *, window: int = WINDOW, min_presence: float = MIN_PRESENCE
) -> list[MonthEnd]:
    """Ranking no último pregão de cada mês com janela completa; usa só pregões ≤ t."""
    months = panel.sessions.astype("datetime64[M]")
    ends = np.flatnonzero(np.r_[months[1:] != months[:-1], True])
    out: list[MonthEnd] = []
    for t in ends.tolist():
        if t + 1 < window:
            continue
        rows = slice(t + 1 - window, t + 1)  # os `window` pregões até t, inclusive
        presence = np.count_nonzero(panel.presence[rows], axis=0) / window
        mean_volume = panel.volume[rows].sum(axis=0) / window
        eligible = np.flatnonzero(presence >= min_presence)
        best: dict[bytes, tuple[float, bytes]] = {}
        for i in eligible.tolist():
            issuer = bytes(panel.issuers[i])
            vol = float(mean_volume[i])
            if issuer not in best or vol > best[issuer][0]:
                best[issuer] = (vol, bytes(panel.instruments[i]))
        ranked = sorted(best, key=lambda k: (-best[k][0], k))
        out.append(
            MonthEnd(
                date=str(panel.sessions[t]),
                n_eligible_instruments=int(eligible.size),
                ranked_issuers=tuple(k.decode() for k in ranked),
                chosen=tuple(best[k][1].decode() for k in ranked),
            )
        )
    return out


def universe(month: MonthEnd, name: str) -> tuple[str, ...]:
    first, last = UNIVERSES[name]
    return month.ranked_issuers[first - 1 : last]


def turnover(months: Sequence[MonthEnd], name: str) -> float:
    """Fração média de membros trocados de um mês para o seguinte."""
    changes = []
    for prev, cur in itertools.pairwise(months):
        a, b = set(universe(prev, name)), set(universe(cur, name))
        if a and b:
            changes.append(1.0 - len(a & b) / max(len(a), len(b)))
    return float(np.mean(changes)) if changes else float("nan")


def survivorship(records: ShareRecords, months: Sequence[MonthEnd]) -> dict[str, Any]:
    """Instrumentos que estiveram em U1/U2 e deixaram de negociar antes do fim da base."""
    last_session = records.date.max()
    members: dict[str, str] = {}
    for month in months:
        for name in UNIVERSES:
            for issuer in universe(month, name):
                isin = month.chosen[month.ranked_issuers.index(issuer)]
                members[isin] = month.date
    stopped = []
    for isin, last_member in sorted(members.items()):
        mask = (records.codisi == isin.encode()) & records.traded
        last_trade = records.date[mask].max()
        if (last_session - last_trade).astype(int) > 31:
            tickers = sorted({t.decode() for t in records.codneg[mask].tolist()})
            stopped.append(
                {
                    "isin": isin,
                    "tickers": tickers,
                    "last_trade": str(last_trade),
                    "last_month_in_universe": last_member,
                }
            )
    return {"n_members_ever": len(members), "stopped_before_end": stopped}


def identity_changes(records: ShareRecords) -> dict[str, Any]:
    pairs = {(i, t) for i, t in zip(records.codisi.tolist(), records.codneg.tolist(), strict=True)}
    by_isin: dict[bytes, set[bytes]] = {}
    by_ticker: dict[bytes, set[bytes]] = {}
    for isin, ticker in pairs:
        by_isin.setdefault(isin, set()).add(ticker)
        by_ticker.setdefault(ticker, set()).add(isin)
    return {
        "isin_with_multiple_tickers": sum(len(v) > 1 for v in by_isin.values()),
        "ticker_with_multiple_isins": sum(len(v) > 1 for v in by_ticker.values()),
        "n_isins": len(by_isin),
        "n_tickers": len(by_ticker),
    }


def special_regime_overlap(records: ShareRecords) -> dict[str, int]:
    """ISINs com registros tanto em lote padrão (02) quanto em regime especial."""
    regular = set(records.codisi[np.isin(records.codbdi, sorted(REGULAR_LOT))].tolist())
    special = set(records.codisi[np.isin(records.codbdi, sorted(SPECIAL_REGIMES))].tolist())
    return {"isins_special_regime": len(special), "isins_in_both": len(regular & special)}


def summarize(months: Sequence[MonthEnd], *, threshold: int = 60) -> dict[str, Any]:
    counts = np.array([len(m.ranked_issuers) for m in months])
    by_year: dict[str, int] = {}
    for m in months:
        year = m.date[:4]
        by_year[year] = min(by_year.get(year, 10**9), len(m.ranked_issuers))
    return {
        "first_month_end": months[0].date,
        "last_month_end": months[-1].date,
        "n_month_ends": len(months),
        "eligible_issuers_min": int(counts.min()),
        "eligible_issuers_min_date": months[int(counts.argmin())].date,
        "eligible_issuers_median": float(np.median(counts)),
        "month_ends_below_60": [m.date for m in months if len(m.ranked_issuers) < threshold],
        "month_ends_below_30": [m.date for m in months if len(m.ranked_issuers) < 30],
        "min_eligible_by_year": by_year,
        "turnover_U1": turnover(months, "U1"),
        "turnover_U2": turnover(months, "U2"),
    }


def run_audit(paths: Sequence[Path]) -> dict[str, Any]:
    report: dict[str, Any] = {"variants": {}}
    sessions: NDArray[np.datetime64] | None = None
    for variant, codes in (
        ("bdi_02", REGULAR_LOT),
        ("bdi_02_e_regimes_especiais", REGULAR_LOT | SPECIAL_REGIMES),
    ):
        records, reasons, files = collect(paths, codes)
        if sessions is None:
            sessions = spot_sessions(paths)
            report["files"] = files
            report["sessions_by_year"] = dict(
                sorted(Counter(str(d)[:4] for d in sessions.tolist()).items())
            )
        months = month_end_rankings(build_panel(records, sessions))
        report["variants"][variant] = {
            "exclusions": dict(sorted(reasons.items())),
            "summary": summarize(months),
            "survivorship": survivorship(records, months),
            "identity": identity_changes(records),
            "special_regime": special_regime_overlap(records),
            "monthly": [
                {
                    "date": m.date,
                    "eligible_instruments": m.n_eligible_instruments,
                    "eligible_issuers": len(m.ranked_issuers),
                    "U1": list(universe(m, "U1")),
                    "U2": list(universe(m, "U2")),
                }
                for m in months
            ],
        }
    return report


def spot_sessions(paths: Sequence[Path]) -> NDArray[np.datetime64]:
    """Calendário: datas com algum registro do mercado à vista (qualquer papel)."""
    dates: set[np.datetime64] = set()
    for path in paths:
        check = cotahist.FileCheck()
        for block in cotahist.iter_quotes(path, check):
            spot = block[block["tpmerc"] == cotahist.SPOT_MARKET]
            if spot.size:
                dates.update(np.unique(cotahist.audit_projection(spot)["date"]).tolist())
    return np.array(sorted(dates), dtype="datetime64[D]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-year", type=int, required=True)
    parser.add_argument("--last-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = load_settings()
    raw_dir = settings.paths.raw_data / "cotahist"
    manifest = settings.paths.manifests / "cotahist.json"
    paths = []
    for year in range(args.first_year, args.last_year + 1):
        paths.append(cotahist.download_year(year, raw_dir, manifest))
        print(f"ok {year}", file=sys.stderr, flush=True)
    report = run_audit(paths)
    report["manifest_sha256"] = cotahist.sha256_file(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({v: r["summary"] for v, r in report["variants"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
