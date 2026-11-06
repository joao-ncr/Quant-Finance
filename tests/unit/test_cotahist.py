"""Leitor do COTAHIST: layout, validações, projeção sem preço e bruto imutável.

Os arquivos são sintéticos, montados aqui nas posições do layout oficial (v2.0).
"""

from __future__ import annotations

import io
import itertools
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frente_e.data import cotahist
from frente_e.errors import DataValidationError


def record(**values: str) -> bytes:
    """Registro 01 de 245 bytes; campos ausentes ficam em branco ou zero."""
    line = bytearray(b" " * cotahist.RECORD_LENGTH)
    defaults = {"tipreg": "01", "data": "20200102", "codbdi": "02", "tpmerc": "010"}
    for name, value in {**defaults, **values}.items():
        start, end = cotahist.FIELDS[name]
        width = end - start + 1
        text = value.rjust(width, "0") if value.isdigit() else value.ljust(width)
        assert len(text) == width, name
        line[start - 1 : end] = text.encode("latin-1")
    return bytes(line)


def header() -> bytes:
    return b"00COTAHIST.2020BOVESPA 20210104".ljust(cotahist.RECORD_LENGTH)


def trailer(total: int) -> bytes:
    return (b"99COTAHIST.2020BOVESPA 20210104" + str(total).zfill(11).encode()).ljust(
        cotahist.RECORD_LENGTH
    )


def write_zip(path: Path, lines: list[bytes], *, eol: bytes = b"\r\n") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("COTAHIST.A2020", b"".join(line + eol for line in lines))
    return path


def quote(**values: str) -> bytes:
    base = {
        "codneg": "PETR4",
        "especi": "PN  N2",
        "totneg": "120",
        "voltot": "123456789",
        "codisi": "BRPETRACNPR6",
        "preult": "3050",
    }
    return record(**{**base, **values})


def read_all(path: Path) -> tuple[list[np.void], cotahist.FileCheck]:
    check = cotahist.FileCheck()
    rows = [r for block in cotahist.iter_quotes(path, check, chunk_records=2) for r in block]
    return rows, check


def test_layout_positions_cover_245_bytes_without_gaps() -> None:
    spans = sorted(cotahist.FIELDS.values())
    assert spans[0][0] == 1
    assert spans[-1][1] == cotahist.RECORD_LENGTH
    for (_, end), (start, _) in itertools.pairwise(spans):
        assert start == end + 1
    assert cotahist.FIELDS["codisi"] == (231, 242)  # conferido no layout oficial


def test_valid_file_is_read_and_counted(tmp_path: Path) -> None:
    lines = [header(), quote(), quote(codneg="VALE3", codisi="BRVALEACNOR0"), quote(), trailer(5)]
    rows, check = read_all(write_zip(tmp_path / "a.zip", lines))
    assert check.complete
    assert (check.n_lines, check.n_quotes, check.trailer_total) == (5, 3, 5)
    assert [bytes(r["codneg"]).strip() for r in rows] == [b"PETR4", b"VALE3", b"PETR4"]
    assert check.trailer_convention == "layout"


def test_trailer_counting_only_quotes_is_accepted_and_recorded(tmp_path: Path) -> None:
    # convenção observada no arquivo de 2025: total = só registros de cotação
    _, check = read_all(write_zip(tmp_path / "q.zip", [header(), quote(), quote(), trailer(2)]))
    assert check.complete
    assert check.trailer_convention == "so_cotacoes"


def test_projection_has_no_price_fields_and_converts_units(tmp_path: Path) -> None:
    lines = [header(), quote(data="20200131", voltot="123456789"), trailer(3)]
    check = cotahist.FileCheck()
    (block,) = list(cotahist.iter_quotes(write_zip(tmp_path / "a.zip", lines), check))
    projection = cotahist.audit_projection(block)
    assert set(projection) == set(cotahist.AUDIT_FIELDS)
    assert not {"preabe", "premax", "premin", "premed", "preult"} & set(projection)
    assert str(projection["date"][0]) == "2020-01-31"
    assert projection["voltot"][0] == pytest.approx(1234567.89)
    assert projection["totneg"][0] == 120
    assert projection["especi"][0] == b"PN  N2"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([quote(), trailer(2)], "header"),
        ([header(), quote()], "trailer ausente"),
        ([header(), quote(), trailer(7)], "declara 7"),
        ([header(), trailer(3), quote()], "trailer"),
        ([header(), quote(tipreg="02"), trailer(3)], "tipo de registro"),
    ],
    ids=["sem header", "sem trailer", "contagem", "apos trailer", "tipo"],
)
def test_layout_violations_are_errors(tmp_path: Path, lines: list[bytes], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        read_all(write_zip(tmp_path / "bad.zip", lines))


def test_wrong_line_terminator_or_length(tmp_path: Path) -> None:
    lines = [header(), quote(), trailer(3)]
    with pytest.raises(DataValidationError):
        read_all(write_zip(tmp_path / "lf.zip", lines, eol=b"\n"))
    with pytest.raises(DataValidationError):
        read_all(write_zip(tmp_path / "short.zip", [header(), quote()[:-1] + b"", trailer(3)]))


def test_invalid_date_is_rejected(tmp_path: Path) -> None:
    check = cotahist.FileCheck()
    path = write_zip(tmp_path / "d.zip", [header(), quote(data="20200231"), trailer(3)])
    (block,) = list(cotahist.iter_quotes(path, check))
    with pytest.raises(DataValidationError, match="data inválida"):
        cotahist.audit_projection(block)


def test_zip_with_more_than_one_member(tmp_path: Path) -> None:
    path = tmp_path / "two.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a", b"")
        zf.writestr("b", b"")
    with pytest.raises(DataValidationError, match="1 membro"):
        read_all(path)


# ------------------------------------------------------------------------ bruto imutável
class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_download_records_manifest_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"conteudo-do-zip"
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        calls.append(request.full_url)
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    raw, manifest = tmp_path / "raw", tmp_path / "manifests" / "cotahist.json"
    path = cotahist.download_year(2020, raw, manifest)
    assert path.read_bytes() == payload
    entry = json.loads(manifest.read_text("utf-8"))["files"]["COTAHIST_A2020.ZIP"]
    assert entry["sha256"] == cotahist.sha256_file(path)
    assert entry["url"].endswith("COTAHIST_A2020.ZIP")
    # segunda chamada: só verifica o hash, não baixa de novo
    assert cotahist.download_year(2020, raw, manifest) == path
    assert len(calls) == 1
    # bruto alterado: erro, nunca sobrescreve
    path.write_bytes(b"alterado")
    with pytest.raises(DataValidationError, match="diverge"):
        cotahist.download_year(2020, raw, manifest)


def test_existing_file_without_manifest_entry_is_an_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "COTAHIST_A2019.ZIP").write_bytes(b"x")
    with pytest.raises(DataValidationError, match="sem registro"):
        cotahist.download_year(2019, raw, tmp_path / "m.json")
