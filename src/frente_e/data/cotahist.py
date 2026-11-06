"""COTAHIST da B3: obtenção imutável (DA-04) e leitura validada (DA-09).

Layout oficial: "SeriesHistoricas_Layout.pdf", versão 2.0 de 05/10/2020, conferido
visualmente em 2026-11-06 (SHA-256 em docs/COTAHIST_AUDIT.md).

- Registros de 245 bytes terminados em CRLF: tipo 00 (header), 01 (cotação) e 99
  (trailer, com o total de registros incluindo header e trailer pelo layout; o arquivo de 2025
  declara só as cotações, e as duas convenções são aceitas e registradas).
- :func:`iter_quotes` lê o membro do zip em blocos (memória limitada) e valida comprimento,
  terminador, tipos, posição do trailer e contagem declarada.
- :data:`AUDIT_FIELDS` e :func:`audit_projection` expõem **só** identificação, presença e
  volume. A auditoria de universo (DD-01/DD-02) não recebe preços, por construção.
- :func:`download_year` grava o bruto uma única vez e registra URL, SHA-256, tamanho e data
  no manifesto versionado. Um arquivo existente nunca é sobrescrito; hash divergente é erro.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.errors import DataValidationError

RECORD_LENGTH = 245
LINE_LENGTH = RECORD_LENGTH + 2  # CRLF
URL_TEMPLATE = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
LAYOUT_VERSION = "SeriesHistoricas_Layout.pdf v2.0 (05/10/2020)"

# Posições 1-based inclusivas do registro 01 (layout, seção 3.2).
FIELDS: dict[str, tuple[int, int]] = {
    "tipreg": (1, 2),
    "data": (3, 10),
    "codbdi": (11, 12),
    "codneg": (13, 24),
    "tpmerc": (25, 27),
    "nomres": (28, 39),
    "especi": (40, 49),
    "prazot": (50, 52),
    "modref": (53, 56),
    "preabe": (57, 69),
    "premax": (70, 82),
    "premin": (83, 95),
    "premed": (96, 108),
    "preult": (109, 121),
    "preofc": (122, 134),
    "preofv": (135, 147),
    "totneg": (148, 152),
    "quatot": (153, 170),
    "voltot": (171, 188),
    "preexe": (189, 201),
    "indopc": (202, 202),
    "datven": (203, 210),
    "fatcot": (211, 217),
    "ptoexe": (218, 230),
    "codisi": (231, 242),
    "dismes": (243, 245),
}

AUDIT_FIELDS = ("date", "codbdi", "codneg", "tpmerc", "especi", "totneg", "voltot", "codisi")
SPOT_MARKET = b"010"  # TPMERC "VISTA" (tabela anexa)

RECORD_DTYPE = np.dtype(
    {
        "names": [*FIELDS, "eol"],
        "formats": [f"S{end - start + 1}" for start, end in FIELDS.values()] + ["S2"],
        "offsets": [start - 1 for start, _ in FIELDS.values()] + [RECORD_LENGTH],
        "itemsize": LINE_LENGTH,
    }
)


@dataclass
class FileCheck:
    """Preenchido durante a leitura; ``complete`` só fica True se o arquivo inteiro validou."""

    member: str = ""
    header: str = ""
    trailer_total: int = -1
    n_lines: int = 0
    n_quotes: int = 0
    trailer_convention: str = ""  # "layout" (todas as linhas) ou "so_cotacoes" (observado em 2025)
    complete: bool = False


def iter_quotes(
    path: Path, check: FileCheck, *, chunk_records: int = 200_000
) -> Iterator[NDArray[np.void]]:
    """Blocos de registros tipo 01 (array estruturado). Qualquer desvio do layout é erro."""
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise DataValidationError(f"{path.name}: esperado 1 membro no zip, há {members}")
        check.member = members[0]
        with archive.open(members[0]) as fh:
            first = fh.read(LINE_LENGTH)
            if len(first) != LINE_LENGTH or first[:2] != b"00" or first[-2:] != b"\r\n":
                raise DataValidationError(f"{path.name}: header ausente ou malformado")
            check.header = first[:31].decode("latin-1")
            check.n_lines = 1
            trailer_seen = False
            while raw := fh.read(chunk_records * LINE_LENGTH):
                if trailer_seen:
                    raise DataValidationError(f"{path.name}: registros após o trailer")
                if len(raw) % LINE_LENGTH:
                    raise DataValidationError(
                        f"{path.name}: comprimento não múltiplo de {LINE_LENGTH} bytes"
                    )
                block = np.frombuffer(raw, dtype=RECORD_DTYPE)
                if np.any(block["eol"] != b"\r\n"):
                    raise DataValidationError(f"{path.name}: terminador de linha inesperado")
                kinds = block["tipreg"]
                if np.any((kinds != b"01") & (kinds != b"99")):
                    raise DataValidationError(f"{path.name}: tipo de registro inesperado")
                trailers = np.flatnonzero(kinds == b"99")
                if trailers.size > 1 or (trailers.size and trailers[0] != block.size - 1):
                    raise DataValidationError(f"{path.name}: trailer fora da última posição")
                if trailers.size:
                    trailer_seen = True
                    check.trailer_total = int(raw[-LINE_LENGTH:][31:42])
                    block = block[:-1]
                check.n_lines += block.size + int(trailers.size)
                check.n_quotes += block.size
                yield block
            if not trailer_seen:
                raise DataValidationError(f"{path.name}: trailer ausente")
            # O layout manda incluir header e trailer na contagem. O arquivo de 2025
            # (baixado em 2026-11-06) declara só as cotações. Aceitam-se exatamente essas duas
            # contagens, com a convenção registrada; qualquer outro valor é erro.
            if check.trailer_total == check.n_lines:
                check.trailer_convention = "layout"
            elif check.trailer_total == check.n_quotes:
                check.trailer_convention = "so_cotacoes"
            else:
                raise DataValidationError(
                    f"{path.name}: trailer declara {check.trailer_total} registros, "
                    f"lidos {check.n_lines} linhas ({check.n_quotes} cotações)"
                )
            check.complete = True


def audit_projection(block: NDArray[np.void]) -> dict[str, NDArray[np.generic]]:
    """Campos sem preço, tipados: data (datetime64[D]), códigos em bytes, TOTNEG (int) e
    VOLTOT em reais ((16)V99)."""
    ymd = block["data"].astype(np.int64)
    years = (ymd // 10000 - 1970).astype("timedelta64[Y]")
    months = (ymd // 100 % 100 - 1).astype("timedelta64[M]")
    days = (ymd % 100 - 1).astype("timedelta64[D]")
    month_start = (np.datetime64("1970", "Y") + years).astype("datetime64[M]") + months
    date = month_start.astype("datetime64[D]") + days
    if np.any(date.astype("datetime64[M]") != month_start):
        raise DataValidationError("data inválida no campo DATA DO PREGÃO")
    return {
        "date": date,
        "codbdi": block["codbdi"].copy(),
        "codneg": np.char.strip(block["codneg"]),
        "tpmerc": block["tpmerc"].copy(),
        "especi": np.char.strip(block["especi"]),
        "totneg": block["totneg"].astype(np.int64),
        "voltot": block["voltot"].astype(np.int64).astype(np.float64) / 100.0,
        "codisi": block["codisi"].copy(),
    }


# ----------------------------------------------------------------------------- obtenção
def sha256_file(path: Path) -> str:
    """SHA-256 em blocos (a camada de dados não depende de ``experiments``)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"source": "B3 COTAHIST", "layout": LAYOUT_VERSION, "files": {}}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def download_year(year: int, raw_dir: Path, manifest_path: Path) -> Path:
    """Baixa o arquivo anual uma única vez; verifica o hash se ele já existir."""
    name = f"COTAHIST_A{year}.ZIP"
    target = raw_dir / name
    manifest = load_manifest(manifest_path)
    entry = manifest["files"].get(name)
    if target.exists():
        if entry is None:
            raise DataValidationError(f"{name} existe sem registro no manifesto")
        if sha256_file(target) != entry["sha256"]:
            raise DataValidationError(f"{name}: SHA-256 diverge do manifesto (bruto alterado)")
        return target
    if entry is not None:
        raise DataValidationError(f"{name} registrado no manifesto mas ausente do disco")
    url = URL_TEMPLATE.format(year=year)
    raw_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "frente-e-research"})
    fd, tmp = tempfile.mkstemp(dir=raw_dir, prefix=".tmp-")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, os.fdopen(fd, "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    manifest["files"][name] = {
        "url": url,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "downloaded_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target
