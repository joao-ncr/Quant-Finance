"""Serialização canônica e hashes (DA-03).

A forma canônica é JSON UTF-8 com chaves ordenadas por ponto de código, sem espaços,
sem escapes ASCII e sem NaN/Infinity. Floats usam ``repr`` do Python (a menor
representação que reconstrói o mesmo IEEE-754), que é determinística entre plataformas.

Regras de tipo, para que a mesma configuração sempre gere os mesmos bytes:
- só ``None``, ``bool``, ``int``, ``float`` finito, ``str``, mapeamentos com chave ``str``
  e listas/tuplas (tuplas viram listas);
- escalares NumPy inteiros/booleanos são rejeitados (converta explicitamente);
- ``1`` e ``1.0`` são valores diferentes e geram bytes diferentes (o TOML preserva o tipo).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from frente_e.errors import ContractViolationError

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None

_CHUNK = 1 << 20


def to_canonical(value: object, path: str = "$") -> JsonValue:
    """Copia ``value`` para tipos JSON puros, validando-o. Não altera a entrada."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolationError(f"{path}: float não finito não é serializável")
        return float(value)
    if isinstance(value, Mapping):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractViolationError(f"{path}: chave {key!r} não é str")
            out[key] = to_canonical(item, f"{path}.{key}")
        return out
    if isinstance(value, list | tuple):
        return [to_canonical(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise ContractViolationError(
        f"{path}: tipo {type(value).__name__} não tem forma canônica; converta explicitamente"
    )


def canonical_json(value: object) -> str:
    return json.dumps(
        to_canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
