"""Armazém local de artefatos endereçados por conteúdo (DA-03, DA-04).

Resultados volumosos (matrizes, tabelas) ficam fora do JSONL. O ledger guarda
apenas :class:`ArtifactRef` (caminho relativo, SHA-256, tamanho, tipo, descrição).
Caminho: ``<raiz>/<sha[:2]>/<sha><sufixo>``. Escrita atômica (arquivo temporário +
``os.replace``). Conteúdo idêntico é armazenado uma vez.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frente_e.errors import ContractViolationError, DataValidationError
from frente_e.experiments.canonical import JsonValue, sha256_file, sha256_hex
from frente_e.experiments.identity import HEX64_PATTERN

_SUFFIX_PATTERN = re.compile(r"^(\.[a-z0-9]{1,10})+$")


@dataclass(frozen=True)
class ArtifactRef:
    relative_path: str  # POSIX, relativo à raiz do armazém
    sha256: str
    size_bytes: int
    media_type: str
    description: str

    def __post_init__(self) -> None:
        if not HEX64_PATTERN.match(self.sha256):
            raise ContractViolationError(f"sha256 inválido: {self.sha256!r}")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ContractViolationError(f"relative_path inseguro: {self.relative_path!r}")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "description": self.description,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ArtifactRef:
        return cls(**data)


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put_bytes(
        self, data: bytes, *, suffix: str, media_type: str, description: str
    ) -> ArtifactRef:
        if not _SUFFIX_PATTERN.match(suffix):
            raise ContractViolationError(f"sufixo inválido: {suffix!r}")
        digest = sha256_hex(data)
        relative = f"{digest[:2]}/{digest}{suffix}"
        target = self.root / relative
        if target.exists():
            if sha256_file(target) != digest:
                raise DataValidationError(f"Artefato existente corrompido: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, target)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise
        return ArtifactRef(
            relative_path=relative,
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
            description=description,
        )

    def put_array(self, array: NDArray[Any], *, description: str) -> ArtifactRef:
        """Grava um array no formato .npy (sem pickle)."""
        if array.dtype.hasobject:
            raise ContractViolationError("arrays de objetos exigiriam pickle; não suportado")
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return self.put_bytes(
            buffer.getvalue(),
            suffix=".npy",
            media_type="application/x-npy",
            description=description,
        )

    def path_of(self, ref: ArtifactRef) -> Path:
        return self.root / PurePosixPath(ref.relative_path)

    def verify(self, ref: ArtifactRef) -> None:
        path = self.path_of(ref)
        if not path.is_file():
            raise DataValidationError(f"Artefato ausente: {path}")
        if path.stat().st_size != ref.size_bytes or sha256_file(path) != ref.sha256:
            raise DataValidationError(f"Artefato não confere com a referência: {path}")

    def load_array(self, ref: ArtifactRef) -> NDArray[Any]:
        self.verify(ref)
        loaded: NDArray[Any] = np.load(self.path_of(ref), allow_pickle=False)
        return loaded
