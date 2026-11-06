from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from hypothesis import settings

# "dev": exploração aleatória padrão. "repro": determinística, para reproduzir uma falha
# ou rodar em CI. Selecione com HYPOTHESIS_PROFILE=repro.
settings.register_profile("dev", deadline=None)
settings.register_profile("repro", deadline=None, derandomize=True, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Raiz de projeto isolada com a configuração real copiada."""
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO_ROOT / "configs" / "project.toml", tmp_path / "configs" / "project.toml")
    return tmp_path
