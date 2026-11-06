"""Frente E: validação estatística de regras simples na B3.

Camadas (docs/ARCHITECTURE.md):
    data        ingestão, normalização, universo point-in-time
    core        lógica matemática pura (sem I/O, sem RNG global)
    experiments configuração, proveniência, ledger de tentativas
    reporting   tabelas e figuras a partir do ledger
"""

from importlib.metadata import version

__version__ = version("frente-e")
