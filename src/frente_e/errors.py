"""Hierarquia de erros do projeto.

Regra: problemas metodológicos ou de dados geram erro explícito, nunca um valor
padrão silencioso (CLAUDE.md §0).
"""


class FrenteEError(Exception):
    """Base de todos os erros do projeto."""


class ConfigError(FrenteEError):
    """Configuração ausente, malformada, com chaves desconhecidas ou valores inválidos."""


class ContractViolationError(FrenteEError, ValueError):
    """Entrada que viola o contrato de uma interface entre camadas."""


class DataValidationError(FrenteEError):
    """Dados que falham em checagens de qualidade ou integridade."""


class ResourceBudgetError(FrenteEError):
    """A execução não cabe no orçamento computacional configurado (DA-07)."""


class ReproducibilityError(FrenteEError):
    """Falta informação necessária para tornar uma execução reproduzível."""


class DegenerateSeriesError(FrenteEError, ValueError):
    """Série para a qual a estatística pedida é indefinida (ex.: variância zero)."""


class LedgerIntegrityError(FrenteEError):
    """O ledger de tentativas está corrompido, adulterado ou inconsistente."""

    def __init__(self, message: str, line: int | None = None) -> None:
        self.line = line
        super().__init__(message if line is None else f"linha {line}: {message}")


class TornTailError(LedgerIntegrityError):
    """A última linha do ledger está incompleta (escrita interrompida)."""


class LedgerLockedError(FrenteEError):
    """Outro escritor já detém o ledger (o desenho exige escritor único)."""


class PendingDecisionError(FrenteEError, NotImplementedError):
    """Caminho de código que depende de uma decisão metodológica ainda não tomada.

    Use em vez de inventar uma convenção. A mensagem aponta o identificador da
    decisão em PROJECT_DECISIONS.md.
    """

    def __init__(self, decision_id: str, detail: str) -> None:
        self.decision_id = decision_id
        self.detail = detail
        super().__init__(
            f"Depende da decisão {decision_id}, ainda não aprovada "
            f"(ver PROJECT_DECISIONS.md): {detail}"
        )
