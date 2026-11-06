"""Estado de cada observação (data, ativo): proposta de política de ausências.

Especificação: docs/METHODOLOGY_CONVENTIONS.md §2 (aguardando confirmação). A camada de
dados publica, junto com preços e retornos, uma tabela ``(date, asset_id, status)``.
Retornos só existem para ``TRADED``; nas demais situações o valor fica ausente, e o
motivo é explícito. Nenhuma ausência é preenchida com zero ou com o último valor; o
tratamento é decidido na construção das carteiras (DE-01), não no núcleo, que recebe
apenas séries finitas.
"""

from enum import StrEnum


class ObservationStatus(StrEnum):
    TRADED = "traded"  # houve negócio e há preço anterior válido: retorno calculável
    NO_TRADE = "no_trade"  # ativo listado, pregão aberto, nenhum negócio no dia
    NOT_YET_LISTED = "not_yet_listed"  # data anterior à primeira negociação
    DELISTED = "delisted"  # data posterior à saída definitiva (deslistagem, incorporação)
    SUSPENDED = "suspended"  # negociação suspensa pela bolsa (se identificável na fonte)
    SOURCE_GAP = "source_gap"  # o dado deveria existir e falta (falha de coleta/fonte)
    NO_PRIOR_PRICE = "no_prior_price"  # há preço hoje, mas não há preço anterior válido
