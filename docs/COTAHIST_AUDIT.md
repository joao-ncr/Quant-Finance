# Auditoria de elegibilidade histórica do COTAHIST (DD-01/DD-02) — 2026-11-06

**Escopo:** só presença e volume financeiro. Nenhum preço, retorno, sinal, Sharpe ou PBO foi calculado. O código da auditoria (`frente_e.data.universe_audit`) recebe apenas `audit_projection`, sem campos de preço, por construção.

**Recomendação, não aprovação:** os universos U1/U2 continuam como proposta (DD-01 `IN PROGRESS`).

## 1. Fontes e proveniência
| Item | Valor |
|---|---|
| Layout | B3, "SeriesHistoricas_Layout.pdf", versão 2.0 de 05/10/2020, conferido visualmente; SHA-256 `8b9633c44e1dc2229fb529931b4deb1d29fda9305a027d9c2c90a7ca96e8268f` |
| Arquivos | `COTAHIST_A1999.ZIP` … `COTAHIST_A2025.ZIP` (27 arquivos, 596 MB), em `https://bvmf.bmfbovespa.com.br/InstDados/SerHist/` |
| Bruto | `data/raw/cotahist/` (imutável, fora do Git) |
| Manifesto | `data/manifests/cotahist.json` (versionado): URL, SHA-256, tamanho e data de download por arquivo. SHA-256 do manifesto: `3b552b7a4fdc4fdfa66ccee77c3e44a18176e77be83db89f73e7831d9e34d6bc` |
| Relatório completo | `reports/cotahist_audit.json` (versionado): contagens mensais, membros de U1/U2 por data, exclusões, sobrevivência e identidade |

## 2. Validação do formato (DA-09)
- Registros de 245 bytes + CRLF, um único membro por zip, header (00) na primeira linha, trailer (99) na última, tipos de registro e datas válidas. Qualquer desvio é erro.
- **1999–2024:** o total do trailer é igual ao número de linhas (header e trailer incluídos), como manda o layout.
- **2025: o trailer declara só os registros de cotação** (3.174.698 = 3.174.700 linhas − header − trailer), em desacordo com o layout.
  - A primeira execução **falhou** por isso.
  - O leitor passou a aceitar **exatamente** essas duas contagens e registra a convenção de cada arquivo (`trailer_convention`); qualquer outro valor continua sendo erro.
  - A coincidência exata com o número de cotações é a evidência de que se trata de mudança de convenção, e não de truncamento.
- **Pregões por ano:** 245 a 251. O calendário é o das datas com algum registro do mercado à vista.

## 3. Regras aplicadas (leitura operacional da proposta DD-01)
| Regra | Implementação | Status |
|---|---|---|
| Mercado | TPMERC 010 (à vista); fracionário (020), termo e opções contados como excluídos | conforme proposta |
| BDI | duas variantes: **só 02** (lote padrão) e **02 + regimes especiais** (05 sancionadas, 06 concordatárias, 07 rec. extrajudicial, 08 rec. judicial, 09 RAET, 11 intervenção) | **escolha pendente** (§5) |
| Classe | primeiro token de ESPECI, sem `*`, em {ON, PN, PNA–PNH, UNT}, e sem o token `REC` | operacional |
| Instrumento / emissor | ISIN / caracteres 3–6 do ISIN | operacional |
| Presença | registro com TOTNEG > 0 no pregão | operacional |
| Janela | os 252 pregões até t, **inclusive** (dados conhecidos no fechamento de t). A proposta dizia "anteriores a t"; adotou-se a leitura sem look-ahead que inclui t | leitura explicitada |
| Elegível | presença ≥ 95% (≥ 240 de 252); por emissor, a classe de maior volume médio na janela | conforme proposta |
| Ranking | volume médio decrescente; desempate pelo código do emissor; U1 = 1–30, U2 = 31–60; revisão mensal (último pregão do mês) | conforme proposta |

## 4. Resultados
**Viabilidade (312 fins de mês, 2000-01-31 a 2025-12-30):**

| Variante | Mínimo de emissores elegíveis | Data do mínimo | Mediana | Fins de mês < 60 | Rotatividade mensal U1 / U2 |
|---|---|---|---|---|---|
| só BDI 02 | 68 | 2002-07-31 | 186,5 | **0** | 2,8% / 5,7% |
| 02 + regimes especiais | 68 | 2002-07-31 | 192 | **0** | 2,8% / 5,8% |

Mínimo por ano (só BDI 02): 2000: 78; 2001: 70; **2002: 68**; 2003: 71; 2004: 82; 2005: 87; 2006: 92; 2007: 106; de 2008 em diante, ≥ 155.

- **[Fato]** Há ≥ 60 emissores elegíveis em **todos** os fins de mês de 2000 a 2025. U1 e U2 são viáveis.
- **[Fato]** Em 2001–2003 a folga é pequena (68–71), então U2 inclui emissores perto do limiar de presença nesse período. **[Inferência]** A comparação de liquidez U1 × U2 tem amplitude menor no início da amostra.

**Sobrevivência:**

| Variante | Instrumentos que já estiveram em U1/U2 | Pararam de negociar > 31 dias antes do fim da base |
|---|---|---|
| só BDI 02 | 259 | 159 |
| 02 + regimes especiais | 260 | 153 |

- **[Fato]** Os arquivos históricos incluem instrumentos hoje inexistentes (ex.: ACES4, AEDU3, AMBV4, ALLL3, AGIN3). **Confirma-se que o universo pode ser reconstruído ponto no tempo**, o que responde à hipótese de §6.6.
- **[Limitação]** O COTAHIST não distingue deslistagem de reorganização (incorporação, troca de classe, migração de unidade). Por exemplo, AMBV4 deixa de negociar em 2013, na reorganização que criou ABEV3. O valor terminal de DE-01.d exige a tabela de eventos (DA-10/DD-03).

**Identidade:** 1.545 ISINs e 1.551 tickers (só BDI 02); 14 ISINs com mais de um ticker e 7 tickers com mais de um ISIN. A identidade por ISIN evita juntar papéis diferentes sob o mesmo ticker, mas as trocas de ISIN numa reorganização aparecem como saída de um instrumento e entrada de outro.

**Regimes especiais:** 133 ISINs passaram por BDI de regime especial, e **123 deles também negociaram no lote padrão (02)**. Com "só BDI 02", uma ação líquida que entra em recuperação judicial sai do universo naquele momento.

**Exclusões** (só BDI 02, registros do mercado à vista):
- BDI fora: 1.306.489;
- BDR (DRN, DR1–DR3, DRE): ≈ 317 mil;
- fundos (CI): 2.439;
- recibos e direitos (R11, R33, R42, RON, RPN…): ≈ 3,2 mil;
- preferenciais resgatáveis (PRA, PRB, PRC): 677;
- ESPECI malformado de ações, com parênteses ou aspas (`PNA"`, `PNA(`, `PNB"`, `PNB(`, `PNC(`, `PND(`, `PNE(`): **89 registros excluídos pela regra de token**, impacto desprezível frente a milhões de registros, mas registrado. Uma primeira versão deste relatório dizia 87, porque o JSON guardava só as 30 causas mais frequentes; corrigido para gravar todas.

## 5. Brechas e pendências encontradas
1. **Membro sem negócio em t:** a presença de 95% é medida na janela, então um instrumento pode estar no universo num fim de mês **depois** do seu último negócio (AGIN3: último negócio em 2010-02-11, ainda membro em 2010-02-26). O sinal de DE-01 já exige negócio em t, mas o instrumento ocupa uma vaga de U1/U2. **Recomendação:** exigir também negócio em t para pertencer ao universo.
2. **Variante de BDI (afeta sobrevivência):** recomenda-se **02 + regimes especiais**. Excluir regimes especiais retira as ações exatamente quando entram em dificuldade, o que é viés de sobrevivência. A contagem mínima não muda (68).
3. **Convenção do trailer de 2025:** aceita e registrada (§2). Se o arquivo de 2025 for republicado pela B3, o hash do manifesto acusará a mudança.
4. **ESPECI com parênteses ou aspas:** decidir se entram (normalizando o token) antes do estudo; o impacto é de 89 registros.
5. **Parquet do bruto tipado (DA-09):** adiado. A auditoria lê os zips diretamente; o Parquet de todas as colunas fica para a normalização, onde é necessário.

## 6. Recomendação para DD-01 (aguarda aprovação)
- Universo U1 (postos 1–30) e U2 (31–60), BDI **02 + regimes especiais**, presença ≥ 95% em 252 pregões até t **e negócio em t**.
- **Período:** formações de 2000-01 a 2025-12. Com o aquecimento de 24 meses de DE-01, a matriz de retornos começaria em 2002. É o mesmo período desta auditoria, viável em todos os fins de mês.
- **Antes do pré-registro:** DD-03 (eventos e valor terminal), DE-01.h (venda adiada), DD-05 e o valor de S.
