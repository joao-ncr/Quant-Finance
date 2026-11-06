# DE-07 — Protocolo de calibração, desenho não nulo e seleção de (N, V)

**Versão:** v1 (2026-11-06). Histórico: v1-proposta (commit `33ed2dd`) → **decisões D1–D7 aprovadas pelo desenvolvedor** (2026-11-06, Etapa B1). A matriz final de células e o manifesto são fixados na Etapa B3 (§7.4); o hash deste arquivo é registrado no manifesto e qualquer alteração posterior exige uma v2.
**Escopo:** só dados sintéticos. Nenhum dado real da B3 é lido neste ciclo.
**Base:** commit `0fe6231`; ledger íntegro com 8 registros, head `8:714a3329…` (verificado em 2026-11-06).

## 0. Decisões aprovadas e classificação do conteúdo

### 0.1 Decisões D1–D7 (aprovadas em 2026-11-06, com as ressalvas do desenvolvedor)
| ID | Decisão aprovada | Ressalva registrada |
|---|---|---|
| D1 | Critério de sucesso 2 → **2a** (taxa de falsos positivos do DSR ≤ α em **toda** estrutura avaliada), **2b** (nível nominal do PSR de uma configuração escolhida antes), **2c** (poder reportado) | as taxas são julgadas com a incerteza de Monte Carlo (intervalos e testes do §5.3), nunca só pela estimativa pontual |
| D2 | α = 0,05 (limiar DSR 0,95) | **só** para avaliar a calibração do método. DE-09 continua PENDING; nenhum limiar universal de PBO é criado |
| D3 | Seleção de (N, V): domínio → validade com Holm → maior poder no pior caso; sem par aprovado, nenhum vencedor | V_ρ = V_H0·(1 − ρ̂̄) entra como **candidato experimental**, derivação própria, não estimador estabelecido na literatura. A inclusão não antecipa aprovação |
| D4 | Médias específicas persistentes de média transversal zero, pela regra DE-01 completa; δ* ∈ {1; 2,5; 4}; h ∈ {63; 252}; célula separada de prêmio de mercado sem momentum | calibração da intensidade **só** pelo Sharpe verdadeiro do gerador (§7.2) |
| D5 | R = 20 000 (gaussiano), 1000 (momentum sem sinal), 500 (momentum com sinal) | **orçamento de até 2 h**, obtido reduzindo **células**, nunca réplicas. As seis células `low_common` com sinal ficam **fora** desta rodada. A matriz de §7.4 (preliminar, ≈ 6 h) é substituída pela matriz final da Etapa B3 |
| D6 | PBO sem limiar universal: distribuição nula da família, AUC nula × sinal, comparação com o sobreajuste verdadeiro | — |
| D7 | Execuções confirmatórias: `SYNTHETIC`, fase confirmatória, vinculadas ao hash do protocolo, só com commit limpo | a Etapa B não registra nenhuma execução confirmatória |

### 0.2 Classificação do conteúdo deste documento
| Categoria | Onde | Natureza |
|---|---|---|
| **Resultados já observados** | §1.2 (execuções exploratórias), §2.2 (conferência da Tabela A com os registros) | exploratórios; nenhum entra nas regras de decisão |
| **Resultados analíticos** | §2.2 (Tabela A), §2.3 (identidade de V), §2.4 (inexistência de N*) | determinísticos, verificados por teste; justificam o desenho, não decidem |
| **Hipóteses e justificativas** | §3, §7.1, §7.3 e as marcas [inferência] | premissas de modelagem |
| **Critérios que serão testados** | §4, §5, §6 | fixados antes da Etapa C |
| **Decisões só possíveis após a Etapa C** | adoção (ou não) de um par (N, V), do estimador de N efetivo e de V_ρ; status final de DE-03, DE-04.b e do item 5 de METHODOLOGY §1.9; insumos para DE-09 | dependem das execuções confirmatórias |

**Registro explícito (P7):** a taxa exploratória "0–8% (média 4%)" de DSR > 0,95 atribuída à grade (METHODOLOGY §7.6, leitura 4) **não foi reproduzida pelo código versionado** e **não pode ser usada como evidência confirmatória**.

**Inalterados por este ciclo:** DE-09, DD-01, DE-01.h, DE-01.i e DD-03 continuam no estado em que estavam.

Marcas usadas: **[fato]** (número registrado no ledger ou identidade matemática verificada por teste), **[inferência]**, **[lit., não conferido]** (referência conhecida, texto integral não lido nesta sessão), **[proposta]** (depende de aprovação).

**Regra de leitura.** Nada neste documento escolhe estimador, V ou limiar. O documento fixa *como* a escolha será feita, antes das simulações confirmatórias. As simulações anteriores são **exploratórias** (§1.2) e só motivam o desenho.

---

## 1. Problemas identificados

### 1.1 Lista
| ID | Problema | Origem | Seção |
|---|---|---|---|
| P1 | A "pendência de escala do PSR" não é um erro de unidade no código. É a proposta (emenda 4) de medir o erro **na escala do PSR**. Há, porém, um descompasso conceitual de escala: o PSR padroniza pelo desvio de **um** ŜR, e o DSR aplica isso ao **máximo** de N ŜR | METHODOLOGY §7.6, emenda 4 | §2.1–2.2 |
| P2 | A taxa P(DSR > 0,95) sob a nula vai de ≈ 0,1% (tentativas independentes) a ≈ 5% (fator comum dominante). O critério de sucesso 2 ("próxima do nominal") não diz em que estrutura vale | DE-07, `run-de53b361…`, `run-36aa72b3…` | §2.2, §5.1 |
| P3 | O erro relativo de SR₀ divide por E[máx ŜR] ≈ 0 na grade. Com 50 réplicas o alvo é ruído, e a regra mecânica escolheu um par com erro de 2710% | DE-03, METHODOLOGY §7.6 | §2.4, §4.3 |
| P4 | Na grade, o "N efetivo" no sentido do DSR **não existe**: E[máx Z] < f(2) | [fato, §2.4] | §2.4 |
| P5 | V entre tentativas (provisório) foi avaliado só sob a nula global, em que V_H0 é exato por construção. Sob sinal, V entre tentativas incorpora a dispersão dos SR verdadeiros | DE-04.b | §2.3 |
| P6 | A PBO de **uma** realização sob a nula tem percentis 5–95% de [0,02–0,21] a [0,76–0,96]. Um limiar universal (0,5) não tem base | DE-09, `run-36aa72b3…` | §6 |
| P7 | A taxa "0–8% (média 4%)" de DSR > 0,95 na grade **não tem código versionado que a produza** (o `grid_study` não calcula DSR; o artefato guarda só os ŜR). É um número post hoc não reproduzível pelo repositório | verificação por busca no código, 2026-11-06 | §1.2 |
| P8 | Numa família só comprada, o SR verdadeiro de todas as configurações inclui o prêmio de mercado sobre o CDI. O DSR testa "SR da melhor > 0", e não "momentum > 0" | DE-01 (long-only, excesso ao CDI) | §3.3 |

### 1.2 Inventário: exploratório × confirmatório
| Execução | Natureza | Pode ser usada como |
|---|---|---|
| `run-de53b361…` (`sim-neff-preliminary`, 9 estruturas gaussianas, 200 mil réplicas) | exploratória | motivação; checagem das fórmulas analíticas (§2.2) |
| `run-de18681b…` (`sim-momentum-mechanics`, 1 réplica) | mecânica | validação do motor; correlações descritivas |
| `run-7dd0ad19…` (falha, emenda 3) | falha registrada | nada (sem resultados) |
| `run-36aa72b3…` (`sim-grid-neff-embargo`, 50 réplicas × 4 cenários) | exploratória; a seleção de par foi declarada não informativa | motivação; decisão do embargo (DE-06, já tomada com regra prévia) |
| Taxa post hoc 0–8% (§7.6, leitura 4) | exploratória, **não reproduzível pelo código** (P7) | nada além de motivação |
| Réplica de cronometragem da emenda 2b | não registrada | nada |

Nenhum desses números entra nas regras de decisão do §5. As execuções confirmatórias usarão sementes novas, fixadas no manifesto congelado (§8).

---

## 2. Fórmulas e convenções revisadas

### 2.1 Análise dimensional (P1)
Todas as grandezas do caminho de cálculo, conferidas no código em 2026-11-06:

| Grandeza | Definição no código | Unidade |
|---|---|---|
| e_t | `r_p,t − r_f,t` (`core.metrics`, `core.momentum`) | retorno diário (adimensional por dia) |
| ŜR | `mean(e)/std(e, ddof=1)` | dia^(−1/2) implícito: SR **diário** |
| γ₃, γ₄ | g₁, b₂ plug-in da mesma série | adimensionais |
| T | nº de linhas da série | observações (pregões) |
| Var(ŜR) sob H₀ | (1 − γ₃ŜR + (γ₄−1)/4·ŜR²)/(T − 1) | SR diário² |
| V_H0 | 1/(T − 1) (`grid_study.aggregate`, `neff_study`) | SR diário² |
| V entre tentativas | `var(ŜR₁…ŜR_M, ddof=1)` | SR diário² |
| SR₀ | √V·f(N) | SR diário |
| z do PSR | (ŜR − SR*)·√(T−1)/√radicando | adimensional |
| Âncoras A1–A7 | V anual 1/2 → 1/(2·250); ŜR anual 2,5 → 2,5/√250 | convertidas para diário antes de entrar |

**[Fato]** Não há mistura de grandezas anuais e diárias em nenhum caminho do cálculo: √252 só aparece em apresentação (`annualized_sharpe`), as âncoras são convertidas no teste e reproduzidas, e T e V usam a mesma série. As 7 âncoras continuam passando.

**[Fato]** A "escala do PSR" mencionada na emenda 4 é a proposta de padronizar o erro de SR₀ por σ_SR = √Var(ŜR), isto é, medir o erro em unidades do z do PSR. **Não é um erro de implementação** e não há correção numérica a fazer.

**[Fato, ruído e não escala]** V marginal frente a V_H0 variou de −36% a +20% na grade (METHODOLOGY §7.6). Com 50 réplicas e configurações com correlação ≈ 0,98 (que quase não diversificam o erro), o intervalo de 95% da razão de variâncias com 49 graus de liberdade é [0,64; 1,43]. O caso −36% está na borda. Com 1000 réplicas o erro-padrão relativo cai para ≈ 4,5%, e o protocolo mede isso de novo (§4.4).

### 2.2 O descompasso de escala do DSR (P1, P2)
O PSR é Φ de (ŜR − SR*)/σ_SR, em que σ_SR é o desvio do estimador **de uma tentativa**. O DSR aplica o PSR a ŜR_máx com SR* = SR₀ = E[máx]. Sob a nula gaussiana, com ŜRᵢ = σ·Zᵢ, Z ~ N(0, C) e σ = 1/√(T−1):

```
P(DSR > 1 − α) = P( máx Z − SR₀/σ > z_{1−α}·√radicando ) ≈ P( máx Z − E[máx Z] > z_{1−α} )
```

A estatística centrada (máx Z − E[máx Z]) **não** tem desvio 1. Seu desvio depende de C:
- tentativas independentes: o desvio do máximo de M normais é ≈ 0,4 para M ~ 100, e a cauda acima de 1,645 é rara;
- equicorrelação: máx Z = √ρ·W + √(1−ρ)·máx ε. Com ρ → 1, o termo √ρ·W domina e o desvio tende a 1.

**Fórmula exata (equicorrelação, V tratado como fixo)**, implementada em `core.null_models.prob_max_exceeds_equicorrelated` e `dsr_null_rejection_rate_equicorrelated`, sem aleatoriedade:

```
P(máx Z > c) = ∫ φ(w)·[1 − Φ((c − √ρ·w)/√(1−ρ))ᴹ] dw
```

com c o limiar do DSR em unidades de Z, que inclui o radicando sob normalidade (raiz de uma quadrática; ver o docstring).

**Tabela A — P(DSR > 0,95) sob a nula global, T = 6049 (o comprimento da grade), retornos normais.** Valores determinísticos, verificados em `tests/unit/test_null_models.py`:

| M | V | ρ = 0 | 0,5 | 0,9 | 0,95 | 0,98 | 0,99 | 0,999 |
|---|---|---|---|---|---|---|---|---|
| 4 | V_H0 (N = M) | 1,39% | 1,24% | 0,75% | 0,62% | 0,52% | 0,47% | 0,38% |
| 4 | E[V entre tentativas] = (1−ρ)σ² | 1,39% | 2,86% | 4,50% | 4,73% | 4,88% | 4,93% | 4,99% |
| 16 | V_H0 | 0,45% | 0,38% | 0,14% | 0,10% | 0,06% | 0,05% | 0,04% |
| 16 | E[V entre tentativas] | 0,45% | 2,04% | 4,30% | 4,62% | 4,83% | 4,90% | 4,98% |
| 144 | V_H0 | 0,12% | 0,09% | 0,015% | 0,007% | 0,004% | 0,002% | 0,001% |
| 144 | E[V entre tentativas] | 0,12% | 1,60% | 4,23% | 4,60% | 4,83% | 4,91% | 4,98% |

**Conferência com os registros exploratórios [fato]:**
- M = 100, T = 1250, independentes, V_H0: 0,145% (analítico) contra 0,13% (registrado, 200 mil réplicas);
- equi 0,9, (M, V entre tentativas): 4,22% contra 4,22%;
- equi 0,5: 1,63% contra 1,55%;
- independentes com V entre tentativas *aleatória*: 0,06% registrado contra 0,14% analítico com V fixo. **[Inferência]** Quando o máximo é alto, a dispersão também é, então SR₀ sobe junto e a taxa cai. A fórmula com V fixo é cota, não valor exato, para V aleatório.

**Leituras.**
1. **[Fato]** Com V = E[V entre tentativas] e N = M, a taxa nunca passa de α na equicorrelação e tende a α quando ρ → 1. **[Inferência]** No limite, a família é uma única tentativa (o componente comum), e o DSR volta a ser o PSR(0) de um teste único, com nível exato. Com tentativas independentes, o DSR é conservador **por construção**: SR₀ = E[máx] fica abaixo do quantil 1 − α do máximo.
2. **[Fato]** Com V_H0, a taxa cai para quase zero quando ρ cresce: V_H0 ignora a correlação e N = M a ignora também.
3. **[Inferência]** O DSR com limiar 0,95 **não é um teste de nível exato 5%**. Ele pode ser um teste **válido** (taxa ≤ α) cujo conservadorismo depende da estrutura. Isso fundamenta a reformulação do critério de sucesso (§5.1).

### 2.3 V: o que representa e como evitar o desconto duplo (P5)
**Identidade exata [fato; `null_cross_variance_factor`, testada]:** para Z ~ N(0, C) com qualquer correlação C,

```
E[s²(Z)] = (tr C − 1ᵀC1/M)/(M − 1) = 1 − ρ̄,   ρ̄ = média das correlações fora da diagonal.
```

Portanto, sob a nula global e com variâncias marginais iguais, **E[V entre tentativas] = σ²·(1 − ρ̄)**. A correlação já está dentro de V.

Consequências:
- **(N efetivo, V entre tentativas):** desconta a correlação duas vezes, uma em N e outra em V. É anticonservador, como os registros mostraram (até −36% em SR₀ e taxa de 7,7%). **Incoerente por construção** [inferência a partir da identidade].
- **(M, V entre tentativas):** desconta uma vez, em V. Na equicorrelação, √(1−ρ)·f(M) ≈ √(1−ρ)·E[máx ε] = E[máx Z] (erro só da aproximação f). Coerente.
- **(N efetivo, V_H0):** desconta uma vez, em N. Coerente em princípio, mas só onde N efetivo está definido (§2.4).
- **(M, V_H0):** não desconta. Cota conservadora.

**Sob sinal [inferência]:** com SR verdadeiros SRᵢ de variância transversal τ², E[V entre tentativas] ≈ σ²(1 − ρ̄) + τ². O benchmark sobe com o próprio sinal, o que custa poder. O artigo define V como "the variance across the trials' estimated SR" [verificado visualmente, eq. 2] e não discute essa contaminação no trecho conferido.

**Candidato experimental [D3, incluído em 2026-11-06]:** V_ρ = V_H0·(1 − ρ̂̄), com ρ̂̄ estimado de Ĉ. É a esperança nula de V entre tentativas, sem τ². **Não está na literatura conferida.** É uma derivação própria a partir da identidade acima. Está no conjunto avaliado, mas sua adoção depende da Etapa C. Ele **não** muda o conjunto de estimadores de N (`ESTIMATORS`).

**Descritivo × inferencial.** V é inferencial quando entra no SR₀ de um DSR reportado. As grandezas "V marginal" (variância de ŜR entre réplicas) e "N*" só existem na simulação e são **referências descritivas**, nunca entradas do DSR.

### 2.4 Quando "N efetivo" existe (P3, P4)
O N implícito N* resolve f(N*) = E[máx Z]. A aproximação f do artigo só representa valores ≥ f(2) = 0,520 (DE-04.c). Na equicorrelação, E[máx Z] = √(1−ρ)·E[máx ε]:

| M | ρ = 0,9 | 0,95 | 0,98 | 0,99 |
|---|---|---|---|---|
| 4 | 0,326 | 0,230 | 0,146 | 0,103 |
| 16 | 0,558 | 0,395 | 0,250 | 0,177 |
| 144 | 0,833 | 0,589 | 0,373 | 0,264 |

**[Fato]** Com a correlação da grade (0,93–0,99, METHODOLOGY §7.2), E[máx Z] < f(2) em quase todas as combinações: **N* não existe**. A família vale "menos de duas tentativas independentes" em termos de máximo esperado. Isso explica, sem recorrer a ruído, por que N* só foi definido em 3 das 12 estruturas e por que Nyholt, Galwey e A.3 deram N̂ < 2.

**[Fato]** E[máx ŜR] na grade não é zero: é ≈ 0,37·σ ≈ 0,0048 (F144, ρ = 0,98), com σ = 0,0129. Com 50 réplicas o erro-padrão de 0,0015 o tornava indistinguível de zero. Com 1000 réplicas, o erro-padrão é ≈ 0,0003.

**Consequência para DE-03 [inferência]:** nessa região, "estimar N efetivo corretamente" é uma pergunta mal posta, porque o alvo não existe. As perguntas que continuam bem postas são (b) reproduzir E[máx] e (c) produzir um DSR válido. Por isso o protocolo separa os três objetivos (§4.3) e decide só por (c).

---

## 3. Hipóteses estatísticas

### 3.1 Nula
- **H₀ global:** SR verdadeiro = 0 para todas as configurações. Em painel: média de todos os ativos = r_f, retornos sem previsibilidade na média (cenários de `momentum_scenarios`, inalterados).
- **Descoberta:** DSR(melhor configuração IS) > 1 − α. Sob H₀ global, toda descoberta é falsa. A taxa estimada é o **nível empírico** do procedimento.

### 3.2 Alternativa de momentum (sinal transversal)
- **H₁-mom:** retornos esperados específicos persistentes e de média transversal zero (§7.1). A carteira de vencedoras tem SR verdadeiro > 0, e as configurações diferem entre si.
- **Descoberta verdadeira:** rejeição **e** SR verdadeiro da configuração escolhida > 0. Rejeição com SR verdadeiro ≤ 0 conta como falsa, mesmo sob H₁.
- **Poder:** P(descoberta verdadeira).

### 3.3 Alternativa de prêmio comum (validade de construto, P8)
- **H₁-com:** todos os ativos têm excesso esperado π > 0, sem momentum. Todas as configurações têm SR verdadeiro ≈ SR do mercado.
- **[Inferência]** O DSR deve rejeitar aqui, e isso é correto para a hipótese que ele testa ("a melhor tem SR > 0"). Para a pergunta de pesquisa, porém, é uma rejeição **sem momentum**. A célula mede a magnitude do problema. Ela não muda DE-01 (decidida), mas é insumo obrigatório da interpretação dos resultados reais (§9 e D6).

### 3.4 PSR de uma configuração pré-especificada
- H₀: SR = 0 para a configuração (j, k) = (6, 6), escolhida antes como a primeira de F4 na ordem canônica.
- Aqui o nível **deve** ser nominal: P(PSR(0) > 1 − α) ≈ α. Este é o item 5 de METHODOLOGY §1.9 (precisão da aproximação assintótica sob caudas e aglomeração de volatilidade).

---

## 4. Métricas de avaliação

### 4.1 DSR (decisórias)
Por estrutura e família, separadamente (nunca agregadas entre estruturas):
- **nível empírico** p̂₀ = #rejeições/R sob H₀, com intervalo Clopper-Pearson de 95%;
- **poder** p̂₁ sob H₁-mom, com intervalo Clopper-Pearson;
- **conservadorismo** p̂₀/α, descritivo;
- taxa de rejeição sob H₁-com, descritiva (§3.3).

### 4.2 PSR (decisória para o item 5 de §1.9)
- p̂ = P(PSR(0) > 1 − α) da configuração (6, 6) sob H₀, por cenário;
- Kolmogorov-Smirnov dos valores de PSR contra U(0,1), descritivo.

### 4.3 N efetivo e V (três objetivos separados)
| Objetivo | Métrica | Uso |
|---|---|---|
| (a) estimar N efetivo | distribuição de N̂ (média, desvio, fração < 2); viés contra N* **só onde N* existe** com folga (E[máx]/σ_marg ≥ f(2) + 2 EP). Onde não existe: "sem referência", sem valor inventado | descritivo |
| (b) reproduzir E[máx] | erro padronizado z_E = (E[√V̂·f(N̂)] − E[máx ŜR])/σ_marg, com EP por réplica pareada. Sempre definido, inclusive com E[máx] ≈ 0 | diagnóstico (explica (c)) |
| (c) inferência válida | nível e poder do DSR (§4.1) | **decisório** |

O erro relativo (critério original de §4.2/§7.3) fica **aposentado** como critério: é mal definido quando o alvo se aproxima de zero (P3). Os valores continuam reportados quando |E[máx]| > 4 EP, só como descrição.

### 4.4 Diagnósticos de escala
- V marginal / V_H0, com intervalo pela distribuição qui-quadrado aproximada;
- E[máx ŜR]/σ_marg contra a Tabela A ou a quadratura equivalente, para checar a aproximação gaussiana com o motor de momentum.

### 4.5 PBO
Ver §6.

---

## 5. Critérios de seleção e rejeição

### 5.1 Critério de sucesso 2, reformulado [DECIDED, D1]
Texto original (documento de partida): *"na simulação sem sinal, a taxa de falsa descoberta fica próxima do nível nominal"*.

Proposta:
> **2a. Validade do DSR.** Para cada estrutura nula do protocolo, o nível empírico de "DSR > 1 − α" não excede α de forma estatisticamente detectável (teste binomial unilateral, correção de Holm entre as estruturas do par, FWER de 5%). O conservadorismo (p̂₀/α) é reportado por estrutura, nunca como média.
> **2b. Calibração do PSR.** Para uma configuração pré-especificada, P(PSR(0) > 1 − α) é compatível com α (bilateral, Bonferroni entre cenários).
> **2c. Poder.** O poder sob H₁-mom é reportado por intensidade de sinal e estrutura. Ele não tem limiar de aprovação, mas decide entre pares válidos (§5.2).

Justificativa:
- o DSR não é um teste de nível exato (§2.2), e exigir "≈ 5%" reprovaria um procedimento válido e conservador com tentativas independentes, ou aprovaria por acaso uma média entre estruturas;
- "não exceder α" é a definição usual de teste válido de nível α; a proximidade com α é uma propriedade de eficiência, que aparece no poder (2c);
- o nível exato fica exigido onde a teoria o promete: o PSR de uma configuração única (2b).

Alternativas consideradas:
- (i) manter "próxima do nominal" com tolerância de Monte Carlo: reprova o DSR em estruturas com tentativas independentes, em que ele é conservador por desenho;
- (ii) trocar o benchmark E[máx] pelo quantil 1 − α do máximo nulo estimado por reamostragem (no espírito de White 2000 e Hansen 2005 [lit., não conferido]): é outro método, não o DSR; fica registrado como possível trabalho futuro e **fora** deste ciclo;
- (iii) tratar o DSR como puramente descritivo, sem critério: abandona a pergunta do projeto.

### 5.2 Seleção do par (N, V) [DECIDED, D3]
Conjunto avaliado: 6 estimadores de N × {V entre tentativas, V_H0, V_ρ} = 18 pares, todos pré-registrados. V_ρ é **candidato experimental** (D3), derivação própria.

1. **Domínio:** um par é inadmissível numa estrutura se N̂ < 2 em alguma réplica. O DSR não é calculável, e trocar N̂ silenciosamente seria um improviso (DE-04.c). A fração indefinida é reportada.
2. **Validade:** o par é inadmissível se violar 2a em qualquer estrutura nula, seja de nível Z ou de nível P (§7).
3. **Escolha:** entre os admissíveis, o que maximiza o **menor poder** nas células H₁-mom de intensidade intermediária (*maximin*).
   - Empate (diferença pareada ≤ 2 EP) → maior poder médio em todas as células H₁-mom → menor max |z_E|.
4. **Sem admissível → nenhum par validado.** Não se escolhe "o menos ruim".
5. **Alcance:** a escolha vale para as estruturas testadas. Extrapolar para dados reais exige verificar que a correlação estimada da grade real cai na faixa coberta (condição de aplicabilidade a registrar).

O estimador de N efetivo "adotado" é o N do par escolhido. Se esse N for o M bruto, a conclusão registrada é que **nenhum estimador de N efetivo foi necessário nem validado** para esta estrutura, porque a correlação entra via V.

### 5.3 Incerteza Monte Carlo (valores calculados para o protocolo)
| R | p = 5%: EP / IC 95% CP | p = 1% | p = 0,1% | excesso detectável (poder 80%), 1 teste / Holm com 12 |
|---|---|---|---|---|
| 500 | 0,97 pp / [3,3; 7,3] | 0,44 pp | 0,14 pp (IC [0; 0,74]) | 7,7% / 9,0% |
| 1000 | 0,69 pp / [3,7; 6,5] | 0,31 pp | 0,10 pp (IC [0; 0,56]) | 6,9% / 7,7% |
| 20 000 | 0,15 pp / [4,7; 5,3] | 0,07 pp | 0,02 pp | — |

- **Eventos raros:** com R ≤ 1000, taxas de ordem 0,1% (independentes, F144) **não são estimáveis** além de "< 0,6%". O nível Z (R = 20 000) cobre esse regime. O nível P não pretende fazê-lo.
- **Percentis da PBO** (intervalos livres de distribuição por estatísticas de ordem): com R = 500, o percentil 5% fica entre os postos 16 e 36; com R = 1000, entre 37 e 65.
- **Poder:** em 50%, EP de 2,2 pp (R = 500).
- **Limitação declarada:** com R = 1000 e Holm entre 12 estruturas, um excesso real de nível menor que ≈ 2,7 pp acima de α pode passar despercebido. O protocolo reporta essa sensibilidade junto com o veredito.

---

## 6. Protocolo da PBO

**Princípio [DECIDED, D6]:** a PBO de uma realização é interpretada **contra a distribuição nula da própria família e estrutura**, sem limiar universal. DE-09 continua PENDING para os dados reais.

Por família (F4, F16, F144) × estrutura:
- **distribuição nula:** média, desvio, percentis {5, 10, 25, 50, 75, 90, 95} com IC por estatística de ordem, e a função de distribuição empírica completa no artefato;
- **probabilidades de cauda descritivas** P₀(φ ≥ t) para t ∈ {0,5; 0,75; 0,9}, como pontos da distribuição, não como limiares;
- **discriminação nula × sinal, sem limiar:** AUC = P(φ_H₁ < φ_H₀) + ½·P(empate) (estatística de Mann-Whitney), por família × célula H₁-mom;
- **PBO contra a verdade (só no sinal):** o simulador conhece o SR verdadeiro de cada configuração, então define-se o sobreajuste verdadeiro O = 1{SR verdadeiro da escolhida IS ≤ mediana dos SR verdadeiros}. Compara-se E[φ] com P(O = 1). Isso testa se a PBO estima o que diz estimar. Fica indefinido sob H₀ (todos os SR iguais);
- **sensibilidade a N:** F4 ⊂ F16 ⊂ F144, na mesma réplica;
- **sensibilidade à correlação:** cenários com correlação média de 0,93 a 0,99 e o nível Z;
- **sensibilidade a sementes:** dispersão entre réplicas (é a própria distribuição); a estrutura temporal fica coberta pela célula `vol_clustering`.

**Regra de interpretação para o estudo real (a fixar no pré-registro, D6):** reportar φ_obs com o seu percentil numa distribuição nula gerada por um modelo nulo **ajustado à estrutura transversal dos dados reais** (médias forçadas a r_f). O método de ajuste é decisão do pré-registro, não deste ciclo.

---

## 7. Cenários e desenho não nulo

### 7.1 Gerador não nulo [DECIDED, D4]
Extensão do gerador atual (`simulate_factor_returns`), com o mesmo fator, betas e volatilidades, mais um termo de média específica:

```
r_i,t = r_f + π + μ_i,t + β_i·σ_t·u_t + s_i·(σ_t/σ̄)·ε_i,t
μ_i,t = a_i,t − (1/n)·Σ_j a_j,t                     (média transversal zero)
a_i,t = φ·a_i,t−1 + √(1 − φ²)·s_a·ξ_i,t,   φ = 2^(−1/h)
```

- **H₀:** s_a = 0 e π = 0, o que reproduz exatamente os cenários atuais.
- **H₁-mom:** s_a > 0 e π = 0. Retornos esperados específicos persistentes geram momentum transversal [lit., não conferido: Conrad e Kaul 1998; Lewellen 2002]. A média transversal zero impede que o sinal vaze para o prêmio de mercado.
- **H₁-com:** s_a = 0 e π > 0.
- **Por que este gerador:** o sinal entra nos **preços** e passa pela regra DE-01 inteira (tercil, pesos iguais, coortes sobrepostas, grade j × k). Nenhuma configuração é favorecida por construção: qual (j, k) vence emerge de h e do ruído. O sinal é independente da seleção posterior.
- **Alternativas descartadas:**
  - plantar alfa nas séries das estratégias: ignora a mecânica e torna a vencedora conhecida por construção;
  - autocorrelação dos retornos: mistura sinal com dependência temporal, que o DE-06 já tratou como inexistente sob H₀.

### 7.2 Intensidades e persistência [DECIDED, D4]
- **Intensidade definida pelo alvo, não pelo parâmetro:** δ* = máx_{F144} SR verdadeiro × √(T − 1) ∈ {1,0; 2,5; 4,0} (fraco, intermediário, forte).
  - Fundamento: δ é o deslocamento do z do PSR. Um teste único pré-especificado de 5% teria poder ≈ 26%, 80% e 99%.
  - Isso ancora as intensidades na escala do próprio teste, **sem olhar o DSR nem a PBO**.
- **Persistência:** meia-vida h ∈ {63, 252} pregões (≈ 3 e 12 meses), para que a configuração ótima varie entre células.
- **Calibração de s_a (fase de desenvolvimento, §8):** para cada (h, δ*), uma busca por bissecção sobre o SR verdadeiro estimado por oráculo. Ela usa só a média dos ŜR em réplicas com fluxo próprio, nunca DSR, PBO ou seleção. A regra é fixada agora; o valor resultante vai para o manifesto antes da execução confirmatória.
- **H₁-com:** π tal que o SR verdadeiro da carteira igualmente ponderada dê δ = 2,5.

### 7.2.1 Implementação e calibração (Etapa B2, 2026-11-06)
**Módulos:**
- `core.signal_models`: μ_i,t (AR(1) estacionário desde t = 0, φ = 2^(−1/h), média transversal exatamente zero) e o deslocamento π + μ_i,t. Puro: recebe o `Generator`.
- `experiments.de07_design`: o painel de uma réplica soma o deslocamento aos retornos simples do cenário (`momentum_scenarios.scenario_panel(..., mean_shift)`). Depois, a grade é construída por `build_momentum_returns` (regra DE-01 inteira: sinal pelos preços, tercil, pesos iguais, coortes sobrepostas). Não há atalho.
- **Fluxos:** `(raiz, *caminho, 0)` gera o ruído do cenário, idêntico aos controles nulos; `(raiz, *caminho, 1)` gera só μ. Sem sinal, o painel é **bit a bit** o cenário nulo original (teste com digest do código anterior).
- `experiments.de07_calibration`: a calibração. `experiments.de07_confirmatory`: o executor de uma célula. `experiments.de07_evaluation` e `core.mc_inference`: as regras de §4–§6, implementadas e testadas antes da Etapa C. `experiments.de07_manifest`: leitura estrita e escrita determinística do manifesto.

**Sharpe verdadeiro (definição exata).** SR_c = E[e_c,t]/√Var(e_c,t): momentos **incondicionais** do excesso diário da carteira DE-01 da configuração c sob o processo gerador estacionário, na escala diária do PSR. Não há forma fechada. O valor é estimado por oráculo: média e variância **agregadas** sobre todas as linhas de R réplicas independentes (combinação exata de grupos), com erro-padrão pelo desvio dos ŜR por réplica dividido por √R. Sob H₀, SR_c = 0 exatamente.

**Intensidade.** δ* = máx_{c∈F144} SR_c·√(T − 1), com T = 6049 linhas do calendário comum.

**Procedimento de calibração (fixado no código antes de rodar):**
1. Para h ∈ {63, 252}: δ̂*(s) numa grade s ∈ {0,00025; 0,0005; 0,001; 0,002; 0,004} (desvio estacionário diário de a_i,t), com 30 réplicas e **números aleatórios comuns** entre escalas. A grade é estendida por fatores de 2 se não cobrir os alvos (até 4 vezes de cada lado).
2. δ̂*(s) deve ser estritamente crescente; caso contrário, erro.
3. s(alvo) vem da interpolação linear em (log s, log δ̂*).
4. Verificação com fluxo independente e 100 réplicas, que produz a tabela de SR verdadeiros congelada no manifesto. ~~Aceita se |δ̂* − alvo| ≤ 0,25 (≈ 2,5 erros-padrão). Senão, há um único refinamento pela inclinação local, com outro fluxo, e nova verificação; se falhar, erro, e a execução falha no ledger.~~ **Substituído pela emenda 5 (§7.2.3).**
5. H₁-com: π = 2,5·σ_EW/√(T − 1), com σ_EW exato (β̄²σ_m² + Σs_i²/n²); SR verdadeiros pelo oráculo.

**Separação calibração × avaliação (testada):**
- a calibração e o gerador não importam DSR, PSR, PBO, CSCV, estimadores de N, a avaliação nem o executor (verificação estática das importações);
- a calibração usa uma semente raiz própria, e o manifesto exige sementes raiz distintas entre calibração e avaliação;
- o executor confirmatório só lê a tabela de verdades congelada (SHA-256 no manifesto).

**Limitações da calibração:**
- δ̂* é o máximo de 144 estimativas ruidosas: viés para cima. Como as configurações têm correlação ≈ 0,98, o componente não comum do erro é pequeno (≈ √0,02 × 0,1 ≈ 0,014 em δ) [inferência];
- a tolerância de 0,25 em δ equivale a cerca de ±7 pp no poder de um teste único com δ = 2,5;
- os SR verdadeiros têm erro de Monte Carlo (EP ≈ 0,1 em unidades de δ), registrado por configuração;
- a definição de intensidade usa F144; as famílias F4 e F16 da mesma célula têm δ ≤ δ*.

### 7.2.2 Execução da calibração e falha registrada (2026-11-06)
**Execução:** `run-434c74846c8042d0a40610301e7ea065` (`de07-signal-calibration`, `phase = calibration`, commit `582910b` com árvore limpa, 2 processos). **Status no ledger: `failed`**, com a mensagem "calibração não convergiu: h=252.0, alvo 1.0, δ̂*=1.452". Sem artefato.

O que foi observado. São valores do gerador, não da avaliação dos métodos:

| h | Grade δ̂*(s), 30 réplicas: s = 0,00025 / 0,0005 / 0,001 / 0,002 / 0,004 | Verificação, 100 réplicas, fluxo independente |
|---|---|---|
| 63 | 0,18 / 0,50 / 1,74 / 5,51 / 15,23 | alvo 1 → 0,857 (s = 0,000733); alvo 2,5 → 2,586 (s = 0,001243); alvo 4 → 4,046 (s = 0,001649): **dentro da tolerância** |
| 252 | 0,66 / 1,28 / 3,46 / 8,94 / 21,02 | alvo 1: fora da tolerância na verificação e no refinamento (último δ̂* = 1,45) → **erro; procedimento interrompido** |

As células h = 252 com δ* ∈ {2,5; 4} e a H₁-com não foram calibradas.

**Diagnóstico [inferência, não verificada por nova execução]:** a tolerância de 0,25 supôs um erro-padrão de δ̂* ≈ 0,1 com 100 réplicas, isto é, só o ruído i.i.d. dos retornos (≈ 1/√R). Com meia-vida longa (h = 252), as médias específicas de cada réplica persistem por cerca de um ano. Assim, o lucro de momentum realizado varia muito entre réplicas, e o erro-padrão de δ̂* fica maior; o da grade (30 réplicas) maior ainda, o que também desloca a escala interpolada. A discrepância de ≈ 0,45 entre grade e verificação é compatível com isso. Não há evidência de defeito de código: os testes do procedimento passam, e h = 63 convergiu.

**Consequência:** sem tabela de verdades válida, o manifesto **não foi gerado nem congelado**. Corrigir o procedimento depois de ver a falha é uma decisão metodológica e aguarda o desenvolvedor (relatório da Etapa B). A execução falha permanece no ledger e não é descartada.

### 7.2.3 Emenda 5 — tolerância proporcional à incerteza (aprovada em 2026-11-06, antes da nova calibração)
**Decisão do desenvolvedor:** opção 1 do relatório da Etapa B (tolerância proporcional à incerteza). Ela substitui a tolerância absoluta de 0,25. A execução que falhou (`run-434c7484…`) permanece no ledger, sem alteração. D1–D7, DD-01, DE-01.h, DE-01.i e DD-03 não mudam.

**Por que a tolerância absoluta era inadequada.** Os 0,25 supunham um erro-padrão de δ̂* ≈ 0,1 com 100 réplicas, ou seja, só o ruído i.i.d. dos retornos. Com meia-vida longa (h = 252), as médias específicas de uma réplica persistem por cerca de um ano, e o lucro de momentum realizado varia entre réplicas por causa do próprio sorteio do sinal. O erro-padrão real depende da célula e é desconhecido a priori: uma tolerância absoluta é estrita demais onde a incerteza é grande e frouxa demais onde é pequena.

**Estimação do erro-padrão (revista).**
| Item | Definição |
|---|---|
| Estatística estimada | SR_c agregado (média e variância de todas as linhas das R réplicas, combinação exata de grupos) e δ̂* = máx_c SR_c·√(T − 1) |
| Unidade amostral independente | **a réplica** (fluxos `generator_for` distintos e independentes) |
| Erro-padrão | **jackknife "deixa uma réplica de fora"** aplicado ao próprio estimador: o SR agregado recalculado sem a réplica r (remoção exata do grupo), EP = √((R−1)/R·Σ(θ₍₋ᵣ₎ − θ̄)²); para δ̂*, o jackknife do **máximo**, que inclui a seleção da configuração |
| Dependência preservada | pregões e ativos de uma réplica nunca são separados: cada réplica entra ou sai inteira, com toda a dependência temporal (persistência, coortes sobrepostas) e transversal (fator comum, média transversal zero) |
| Correção feita no código | antes, o EP era o desvio dos ŜR por réplica / √R: a unidade era correta, mas o estimador era outro (média de ŜR, não o SR agregado), e o EP de δ̂* ignorava o máximo. Agora o EP é o jackknife do estimador efetivamente reportado |
| Hipóteses da regra de 2 EP | réplicas i.i.d. (verdadeiro por construção); δ̂* aproximadamente normal entre réplicas (TLC sobre R = 100 réplicas); jackknife consistente para o SR agregado (função suave de médias). Para o máximo, o jackknife é aproximado quando há empate próximo entre configurações; com correlação ≈ 0,98 e a vencedora estável, o efeito esperado é pequeno [inferência] |

**Regra de calibração (em unidades de δ; equivalente em SR, dividindo por √(T − 1)):**
- **calibrada:** |δ̂* − δ*| ≤ 2·EP;
- **reprovada:** |δ̂* − δ*| > 2·EP; recebe um único refinamento pela inclinação local, com fluxo próprio, e nova classificação;
- **inconclusiva:** 2·EP > L(δ*), com **L(δ*) = metade da distância de δ* ao nível mais próximo do desenho em {0, 1, 2,5, 4}**. Isso dá L = 0,5 para δ* = 1 (vizinho 0, a nula) e L = 0,75 para δ* = 2,5 e 4. Uma célula inconclusiva não é refinada, porque uma nova verificação com o mesmo R tem a mesma imprecisão. Ela não é aprovada nem descartada.

**Justificativa de L (fixada antes da nova execução).** A calibração existe para colocar cada célula num nível de intensidade **distinguível** dos demais níveis do desenho, inclusive da nula. Se 2·EP > L, o intervalo de aceitação inclui valores mais próximos de um nível vizinho que do alvo, e "aprovar" não garantiria que a célula representa o nível pretendido. Por isso um EP grande torna a célula inconclusiva, não aprovada. O limite deriva só da grade de intensidades aprovada em D4, e não de resultados.

**H₁-com.** π é analítico (σ_EW exato), então δ_EW alcançado = 2,5 por construção (classificação "calibrada", método analítico). O δ̂* em F144 e seu EP são registrados como informação.

**Registro por célula:** alvo δ*, δ̂* alcançado, SR verdadeiro estimado (tabela por configuração), EP, tolerância (2·EP), limite L, |δ̂* − δ*|, status, marca de inconclusiva e tentativas. **O rótulo δ* não é o valor alcançado:** a Etapa C usa a tabela de SR verdadeiros efetivamente estimados.

**Execução:** todas as células (h ∈ {63, 252} × δ* ∈ {1; 2,5; 4} e H₁-com), com **semente raiz fixa 7514066048065518955**, nova, registrada aqui antes da execução e no ledger. A grade (30 réplicas) e a verificação (100 réplicas) mantêm os fluxos separados; nenhuma réplica foi aumentada. O manifesto só é gerado se **todas** as células estiverem calibradas (o gerador recusa os outros status). Se alguma sair inconclusiva ou reprovada, o trabalho para e aguarda decisão.

### 7.3 Representatividade e redundância dos cenários nulos (§4.2 do pedido)
| Regime | Estruturas | Cobertura | Observação |
|---|---|---|---|
| Independente | gaussiano ρ = 0 | DSR muito conservador; eventos raros | só nível Z |
| Moderado | equi 0,5; blocos 10×10 0,9; AR(1) 0,9 | taxa intermediária | só nível Z |
| Fator comum forte | equi 0,9; AR(1) 0,99; grades separáveis | taxa → α | nível Z |
| Grade de momentum (0,93–0,99) | `base`, `low_common`, `heterogeneous`, `vol_clustering` | a estrutura real do problema | nível P |
| Correlação extrema | equi {0,95; 0,98; 0,99} × M ∈ {4, 16, 144} | ponte entre nível Z e nível P | nível Z, **acrescentado** |

**[Fato]** `base` e `vol_clustering` têm a mesma correlação média (0,983 contra 0,984). **[Inferência]** São redundantes para N e V, mas não para o PSR: só `vol_clustering` gera curtose e agrupamento de volatilidade. Os dois são mantidos. `low_common` e `heterogeneous` dão as menores correlações da grade (0,93–0,97).

**Lacuna declarada:** nenhum cenário de nível P cobre correlação moderada, porque a carteira só comprada herda o fator de mercado (DE-01). O nível Z cobre esse regime, sem a mecânica.

### 7.4 Desenho completo [proposta original de D5; **substituída pela matriz final da Etapa B3**, que respeita o orçamento de 2 h]
| Nível | Células | R por célula |
|---|---|---|
| Z (gaussiano, ŜR = σ·Z, C fixa) | 9 genéricas + 9 de correlação extrema = 18, só H₀ | 20 000 |
| Z, sinal | as mesmas 18, com SR verdadeiros = δ·σ·(vetor de sinal pré-especificado: uma configuração; 10% das configurações) | 20 000 |
| P, H₀ | 4 cenários × (F4, F16, F144 na mesma réplica) | **1000** |
| P, H₁-mom | `base` × δ* ∈ {1; 2,5; 4} × h ∈ {63; 252} = 6 células | 500 |
| P, H₁-com | `base` × δ = 2,5 | 500 |
| P, oráculo | por célula de sinal, fluxo próprio, só médias de ŜR | 100 |

`low_common` com sinal (+6 células) fica como opção de D5.

---

### 7.5 Matriz de células (Etapa B3, 2026-11-06; substitui §7.4) — **proposta; congelamento bloqueado pela falha da calibração (§7.2.2)**
Gerada por código (`experiments.de07_build_manifest`) e congelada em `configs/studies/de07_confirmatory_v1.toml`. Os SR verdadeiros das células com sinal ficam em `configs/studies/de07_signal_truth_v1.json`. R segue D5 sem exceção: 20 000 por célula gaussiana, 1000 por célula de momentum sem sinal, 500 por célula de momentum com sinal.

| Célula(s) | Hipótese | Parâmetros | R | Métricas | Objetivos cobertos | Custo estimado (relógio) |
|---|---|---|---|---|---|---|
| `G0-*` (18) | H₀ | 9 estruturas genéricas (§4.4 de METHODOLOGY) + equi ρ ∈ {0,95; 0,98; 0,99} × M ∈ {4, 16, 144}; T = 6049; N̂ sobre a C verdadeira | 20 000 | rejeição do DSR nos 18 pares, z_E, N̂ | 1, 2 | 18 × 10,5 s |
| `G1-*` (33) | H₁-gauss | as mesmas estruturas; sinal δ = 2,5 em `single` (1 configuração) e `decile` (⌈0,1·M⌉; omitido com M = 4) | 20 000 | poder complementar, **não decisório** | 4 (complementar) | 33 × 10,5 s |
| `P0-base` | H₀ | cenário `base`; F4, F16, F144; CSCV S = 16 nas três | 1000 | DSR (18 pares), PSR (6, 6), PBO, N̂, V, ρ̂̄ | 1, 2, 3, 6 (nula e referência das AUC) | 3687 s |
| `P0-vol_clustering` | H₀ | cenário `vol_clustering`; F4, F16; CSCV nas duas | 1000 | idem | 1, 2, 3 (caudas e aglomeração), 6 | 533 s |
| `P1-mom-h{63,252}-d{1; 2,5; 4}` (6) | H₁-mom | `base` + μ_i,t calibrado (§7.2.1); F4, F16; CSCV nas duas | 500 | idem + descoberta verdadeira, PBO verdadeira, sobreajuste verdadeiro | 4, 6, 7; as duas com δ* = 2,5 são as `power_cells` de D3 | 6 × 270 s |
| `P1-com-d2.5` | H₁-com | `base` + π (δ_EW = 2,5), sem μ; F4, F16; CSCV | 500 | idem | 5, 6 | 270 s |

Total estimado: **≈ 1,85 h** (momentum: custo sequencial medido / ganho 1,5 com 2 processos; gaussiano sequencial; 10 s de sobrecarga por célula). Fica abaixo do limite de 2 h, com folga de ≈ 8%.

**Cobertura dos objetivos (Etapa B3):**
1. Calibração sob diferentes dependências: 18 estruturas gaussianas e 2 estruturas da grade.
2. Falsos positivos do DSR: todas as células H₀, com Holm por par (D3).
3. Nível do PSR pré-selecionado: (6, 6) em `P0-base` e `P0-vol_clustering`.
4. Poder: 3 intensidades × 2 persistências (F4/F16), mais o nível gaussiano complementar.
5. Prêmio sem momentum: `P1-com-d2.5`.
6. PBO sob a nula e sob sinal: distribuições por família; AUC contra `P0-base` com o mesmo cenário e família.
7. PBO × sobreajuste verdadeiro: PBO verdadeira e sobreajuste verdadeiro nas células com sinal.

**Células excluídas e motivo:**
| Exclusão | Motivo |
|---|---|
| Qualquer célula com sinal em F144 | ≈ 1840 s cada, total ≈ 2,3 h. **Limitação:** poder e PBO sob sinal ficam medidos para N ∈ {4, 16}; para F144 há só evidência de validade (nível sob H₀) |
| `P0-low_common`, `P0-heterogeneous` | correlação de 0,93–0,97 coberta pelas gaussianas equi 0,95 e pela `P0-base`; mesma mecânica; +≈ 530 s cada, sem folga |
| `P0-vol_clustering` com F144 | +≈ 3150 s; o objetivo da célula (PSR e nível sob aglomeração) é atendido com F4/F16 |
| Seis células `low_common` com sinal | excluídas por D5 |
| Padrão gaussiano `decile` com M = 4 | idêntico a `single` (⌈0,4⌉ = 1) |

**Ordem de execução:** células gaussianas (rápidas, um processo) → `P0-base` → `P0-vol_clustering` → `P1-mom-*` → `P1-com-d2.5`. Cada célula é uma execução no ledger. Uma célula interrompida é reexecutada inteira, com a mesma semente, e a falha permanece registrada.

**Sementes:** uma raiz por nível (`gaussian`, `momentum`), distintas entre si e da raiz da calibração. Fluxo da réplica = (raiz, ordem da célula, réplica, 0/1).

## 8. Réplicas, custo e separação das fases

### 8.1 Justificativa de R
- **Nível P, H₀: R = 1000.** A validade (2a) é o critério decisório. Com R = 500, um nível real de 9% passaria pela correção de Holm com 20% de chance (§5.3).
- **Nível P, H₁: R = 500**, o mínimo do pedido. O poder decide entre pares válidos, e EP ≤ 2,2 pp basta para separar diferenças de ≈ 6 pp.
- **Nível Z: R = 20 000.** O custo é desprezível e permite medir taxas de 0,1%.
- Nenhuma réplica é reduzida para caber no orçamento (CLAUDE.md §11.2).

### 8.2 Custo (medido em 2026-11-06, só engenharia, sem estatísticas; estimativa da proposta original, revista na Etapa B3)
- Grade F144: 3,7 s. CSCV F144 com S = 16: 1,1 s. Réplica completa: ≈ 5 s.
- Nível P: 4000 (H₀) + 3500 (H₁) + 700 (oráculo, ≈ 3,7 s cada) ≈ 11,2 h sequencial, ≈ 6 h com 2 processos.
- Calibração de s_a: ≈ 1 h.
- Nível Z: minutos.
- **Excede o orçamento provisório de "≤ 2 h por estudo completo" (DA-07).** Isso exige aprovação (D5). As alternativas são reduzir **células**, nunca réplicas.

### 8.2.1 Medições da Etapa B2 (2026-11-06, só engenharia: tempo e memória, nenhuma estatística lida)
| Medida | Valor |
|---|---|
| Réplica de momentum F4 + F16 + F144 com CSCV S = 16 nas três, sequencial | 5,52 s |
| Réplica F4 + F16 com CSCV nas duas, sequencial | 0,79 s |
| Réplica F144 completa, 2 processos, regime estável (40 réplicas) | **3,50 s efetivos (ganho 1,58)** |
| Ganho com 2 processos em lotes pequenos (8–16 réplicas) | 1,04–1,32 (custo de iniciar processos) |
| Pico de memória por processo (réplica F144 completa) | 247 MiB (`PER_WORKER_BYTES` = 300 MiB) |

- **Pré-condição de memória:** 2 processos exigem cerca de 0,6 GiB dentro do orçamento de 60% da RAM disponível, ou seja, ≥ 1 GiB disponível. Em medições anteriores havia só 0,75 GiB (orçamento de 0,45 GiB, um processo).
- O executor confirmatório **falha explicitamente** se não puder usar os processos previstos no orçamento do manifesto, em vez de exceder o tempo em silêncio.

### 8.3 Fases e registro (sem segundo mecanismo)
| Fase | O que é | Registro | Pode ser reportado como evidência? |
|---|---|---|---|
| Exploratória | §1.2, já feita | ledger existente | não |
| Desenvolvimento | testes de implementação com R pequeno; calibração de s_a pelo oráculo | `RunKind.DEVELOPMENT` (testes) e `RunKind.SYNTHETIC` com `parameters.phase = "calibration"` (calibração) | só o valor de s_a, que vira parâmetro |
| Confirmatória | protocolo congelado | `RunKind.SYNTHETIC`, `parameters.phase = "confirmatory"`, `preregistration_reference` = (este arquivo, versão, SHA-256), sementes fixadas no manifesto versionado, commit limpo exigido | sim |

- **Congelamento:** depois de D1–D7 aprovadas e de s_a calibrado, o protocolo vira v1 com manifesto `configs/studies/de07_confirmatory_v1.toml`, contendo cenários, intensidades, R, sementes raiz e SHA-256 deste documento, num commit próprio **antes** da execução.
- **Mudança depois de congelar:** nova versão (v2), novo SHA e novas execuções. Resultados de versões diferentes não se misturam.
- **Falhas e interrupções:** ficam no ledger (`failed`/`aborted`). Uma célula interrompida é reexecutada por inteiro com a mesma semente, e o registro da falha permanece.

---

## 9. Regras de interpretação
1. O nível e o poder são lidos **por estrutura**. Uma média entre estruturas nunca sustenta "calibrado".
2. "DSR válido" significa nível ≤ α nas estruturas testadas, não nível exato.
3. Sem par admissível, a conclusão é "nenhum validado", e o estudo real reporta o DSR com **todos** os pares como sensibilidade (DE-03 protocolo), sem principal.
4. Se o par escolhido for (M, ·), registra-se que N efetivo não foi necessário. Isso não valida nenhum estimador de N.
5. PBO alta sob H₀ é esperada. φ_obs só é informativa em relação à distribuição nula da família.
6. **Prêmio comum (P8):** se a célula H₁-com rejeitar com frequência, o relatório real deve separar "a melhor configuração tem SR > 0" de "há momentum". A forma de separar (por exemplo, reportar também a carteira igualmente ponderada do universo como configuração de referência) é decisão do pré-registro e **não** altera DE-01.
7. Contradições entre DSR e PBO (DSR fraco com PBO baixa, PBO alta sob H₀) são reportadas como achados, não eliminadas.

---

## 10. Decisões D1–D7 (histórico da proposta)
**Estado: todas aprovadas em 2026-11-06**, com as ressalvas de §0.1. A tabela abaixo é a proposta original, preservada como histórico; onde ela diverge de §0.1 (orçamento de D5), vale §0.1.

| ID | Decisão | Recomendação | Consequência se aprovada | Consequência se rejeitada |
|---|---|---|---|---|
| **D1** | Reformular o critério de sucesso 2 como 2a/2b/2c (§5.1) | aprovar | o DSR é avaliado por validade + poder; o PSR, por calibração exata | manter "próxima do nominal" exige escolher a estrutura em que vale, e o DSR seria reprovado com tentativas independentes |
| **D2** | α = 0,05 (limiar 0,95) **só para a avaliação do método** | aprovar; DE-09 continua aberto para os dados reais | taxas comparáveis com os registros anteriores | outro α muda R e o efeito detectável |
| **D3** | Regra de seleção do §5.2 (domínio → validade com Holm → maximin de poder) e inclusão do candidato V_ρ | aprovar a regra; V_ρ como **candidato adicional** (derivação própria, não literatura) | o par só é adotado se for válido em todas as estruturas | sem V_ρ, o conjunto fica com V entre tentativas e V_H0 |
| **D4** | Gerador não nulo (§7.1) com δ* ∈ {1; 2,5; 4}, h ∈ {63; 252} e uma célula H₁-com | aprovar | sinal passa pela mecânica DE-01; intensidades ancoradas no z do PSR | é preciso outro desenho antes da Etapa B |
| **D5** | R (Z: 20 000; P-H₀: 1000; P-H₁: 500; oráculo: 100) e orçamento de ≈ 6 h com 2 processos | aprovar; opcional: +6 células `low_common` (+≈ 4 h) | excede o "≤ 2 h" de DA-07 | só é possível cortar células (por exemplo, h = 252 apenas); cortar réplicas está proibido |
| **D6** | PBO sem limiar universal: percentil nulo, AUC e comparação com o sobreajuste verdadeiro (§6) | aprovar | DE-09 fica com insumo, sem limiar fixado agora | é preciso fixar um limiar com base própria |
| **D7** | Registro confirmatório como `SYNTHETIC` + `phase` + `preregistration_reference` + commit limpo (§8.3) | aprovar (decisão de engenharia) | separação auditável sem novo mecanismo | — |

**Fora do escopo e inalterados:** DD-01 (U1/U2), DE-01.h, DE-01.i, DD-03 e o pré-registro geral.

---

## 11. Limitações
- O modelo gaussiano de nível Z ignora caudas e dependência temporal. O nível P cobre a aglomeração de volatilidade, mas não caudas assimétricas nem dependência na média.
- A fórmula analítica (Tabela A) trata V como fixo. Com V aleatório ela é aproximada (§2.2).
- O gerador H₁-mom é **um** mecanismo de momentum. Outros (reação lenta a notícias, reversão) não são cobertos, e a conclusão vale para este mecanismo.
- Nyholt, Li-Ji, Gao e Galwey continuam conferidos só no código do `poolr`, não nos artigos. A atribuição do momentum a médias persistentes também não foi conferida no original.
- R = 1000 não detecta excessos de nível menores que ≈ 2,7 pp após a correção de Holm (§5.3).
- O orçamento computacional excede o provisório (D5).
