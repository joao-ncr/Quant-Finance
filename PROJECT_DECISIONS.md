# Registro de decisões — Frente E

Trilha de decisões estatísticas e arquiteturais do projeto *Validação estatística de regras simples na B3*.
Arquivo **versionado** (público). Status possíveis: `PENDING` · `IN PROGRESS` · `DECIDED`.

**Regras do registro**
- Uma decisão só vira `DECIDED` com justificativa escrita e data.
- Decisões que afetam a grade, o universo ou critérios de sucesso devem ser tomadas **antes** de ver resultados. Se uma delas for revista depois, registre a revisão (não sobrescreva) e conte as tentativas anteriores no log.
- Alternativas listadas abaixo são candidatas levantadas na fase de documentação, não recomendações.

**Modelo de entrada**
```
### ID — Título
- Status:
- Contexto e problema:
- Alternativas consideradas:
- Justificativa:
- Consequências:
- Referências:
- Data / revisão:
```

---

## A. Decisões estatísticas

### DE-01 — Regra de momentum e grade de parâmetros
- **Status:** **DECIDED (2026-11-06), regra A–D; não implementada.** Histórico: PENDING até esta data. Decidida **antes de qualquer dado real**: o ledger tem só uma execução sintética. Especificação completa em METHODOLOGY_CONVENTIONS.md §6.
- **Definição:**
  - momentum cross-sectional, só comprado: tercil superior dos elegíveis pelo retorno total de j meses (21·j pregões), formado no último pregão do mês, sem *skip*;
  - execução no fechamento do pregão seguinte;
  - k coortes sobrepostas com 1/k do capital cada e pesos iguais;
  - excesso `r_p − CDI`;
  - custos zero no primário, com giro reportado;
  - preço defasado só para avaliar posições, nunca para sinal ou execução;
  - deslistagem pelo valor terminal documentado, com limites de sensibilidade.
- **Implementação (2026-11-06, ciclo seguinte):** `core.momentum`, validado contra uma implementação de referência dia a dia (METHODOLOGY_CONVENTIONS.md §7.1). Lacunas encontradas na regra:
  - **DE-01.g DECIDED:** direção do desempate na fronteira do tercil — vence o menor identificador canônico, e os empates são contados;
  - **DE-01.h PENDING:** capital de uma venda adiada com coortes sobrepostas; o motor gera `PendingDecisionError`; decidir antes do estudo real;
  - **DE-01.i PENDING:** definição do giro;
  - coorte sem elegíveis é erro explícito.
- **Grade:** j, k ∈ {1, …, 12} meses (F144), com subfamílias aninhadas F4 = {6, 12}² e F16 = {3, 6, 9, 12}² (grade de Jegadeesh e Titman 1993) para a variação com N. Fixada antes dos dados e não ampliável.
- **Alternativas descartadas:**
  - série temporal (MOP 2012): desenhada para futuros e dependente de ponderação por volatilidade, que Kim, Tse e Wald (2016) apontam como fonte do resultado;
  - long-short: custo e restrições de aluguel na B3, e excesso ambíguo (DE-02, ressalva 1);
  - dimensões adicionais (quantil, skip, pesos), que inflariam N sem servir à pergunta.
- **Consequências:**
  - resolve a ressalva 1 de DE-02 (long-only ⇒ excesso bem definido) e o *timing* de DD-04;
  - H_max = 252 pregões alimenta a regra de S;
  - permite gerar a correlação da grade com preços sintéticos para escolher o estimador de N efetivo.
- **Limitações:**
  - o retorno mistura mercado e momentum;
  - sem custos, k curto fica favorecido;
  - a evidência brasileira é mista e só foi lida em resumos (DE-10);
  - a grade e a sobreposição de JT foram citadas sem leitura do texto integral.
- **Contexto original:**
- **Contexto e problema:** O documento de partida define só "momentum com grade de janelas de formação e de manutenção". O tamanho da grade é N, uma das variáveis da pergunta de pesquisa.
- **Alternativas consideradas:** momentum de série temporal vs. cross-sectional; long-only vs. long-short; sinal em retorno bruto vs. em excesso ao CDI; com ou sem "skip" do período mais recente; grades de tamanhos diferentes para estudar a variação com N.
- **Justificativa:** —
- **Consequências:** Define N, a matriz T × N da CSCV e o custo computacional (DA-07).
- **Data / revisão:** aberto em 2026-11-06.

### DE-02 — Convenções do Sharpe e PSR
- **Status:** **DECIDED (2026-11-06, 2ª sessão).** Histórico: IN PROGRESS (proposta formal, 1ª sessão) → DECIDED. O desenvolvedor delegou ao agente o fechamento das decisões da Etapa A.
- **Decisão:**
  - retornos simples diários em excesso ao CDI;
  - ŜR = média/desvio com ddof = 1;
  - √252 só para apresentação;
  - g₁ e curtose de Pearson b₂ plug-in, sem correção de viés;
  - PSR, DSR e MinTRL calculados em unidades diárias.
- **Justificativa (fatos verificados):** as definições do original conferem (2012, eq. 41 e código do A.3: SR "non-annualized", γ₄ bruta, entradas "on excess returns"; 2014, eq. 2). As 7 âncoras publicadas foram reproduzidas.
- **Ressalvas mantidas como dependências:**
  1. o retorno de carteira long-short autofinanciada já é um excesso (DE-01);
  2. a variância do PSR supõe i.i.d., e não se implementa correção de autocorrelação (DE-06/DE-07).
- **Alternativas descartadas:**
  - SR anualizado nas fórmulas: quebra a unidade de T;
  - G₁/G₂ corrigidos: o artigo usa momentos populacionais, e a diferença é O(1/T);
  - log-retornos: não se agregam entre ativos.
- **Implementação:** `core.metrics` (já existente) e `core.sharpe_inference` (novo).
- **Limitação:** a mistura de ddof = 1 no ŜR com m₂ plug-in nos momentos é uma inconsistência de ordem 1/T, aceita e documentada (METHODOLOGY_CONVENTIONS.md §1).
- **Proposta (resumo):** retornos simples diários em excesso ao CDI (`e = r − r_f`, com o CDI composto no intervalo [t₋₁, t)); Sharpe diário `mean/sd(ddof=1)`; anualização √252 só para apresentação; assimetria g₁ e curtose de **Pearson** b₂ plug-in (sem correção de viés); PSR com √(T−1) e termo (γ₄−1)/4; tudo em unidades diárias. Séries constantes ou com T < 2 geram erro explícito.
- **Implementado e testado** em `core.metrics`/`core.rates` (Sharpe, momentos, CDI). PSR/DSR não implementados.
- **Sub-decisão DE-02.a — DECIDED (2026-11-06):** **não há tamanho mínimo de amostra.**
  - O código impõe só o domínio matemático: T > 1, entradas finitas, γ₄ ≥ 1 + γ₃² e radicando > 0, com erro explícito em cada caso.
  - A precisão da aproximação é questão empírica (cobertura sob H₀, DE-07).
  - O MinTRL (eq. 13 de 2012) é reportado como informação, nunca como bloqueio.
  - Descartados:
    - "T > 30": regra de livro-texto citada pelo próprio artigo, sem base para caudas pesadas;
    - limiar por simulação agora: depende do T e das caudas do estudo.
  - Ver METHODOLOGY_CONVENTIONS.md §1.6.1–1.6.2.
- **Verificação (2026-11-06, visual):** Bailey e López de Prado (2012), eq. 41 e código do Apêndice A.3: SR não anualizado, √(n−1), γ₄ = curtose bruta (normal = 3), entradas sobre retornos excedentes. Análise da consistência da proposta e de DE-02.a (recomendação: sem limiar arbitrário, reportar o MinTRL e calibrar por simulação) em docs/DECISION_REVIEW_2026-11-06.md §2 e §8, aguardando aprovação.
- **Contexto e problema:** PSR depende de escala (diária vs. anual), de excesso ao CDI e de assimetria/curtose. A fórmula da literatura usa curtose bruta (não excesso).
- **Alternativas consideradas:** SR diário não anualizado nos cálculos e anualizado só no relatório; estimadores de momentos com ou sem correção de viés.
- **Justificativa:** — (conferir a fórmula no artigo original antes de decidir)
- **Consequências:** Escala inconsistente entre PSR e DSR invalida o resultado.
- **Referências:** Bailey e López de Prado (2012), *The Sharpe Ratio Efficient Frontier*.

### DE-03 — Número efetivo de tentativas independentes
- **Status:** **IN PROGRESS.**
  - Protocolo: DECIDED.
  - DE-03.a: DECIDED.
  - **Estimador principal: PENDING**, com critério pré-especificado e emenda 1 registrados.
  - Histórico: PENDING até 2026-11-06.
- **DE-03.a — DECIDED (2026-11-06): nenhum arredondamento.**
  - N efetivo contínuo = N usado na fórmula.
  - `floor(N)` existe só nos testes de comparação com o `poolr`.
  - Fundamento: Φ⁻¹(1−1/N) está definido para N real, e o `floor` é sempre anticonservador (de 7,99 para 7 o fator de SR₀ cai 4,9%).
  - **Achado:** Li-Ji é inteiro por construção (traço = M ⇒ Σf(λ) = M − Σ_{λ≥1}(⌊λ⌋−1)) e Gao é uma contagem, então o arredondamento só afeta Nyholt, Galwey e A.3.
- **Protocolo (DECIDED):**
  - candidatos pré-registrados: M bruto, Nyholt, Li-Ji, Gao, Galwey e A.3 (`core.effective_trials.ESTIMATORS`);
  - o DSR é reportado para todos;
  - o principal é escolhido por simulação **só com dados sintéticos e antes de dados reais**;
  - proibido escolher pelo DSR ou pelo resultado mais favorável.
  - Critério e emenda: METHODOLOGY_CONVENTIONS.md §4.2 e §4.5.
- **Resultado preliminar (2026-11-06, `run-de53b36114814d0398437a14fd8d100f`, 9 estruturas genéricas):**
  - Pelo critério original (V marginal correto), **nenhum candidato é adequado**: o pior |erro| de SR₀ vai de 104% (Li-Ji) a 219% (M bruto).
  - O estimador principal continua **aberto**. A escolha final depende das estruturas induzidas pela grade de momentum (após DE-01).
- **Avaliação na grade de momentum (2026-11-06, `run-36aa72b3…`, METHODOLOGY_CONVENTIONS.md §7.3 e §7.6):**
  - Com correlação de 0,93 a 0,99 entre configurações, E[máx ŜR] sob a nula é indistinguível de zero em 8 de 12 estruturas. O critério de erro **relativo** fica mal posto (erros de −647% a +5384%).
  - A saída mecânica da regra (`gao + V_H0`, pior erro de 2710%) **não é adotada**. **Nenhum estimador foi validado; o estimador principal continua PENDING.**
  - Os estimadores por autovalores dão N̂ < 2 em F4/F16 de vários cenários.
  - Próximo passo proposto (emenda 4, pendente): erro padronizado na escala do PSR, calibração P(DSR > 0,95) e ≥ 500 réplicas, pré-registrados junto com o desenho não nulo (DE-07).
  - **Etapa A de DE-07 (2026-11-06):** a emenda 4 foi absorvida pelo protocolo v1 proposto (docs/DE07_PROTOCOL.md §4.3 e §5.2). Os três objetivos ficam separados: estimar N (descritivo), reproduzir E[máx] (diagnóstico em unidades de σ_SR) e inferência válida (decisório). Com a correlação da grade, o N implícito não existe (E[máx Z] < f(2)). O estimador principal **continua PENDING**.
- **Limitação:** Nyholt, Li-Ji, Gao e Galwey foram implementados a partir do código do `poolr`; os artigos originais não foram conferidos. Galwey amplifica ruído de autovalores nulos (√ε). A fórmula A.3 extrapola para N̂ > M (até M + 1) quando a correlação média é negativa; o valor não é cortado (contraexemplo achado por teste de propriedade em 2026-11-06).
- **Contexto e problema:** Configurações vizinhas são correlacionadas. N bruto superdeflaciona; ignorar tentativas subdeflaciona.
- **Alternativas consideradas:** (a) N bruto como cota superior; (b) clusterização das séries de retornos, com N efetivo = nº de clusters (ONC: López de Prado e Lewis, *Quantitative Finance* 19(9), 2019); (c) medidas baseadas em autovalores da matriz de correlação (Nyholt 2004; Li e Ji 2005, *Heredity* 95(3); Galwey 2009, com implementação de referência em `poolr::meff`); (d) reportar sensibilidade do DSR a uma faixa de N.
- **Nota (2026-11-06, referência):** `poolr::meff` aplica `floor()` em todos os métodos; o DSR aceita N real. **Sub-decisão pendente DE-03.a:** arredondar ou não o N efetivo. No caso sintético de 4 blocos, os métodos da referência discordam muito (Nyholt 16, Gao 17, Li e Ji 7, Galwey 7); ver METHODOLOGY_CONVENTIONS.md §3.2.
- **Nota (2026-11-06):** os métodos espectrais vêm da genética (testes de associação) e assumem uma estrutura de correlação estável; avaliar se isso vale para retornos de estratégias antes de adotá-los.
- **Verificação (2026-11-06, visual):** o próprio artigo do DSR (Apêndice A.3, p. 14–15) propõe N̂ = ρ̂ + (1 − ρ̂)·M, com ρ̂ a correlação média entre as M tentativas, e aponta limitações (dependência só linear; matriz mal condicionada se T < M(M−1)/2). Análise dos estimadores, de `floor` (DE-03.a) e um protocolo de sensibilidade com escolha do estimador principal por simulação em docs/DECISION_REVIEW_2026-11-06.md §6, aguardando aprovação.
- **Justificativa:** —
- **Consequências:** Independentemente da escolha, o log de tentativas deve guardar as séries de retornos de cada tentativa (ver DA-03).
- **Referências:** Bailey e López de Prado (2014); López de Prado, *Backtesting* (SSRN 2606462).

### DE-04 — DSR: fórmula, variância entre tentativas e escopo de N
- **Status:** **IN PROGRESS.**
  - DE-04.a: DECIDED.
  - DE-04.c: DECIDED.
  - **DE-04.b: PROVISÓRIO** (emenda 1).
  - Histórico: PENDING até 2026-11-06.
- **DE-04.a — DECIDED:**
  - tentativa = configuração distinta da família avaliada nos dados do estudo (não é `run_id` nem fingerprint);
  - conta em M: concluídas, falhas ou abortadas nunca concluídas, e exploração com dados reais;
  - repetições idênticas e sementes de regra determinística contam uma vez;
  - execuções sintéticas não contam;
  - execuções com dados reais antes do pré-registro são proibidas.
  - Tabela em METHODOLOGY_CONVENTIONS.md §4.1.
  - Aberto para o pré-registro: M por faixa de liquidez ou total.
- **DE-04.c — DECIDED:**
  - N ≥ 2, e N < 2 gera `ContractViolationError`;
  - fundamento: o fator de SR₀ é negativo para N < 1,2836 e −∞ em N = 1;
  - N efetivo < 2 é resultado a reportar, não valor a corrigir.
- **DE-04.b — PROVISÓRIO (emenda 1, 2026-11-06, após a simulação preliminar e antes de qualquer dado real):**
  - par primário provisório: **(N = M bruto, V = variância entre tentativas com ddof = 1)**;
  - sensibilidade obrigatória: todos os N efetivos com V_H0 = 1/(T−1).
  - Motivo: nas 9 estruturas, o par (M, V entre tentativas) errou SR₀ entre +1% e +43%, sempre conservador e quase exato na equicorrelação. Os pares (N efetivo, V entre tentativas) chegaram a −36% (anticonservadores, porque descontam a correlação duas vezes).
  - A proposta anterior (V entre tentativas com N efetivo) fica registrada como **descartada por evidência**.
  - Revisar com as estruturas da grade (DE-01) e com um desenho não nulo (DE-07), porque sob a nula global V_H0 é exato por construção.
  - **Grade de momentum (2026-11-06, `run-36aa72b3…`):** V entre tentativas = 2–8% de V marginal.
    - Descritivo post hoc: (M, V entre tentativas) teve |erro| ≤ 0,20 σ_SR e taxa de DSR > 0,95 sob a nula de 0 a 8% (média 4%), nas 12 estruturas. Os pares com N efetivo chegaram a 10%; V_H0 foi conservador demais.
    - O primário provisório se mantém, **ainda PROVISÓRIO**, porque falta o desenho não nulo (DE-07) e a evidência é descritiva (o critério formal foi não informativo).
    - **Nota (2026-11-06, Etapa A de DE-07):** a taxa "0–8%" acima não é reproduzível pelo código versionado (o `grid_study` não calcula DSR). Fica classificada como exploratória, sem peso decisório. A coerência do par (M, V entre tentativas) agora tem base analítica: E[V entre tentativas] = σ²(1 − ρ̄) (identidade exata). Sob sinal, V entre tentativas incorpora a dispersão dos SR verdadeiros. Candidato adicional: V_ρ = V_H0·(1 − ρ̂̄), **incluído como candidato experimental pela D3 (DECIDED 2026-11-06)**, derivação própria, sem aprovação antecipada como V principal. Ver docs/DE07_PROTOCOL.md §2.3. **DE-04.b continua PROVISÓRIO** até a Etapa C.
- **Contexto e problema:** DSR = PSR com benchmark igual ao máximo esperado sob a nula, que depende de N e de V[{ŜRₙ}]. Falta definir quais tentativas entram em N e V (só a grade final? todas as já executadas?).
- **Alternativas consideradas:** N/V apenas da grade final vs. de todo o log de tentativas.
- **Sub-decisão pendente:** ddof da variância V entre tentativas (o texto extraído do artigo não especifica).
- **Forma da fórmula:** confirmada indiretamente pela reprodução do exemplo publicado (METHODOLOGY_CONVENTIONS.md §1.7–1.8), e visualmente na eq. 2 (p. 8) em 2026-11-06. O artigo define V como "the variance across the trials' estimated SR", sem ddof.
- **Análise (2026-11-06):** propostas DE-04.a (o que conta em N), DE-04.b (V e ddof) e DE-04.c (rejeitar N < 2, porque a expressão de SR₀ é negativa para N < 1,2836) em docs/DECISION_REVIEW_2026-11-06.md §7, aguardando aprovação.
- **Justificativa:** —
- **Consequências:** Determina a deflação reportada, que é o resultado central.
- **Referências:** Bailey e López de Prado, *The Deflated Sharpe Ratio*, JPM 40(5), 2014 (SSRN 2460551).

### DE-05 — CSCV e métrica da PBO
- **Status:** **IN PROGRESS.**
  - **Implementação (2026-11-06, Etapa B):** `core.cscv` e `experiments.cscv_input`, validados contra os 4 golden do `pbo`, inclusive o discriminante `pbo_odd`, contra a definição do artigo e contra a expectativa exata sob a nula (METHODOLOGY_CONVENTIONS.md §5.7).
  - Custo medido: 131 s com S = 16, T = 4528, N = 144. Há gatilho de otimização antes de simulações com muitas réplicas.
  - **Otimizada (2026-11-06):** `block_partition_performance` (agregação exata de estatísticas por bloco), 1,1 s com S = 16 e N = 144. Equivalente à forma direta, ao valor exato e aos golden do R (rtol 1e-12); `cscv_pbo` passou a usá-la (METHODOLOGY_CONVENTIONS.md §7.5).
  - Abertos: o valor de S (pré-registro) e o embargo (DE-06).
- Convenções DECIDED (2026-11-06):
  - **DE-05.a:** ω̄ = r̄/(N+1), como no artigo; o rank/N do `pbo` é descartado;
  - **DE-05.b:** o núcleo rejeita T % S ≠ 0; o experimento descarta as T mod S observações mais antigas por regra pré-registrada e registra as datas no ledger;
  - **DE-05.c:** colunas idênticas são erro; empate IS → primeira configuração na ordem canônica, contado; empates OOS → posto médio;
  - métrica = Sharpe DE-02 em J e J̄;
  - **S:** procedimento DECIDED (o maior par ≤ 16 com ⌊T/S⌋ ≥ max(63, H_max), sensibilidade S ∈ {8, 12, 16}); o **valor** é fixado no pré-registro, porque depende de T (DD-01) e de H_max (DE-01).
  - Justificativas e alternativas: METHODOLOGY_CONVENTIONS.md §5.1–5.5.
  - Histórico: PENDING até 2026-11-06.
- **Contexto e problema:** Escolher S (nº de blocos, par), métrica de desempenho, limiar e tratamento de empates. O pacote `pbo` usa S=4 por padrão e Omega como exemplo; o artigo usa Sharpe.
- **Alternativas consideradas:** S ∈ {8, 10, 16} vs. padrão; Sharpe vs. Omega; tamanho mínimo de bloco.
- **Achados na referência (`pbo` 1.3.5, código-fonte, 2026-11-06):** ω̄ = rank/N com empates pela média, e λ = +∞ substituído por 6; `which.max` resolve empates pelo primeiro índice; exige T divisível por S. **Sub-decisões pendentes:** ω̄ = rank/N (pacote) ou rank/(N+1) (verificar no artigo); regra para T não divisível por S; empates. Ver METHODOLOGY_CONVENTIONS.md §3.1.
- **Golden files** gerados com S ∈ {4, 8, 12}, T = 480 e N = 20 (ruído, sinal plantado, empates).
- **Golden discriminante (2026-11-06, Etapa B):** `pbo_odd`, ruído i.i.d. com N = 9 (ímpar), T = 480, S ∈ {4, 8, 12}, fluxo 6 da semente 20261106. Em S = 8 e 12 há partições com rank OOS de n* = (N+1)/2 = 5 (1 e 10 partições), em que as convenções divergem.
  - φ do `pbo` (rank/N): 0,1429 (S = 8) e 0,0942 (S = 12);
  - φ pela definição do artigo (rank/(N+1)), recalculado do `os_rank` do R: 0,1571 e 0,1050;
  - S = 4 coincide e serve de controle.
  - Há também partições com rank OOS = N (λ = +∞ → 6 no pacote; ln 9 no artigo).
  - Adicionar o caso regenerou os golden anteriores byte a byte idênticos.
- **Verificação no artigo original (2026-11-06, visual):** ω̄ = r̄/(N + 1) ∈ (0, 1) (p. 12); submatrizes "of equal dimensions", de ordem T/S × N (p. 11); φ = ∫₋∞⁰ f(λ) dλ (p. 13); o artigo não especifica desempate. O `pbo` (rank/N) diverge do artigo. Nos 8 golden files (N = 20), φ coincide nas duas convenções, mas a distribuição de λ não. Análise e recomendações (DE-05.a/b/c) em docs/DECISION_REVIEW_2026-11-06.md §5, aguardando aprovação.
- **Justificativa:** —
- **Consequências:** S define C(S, S/2) partições e o custo computacional.
- **Referências:** Bailey, Borwein, López de Prado e Zhu (SSRN 2326253).

### DE-06 — Janelas sobrepostas e vazamento
- **Status:** **IN PROGRESS.**
  - **Regra de construção DECIDED (2026-11-06):** a matriz M tem retornos diários realizados, posições com informação até t−1 e um calendário comum iniciado após a janela de formação mais longa. Por isso a purga é desnecessária.
  - **Embargo:** não adotado, em aberto com critério de adoção (afastamento sistemático da PBO na simulação da grade sem sinal).
  - **Embargo — DECIDED: não adotar (2026-11-06), para a grade DE-01:**
    - **Critério:** o de §5.6, corrigido antes da execução registrada pelas emendas 2 e 2b. A linha de base passou a ser a permutação de linhas, porque o valor exato para configurações permutáveis confunde estrutura transversal com dependência temporal.
    - **Evidência** (`run-36aa72b3…`): nenhum efeito temporal detectável em 12 estruturas (efeito mínimo detectável ≈ 0,02 de PBO); embargo de 21/63 sem ganho; |ACF| ≤ 0,003.
    - **Limitação:** cenários sintéticos sem dependência na média. Revisar só se a construção da carteira mudar.
  - Análise em METHODOLOGY_CONVENTIONS.md §5.6.
- **Contexto e problema:** Formação e manutenção sobrepostas geram autocorrelação nos retornos e dependência entre blocos CSCV adjacentes.
- **Alternativas consideradas:** embargo/purga entre blocos; blocos longos em relação à janela máxima de formação; correção de autocorrelação no PSR; manutenção sem sobreposição.
- **Justificativa:** —
- **Consequências:** Afeta a validade de PSR/DSR e da PBO.

### DE-07 — Simulação sem sinal e critério de taxa nominal
- **Status:** **IN PROGRESS.** Protocolo v1 com **D1–D7 DECIDED (2026-11-06, aprovadas pelo desenvolvedor)**; matriz de células e manifesto na Etapa B3; nenhuma simulação confirmatória executada. Histórico: PENDING → IN PROGRESS (protocolo proposto, Etapa A, `33ed2dd`) → D1–D7 DECIDED (Etapa B1).
- **Decisões aprovadas (docs/DE07_PROTOCOL.md §0.1):**
  - **D1 — DECIDED:** critério de sucesso 2 = 2a (falsos positivos do DSR ≤ α em toda estrutura avaliada), 2b (nível nominal do PSR de configuração pré-escolhida, (6, 6)), 2c (poder reportado), sempre com incerteza de Monte Carlo (Clopper-Pearson, teste binomial, Holm).
  - **D2 — DECIDED:** α = 0,05 (limiar 0,95) só para avaliar o método. **DE-09 continua PENDING**; nenhum limiar universal de PBO.
  - **D3 — DECIDED:** seleção de (N, V) por domínio → validade (Holm) → maior poder no pior caso; sem par aprovado, nenhum vencedor. **V_ρ = V_H0·(1 − ρ̂̄) é candidato experimental** (derivação própria, não literatura); a inclusão não antecipa aprovação.
  - **D4 — DECIDED:** gerador com médias específicas persistentes de média transversal zero, pela regra DE-01 completa; δ* ∈ {1; 2,5; 4}; h ∈ {63; 252}; célula de prêmio de mercado sem momentum; calibração só pelo Sharpe verdadeiro.
  - **D5 — DECIDED:** R = 20 000 / 1000 / 500; orçamento de até 2 h, reduzindo células e nunca réplicas; sem as células `low_common` com sinal.
  - **D6 — DECIDED:** PBO contra a distribuição nula da família, AUC e sobreajuste verdadeiro; sem limiar universal.
  - **D7 — DECIDED:** confirmatórias como `SYNTHETIC` + fase confirmatória + hash do protocolo + commit limpo.
- **Evidência que não vale como confirmatória:** a taxa "0–8%" de DSR > 0,95 na grade não foi reproduzida pelo código versionado (ver DE-04.b).
- **Etapa B2 (2026-11-06): gerador não nulo, calibração e executor implementados e testados** (docs/DE07_PROTOCOL.md §7.2.1 e §8.2.1). O sinal entra só como deslocamento da média dos retornos e passa pela regra DE-01 inteira. Os controles nulos ficam bit a bit idênticos. A calibração usa só o Sharpe verdadeiro (oráculo). A fase confirmatória do executor exige manifesto verificado, árvore limpa, código congelado, processos do orçamento e a flag explícita da Etapa C. Nenhuma execução confirmatória foi feita.
- **Calibração do sinal (2026-11-06): FALHOU** (`run-434c7484…`, `failed` no ledger). h = 63 convergiu para os três alvos; h = 252 com δ* = 1 ficou fora da tolerância de 0,25 após o refinamento (δ̂* = 1,45). Causa provável [inferência]: erro-padrão de δ̂* com h = 252 maior que o suposto na tolerância. **Manifesto não congelado; a correção do procedimento aguarda decisão** (docs/DE07_PROTOCOL.md §7.2.2).
- **Emenda 5 — DECIDED (2026-11-06, aprovada pelo desenvolvedor antes da nova calibração):** tolerância |δ̂* − δ*| ≤ 2·EP, com EP jackknife por réplica aplicado ao estimador reportado (máximo do SR agregado). Célula **inconclusiva** se 2·EP > metade da distância ao nível mais próximo em {0, 1, 2,5, 4}: não é aprovada nem descartada, e a calibração para. Semente fixa da nova calibração: 7514066048065518955. A falha anterior permanece no ledger. Detalhes em docs/DE07_PROTOCOL.md §7.2.3.
- **Só após a Etapa C:** adoção de um par (N, V), do estimador de N efetivo e de V_ρ; item 5 de METHODOLOGY §1.9.
- **Etapa A (2026-11-06), achados:**
  - **Escala do PSR:** a análise dimensional não encontrou mistura de unidades em nenhum caminho do cálculo (§2.1). A "escala do PSR" da emenda 4 é a proposta de medir o erro de SR₀ em unidades de σ_SR, não um erro de código.
  - **Causa da calibração dependente da estrutura [fato, fórmula exata testada]:** o PSR padroniza pelo desvio de **um** ŜR, e o DSR o aplica ao máximo. Na equicorrelação com V = E[V entre tentativas] e N = M, P(DSR > 0,95) vai de 0,12% (ρ = 0, M = 144) a 4,98% (ρ = 0,999), **sem passar de α**. A conta reproduz os registros exploratórios (4,22% contra 4,22%; 0,145% contra 0,13%). `core.null_models.dsr_null_rejection_rate_equicorrelated`.
  - **Identidade de V [fato]:** E[V entre tentativas] = σ²(1 − ρ̄) sob a nula, para qualquer correlação. Explica o desconto duplo dos pares (N efetivo, V entre tentativas).
  - **N efetivo na grade [fato]:** com ρ ≥ 0,95, E[máx Z] < f(2) = 0,52, então o N implícito não existe. O erro relativo é aposentado como critério decisório.
  - **Reprodutibilidade [fato]:** a taxa "0–8%" de DSR > 0,95 na grade (DE-04.b) não tem código versionado que a produza. Fica classificada como exploratória.
  - **Validade de construto:** numa família só comprada, o DSR também rejeita por prêmio de mercado (célula H₁-com do protocolo); insumo do pré-registro, sem alterar DE-01.
- **Propostas aguardando aprovação:** critério de sucesso 2 → 2a (nível ≤ α por estrutura, Holm), 2b (PSR de configuração pré-especificada com nível exato), 2c (poder); seleção de (N, V) por domínio → validade → maximin de poder, sem "menos ruim"; gerador não nulo com médias específicas persistentes de média transversal zero (δ* ∈ {1; 2,5; 4}, h ∈ {63; 252}); R = 20 000 (nível Z), 1000 (P, H₀), 500 (P, H₁); custo ≈ 6 h com 2 processos, acima do orçamento provisório de DA-07.
- **Achado descritivo (2026-11-06, `run-de53b361…`):**
  - Sob a nula global, com tentativas **independentes**, M = 100 e T = 1250, a regra DSR > 0,95 rejeitou em só 0,06% das réplicas (V entre tentativas) a 0,13% (V_H0).
  - Isso é coerente com SR₀ = E[máx], que não é o quantil de 95% do máximo [inferência].
  - **O DSR com limiar 0,95 não é um teste de nível 5%.**
  - **Complemento (2026-11-06, grade de momentum, descritivo post hoc):** com a correlação ≈ 0,98 da grade, o par (M, V entre tentativas) rejeitou em 0–8% (média 4%). **A calibração do DSR sob a nula depende da estrutura de correlação**: perto do nominal com fator comum dominante, muito abaixo com tentativas independentes. O critério de sucesso 2 precisa dizer em que estrutura vale. O critério de sucesso 2 ("próxima do nível nominal") precisa ser reformulado antes de qualquer simulação de validação: o que exatamente o DSR afirma controlar.
- **Contexto e problema:** Critério de sucesso: "taxa de falsa descoberta próxima do nível nominal". Faltam processo gerador, α, nº de réplicas, tolerância e definição de "descoberta".
- **Alternativas consideradas:** normal i.i.d.; permutação/bootstrap de retornos reais (preserva marginais, remove sinal); geração em preços (passando pela regra) vs. em retornos de estratégias.
- **Justificativa:** —
- **Consequências:** Tolerância deve considerar o erro de Monte Carlo e ser fixada antes de rodar.

### DE-08 — Referência de validação para DSR
- **Status:** **DECIDED (2026-11-06, 2ª sessão).**
  - As sete âncoras A1–A7 (DSR 2014: 3; PSR/MinTRL 2012: 4) estão congeladas em `tests/fixtures/published/sharpe_anchors.json`, com valores **como impressos**, páginas, URLs e SHA-256 dos PDFs. Todas são reproduzidas em `tests/reference/test_published_anchors.py`.
  - **Tolerância revista:** a meia unidade da saída **não basta**. A6 (0,95351 contra 0,953) a excede, porque o artigo imprimiu as entradas arredondadas. A regra adotada é a meia unidade da saída somada à propagação de primeira ordem da meia unidade das entradas impressas arredondadas.
  - O `pbo` não é referência de DSR.
  - Histórico: IN PROGRESS → DECIDED.
- Histórico anterior: IN PROGRESS. Referência identificada: o exemplo numérico de Bailey e López de Prado (2014), cujas três âncoras textuais (DSR ≈ 0,90; N = 46 → 0,9505; normal com N = 88 → 0,95) foram reproduzidas em 4 casas decimais por cálculo independente. Ver METHODOLOGY_CONVENTIONS.md §1.8.
- **Verificação (2026-11-06):** os parâmetros N = 100, V = 1/2 (anualizado, convertido no artigo para 1/(2·250)), T = 1250, γ₃ = −3, γ₄ = 10 e 250 obs/ano foram **conferidos visualmente** nas páginas 9–10 do preprint (SHA-256 em docs/DECISION_REVIEW_2026-11-06.md §0). Quatro âncoras adicionais do PSR, de Bailey e López de Prado (2012, p. 10–11 e 20), também foram reproduzidas. Proposta de fixture de constantes publicadas e tolerâncias na §10 desse documento, aguardando aprovação.
- **Contexto e problema:** O critério de sucesso cita "o pacote de referência", mas o `pbo` do CRAN implementa PBO, não DSR.
- **Alternativas consideradas:** exemplos numéricos do artigo original; cálculo manual independente; outra implementação pública auditada.
- **Justificativa:** —
- **Consequências:** Sem referência, o critério de sucesso 1 fica incompleto para DSR.

### DE-09 — Limiares de interpretação
- **Status:** PENDING
- **Contexto e problema:** A hipótese fala em DSR "bem menor" que o Sharpe bruto e PBO "material", sem limiares.
- **Alternativas consideradas:** limiares fixos (ex.: DSR > 0,95 como "significativo"; PBO > 0,5 como "material") vs. reporte contínuo sem dicotomia.
- **Evidência (2026-11-06, `run-36aa72b3…`):**
  - Sob a nula, a PBO de **um** conjunto de dados da grade varia muito: percentis 5% / 50% / 95% entre [0,02–0,21] / [0,36–0,61] / [0,76–0,96]. Uma PBO de 0,9 ocorre sem nenhum sinal.
  - "Material" deve ser definida contra a distribuição nula da **própria família**, obtida por simulação pré-registrada, e não contra 0,5.
- **Nota (2026-11-06, DE-07 Etapa A):** o protocolo v1 proposto (D6) avalia a PBO sem limiar: distribuição nula por família e estrutura, AUC nula × sinal e comparação com o sobreajuste verdadeiro conhecido no simulador. DE-09 continua **PENDING** para os dados reais.
- **Justificativa:** — (fixar antes de ver resultados)

### DE-10 — Revisão de literatura
- **Status:** PENDING
- **Contexto e problema:** Se já existir aplicação sistemática de DSR/PBO à B3, o plano é migrar para a Frente A.
- **Nota (2026-11-06):** segundo o desenvolvedor, a revisão preliminar concluiu que a metodologia DSR/PBO é aplicável ao problema. A pergunta específica (se já existe aplicação sistemática à B3) continua sem registro de resposta.
- **Nota (2026-11-06):** uma busca preliminar só por resumos encontrou evidência brasileira **mista** sobre momentum: Bonomo e Dall'Agnol (2003) e Fonte e Carmona (2005) rejeitam o efeito; Mussa et al. (2008) replicam JT; trabalhos mais recentes reportam o efeito em alguns períodos. Nenhum desses resumos menciona DSR ou PBO, mas isso **não** responde à pergunta de DE-10, que exige uma revisão própria.
- **Consequências:** Bloqueia investimento grande em implementação.

### Decisões de dados (estatísticas)

#### DD-01 — Universo e filtro de liquidez
- **Status:** **IN PROGRESS (proposta concreta, 2026-11-06).** Histórico: PENDING.
- **Proposta:**
  - ações do mercado à vista, lote padrão, uma classe por emissor (a mais negociada);
  - liquidez = volume financeiro médio dos 252 pregões anteriores, com presença ≥ 95%;
  - dois universos ponto no tempo: **U1** (postos 1–30) e **U2** (postos 31–60), estudados separadamente, sem seleção entre eles;
  - período pós-Plano Real (candidato: 2000 até o último ano completo).
- **Para fechar:** a auditoria DD-02 precisa confirmar, usando só presença e volume (nunca retornos), que há ≥ 60 emissores elegíveis em todas as datas, os códigos do layout e a presença de ativos deslistados. Se faltar, encurta-se o período; U2 não é reduzido.
- Detalhes em METHODOLOGY_CONVENTIONS.md §6.6.
- **Auditoria (2026-11-06, docs/COTAHIST_AUDIT.md; só presença e volume, sem preços):**
  - ≥ 60 emissores elegíveis em **todos** os 312 fins de mês de 2000-01 a 2025-12 (mínimo de 68, em 2002-07; mediana de 186). **U1/U2 são viáveis**, com folga pequena em 2001–2003.
  - Recomendações para aprovação:
    - BDI 02 **+ regimes especiais**: só 02 tira as ações em recuperação judicial do universo; 123 ISINs passam pelos dois;
    - exigir **negócio em t** para pertencer ao universo: a brecha é que um instrumento pode ser membro depois do último negócio;
    - período de formações de 2000-01 a 2025-12.
  - **Status inalterado: IN PROGRESS**, aguardando aprovação.
- **Contexto e problema:** "Universo pequeno de ações líquidas". Liquidez também é variável de análise da pergunta.
- **Alternativas consideradas:** volume financeiro médio; presença em pregões; participação em índice (risco de viés de seleção); universo point-in-time revisado periodicamente.

#### DD-02 — Auditoria do COTAHIST (viés de sobrevivência)
- **Status:** **IN PROGRESS (auditoria de elegibilidade concluída em 2026-11-06; falta a parte de eventos).** Histórico: PENDING.
- **Achados** (docs/COTAHIST_AUDIT.md):
  - os arquivos históricos incluem instrumentos hoje inexistentes (159 de 259 membros de U1/U2 pararam de negociar antes do fim da base), então o universo é reconstruível ponto no tempo;
  - o COTAHIST **não distingue** deslistagem de reorganização (ex.: AMBV4 → ABEV3), o que exige DD-03/DA-10;
  - 14 ISINs têm mais de um ticker e 7 tickers têm mais de um ISIN;
  - o arquivo de 2025 declara no trailer só as cotações, contrariando o layout; a convenção é aceita e registrada;
  - 89 registros de ações têm ESPECI malformado (parênteses ou aspas) e ficam excluídos pela regra de token, pendente de decisão.
- Bruto em `data/raw/cotahist/` (596 MB, 1999–2025); manifesto `data/manifests/cotahist.json`.
- **Contexto e problema:** Confirmar se ativos deslistados estão presentes e como reconstruir o universo histórico.

#### DD-03 — Proventos e desdobramentos
- **Status:** PENDING
- **Nota (2026-11-06):** a auditoria mostrou que muitos membros do universo deixam de negociar por reorganizações que o COTAHIST não identifica. A fonte de eventos também precisa resolver o valor terminal de DE-01.d (incorporações, trocas de classe e OPAs).
- **Contexto e problema:** COTAHIST traz cotações brutas. Ajuste não pode introduzir look-ahead.
- **Alternativas consideradas:** fator de ajuste a partir de eventos corporativos da B3; fonte externa de preços ajustados (com auditoria); retorno total calculado evento a evento.

#### DD-04 — Alinhamento temporal preços × CDI
- **Status:** IN PROGRESS (parte do CDI). Proposta em METHODOLOGY_CONVENTIONS.md §1.3: a taxa datada em d remunera até o próximo dia útil, e o intervalo de pregões (t₋₁, t] compõe o CDI das datas [t₋₁, t), com erro explícito se houver desalinhamento. Implementado e testado em `core.rates`. Timing do sinal: **DECIDED em DE-01 (2026-11-06)**: sinal no fechamento do último pregão do mês t, execução no fechamento de t+1, retorno da posição a partir de (t+1, t+2].
- **Contexto e problema:** Calendários, convenção de taxa diária do CDI, timing do sinal (fechamento t → posição t+1).

#### DD-05 — Política de retornos ausentes
- **Status:** IN PROGRESS. Proposta em METHODOLOGY_CONVENTIONS.md §2, aguardando confirmação.
- **Proposta:** o núcleo rejeita NaN/inf (implementado). A camada de dados publica um estado por (data, ativo) (`traded`, `no_trade`, `not_yet_listed`, `delisted`, `suspended`, `source_gap`, `no_prior_price`; enum implementado), sem preencher com zero, sem *forward fill* e sem descartar em silêncio. A construção das carteiras entrega a matriz T × N finita num calendário comum.
- **Sub-decisões pendentes:** retorno de deslistagem; posição mantida em dia `no_trade`; `source_gap` em ativo mantido; datas removidas para todas as configurações (afeta os blocos da CSCV).

---

## B. Decisões arquiteturais

Os registros completos (contexto, alternativas, critérios, justificativa, consequências) estão em [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md). Abaixo, apenas o resumo. `IN PROGRESS` = proposta feita em 2026-11-06, aguardando aprovação do desenvolvedor.

| ID | Tema | Proposta | Status |
|---|---|---|---|
| DA-01 | Organização | Monólito modular em camadas (pacote Python único + CLI) | IN PROGRESS |
| DA-02 | Linguagem | Python no caminho de execução; R só para validação | IN PROGRESS |
| DA-03 | Ledger de tentativas | JSONL append-only encadeado por hash, versionado; único escritor | IN PROGRESS |
| DA-04 | Versionamento | Manifestos SHA-256 + configs TOML versionadas + pré-registro; DVC por gatilho | IN PROGRESS |
| DA-05 | Armazenamento | Parquet + Polars (dados), NumPy (núcleo) | IN PROGRESS |
| DA-06 | Testes | pytest + hypothesis; golden files do R; testes de vazamento | IN PROGRESS |
| DA-07 | Desempenho | NumPy + ProcessPool + SeedSequence; Numba excluído nesta fase. Orçamento provisório definido (2026-11-06) | IN PROGRESS |
| DA-09 | Ingestão COTAHIST | Parser próprio; `rb3` como checagem cruzada | IN PROGRESS |
| DA-10 | Eventos corporativos | Tabela de eventos + retorno total forward. **Fonte: PENDING (DD-03)** | IN PROGRESS |
| DA-11 | Ambiente | uv + renv (pixi se pacotes R estiverem no conda-forge) | IN PROGRESS |
| DA-12 | Frameworks de backtesting | Não adotar; motor vetorizado próprio | IN PROGRESS |
| DA-13 | Relatórios | Figuras/tabelas por código; ferramenta de documento PENDING | IN PROGRESS |
| DA-14 | CDI | Verificar `python-bcb`; alternativa: API SGS direta | IN PROGRESS |

### DA-08 — CLAUDE.md privado e fora do controle de versão
- **Status:** DECIDED (2026-11-06)
- **Contexto e problema:** O usuário quer usar `CLAUDE.md` como instrução privada para agentes, sem publicá-lo no GitHub.
- **Alternativas consideradas:** (a) versionar `CLAUDE.md`; (b) ignorá-lo via `.gitignore`; (c) usar só `CLAUDE.local.md`.
- **Justificativa:** Requisito explícito do usuário. (b) mantém o nome que o Claude Code carrega automaticamente.
- **Consequências:** Um clone do repositório não terá o `CLAUDE.md`. Por isso, decisões duráveis ficam neste arquivo versionado. `.claude/settings.local.json` também é ignorado; `.claude/settings.json`, `.claude/agents/`, `.claude/commands/` e `.claude/skills/` continuam versionáveis.

---

## Referências
- Bailey, Borwein, López de Prado e Zhu. *The Probability of Backtest Overfitting*. SSRN 2326253.
- Bailey e López de Prado. *The Deflated Sharpe Ratio*. JPM 40(5), 2014. SSRN 2460551.
- Bailey e López de Prado. *The Sharpe Ratio Efficient Frontier*. 2012.
- López de Prado. *Backtesting*. SSRN 2606462.
- Pacote `pbo` (CRAN), pacote `rb3` (rOpenSci), `python-bcb` (PyPI).
