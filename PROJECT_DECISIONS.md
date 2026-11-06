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
- **Status:** PENDING
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
  - Análise em METHODOLOGY_CONVENTIONS.md §5.6.
- **Contexto e problema:** Formação e manutenção sobrepostas geram autocorrelação nos retornos e dependência entre blocos CSCV adjacentes.
- **Alternativas consideradas:** embargo/purga entre blocos; blocos longos em relação à janela máxima de formação; correção de autocorrelação no PSR; manutenção sem sobreposição.
- **Justificativa:** —
- **Consequências:** Afeta a validade de PSR/DSR e da PBO.

### DE-07 — Simulação sem sinal e critério de taxa nominal
- **Status:** PENDING
- **Achado descritivo (2026-11-06, `run-de53b361…`):**
  - Sob a nula global, com tentativas **independentes**, M = 100 e T = 1250, a regra DSR > 0,95 rejeitou em só 0,06% das réplicas (V entre tentativas) a 0,13% (V_H0).
  - Isso é coerente com SR₀ = E[máx], que não é o quantil de 95% do máximo [inferência].
  - **O DSR com limiar 0,95 não é um teste de nível 5%.** O critério de sucesso 2 ("próxima do nível nominal") precisa ser reformulado antes de qualquer simulação de validação: o que exatamente o DSR afirma controlar.
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
- **Justificativa:** — (fixar antes de ver resultados)

### DE-10 — Revisão de literatura
- **Status:** PENDING
- **Contexto e problema:** Se já existir aplicação sistemática de DSR/PBO à B3, o plano é migrar para a Frente A.
- **Nota (2026-11-06):** segundo o desenvolvedor, a revisão preliminar concluiu que a metodologia DSR/PBO é aplicável ao problema. A pergunta específica (se já existe aplicação sistemática à B3) continua sem registro de resposta.
- **Consequências:** Bloqueia investimento grande em implementação.

### Decisões de dados (estatísticas)

#### DD-01 — Universo e filtro de liquidez
- **Status:** PENDING
- **Contexto e problema:** "Universo pequeno de ações líquidas". Liquidez também é variável de análise da pergunta.
- **Alternativas consideradas:** volume financeiro médio; presença em pregões; participação em índice (risco de viés de seleção); universo point-in-time revisado periodicamente.

#### DD-02 — Auditoria do COTAHIST (viés de sobrevivência)
- **Status:** PENDING
- **Contexto e problema:** Confirmar se ativos deslistados estão presentes e como reconstruir o universo histórico.

#### DD-03 — Proventos e desdobramentos
- **Status:** PENDING
- **Contexto e problema:** COTAHIST traz cotações brutas. Ajuste não pode introduzir look-ahead.
- **Alternativas consideradas:** fator de ajuste a partir de eventos corporativos da B3; fonte externa de preços ajustados (com auditoria); retorno total calculado evento a evento.

#### DD-04 — Alinhamento temporal preços × CDI
- **Status:** IN PROGRESS (parte do CDI). Proposta em METHODOLOGY_CONVENTIONS.md §1.3: a taxa datada em d remunera até o próximo dia útil, e o intervalo de pregões (t₋₁, t] compõe o CDI das datas [t₋₁, t), com erro explícito se houver desalinhamento. Implementado e testado em `core.rates`. Timing do sinal (fechamento t → posição t+1) segue PENDING (DE-01).
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
