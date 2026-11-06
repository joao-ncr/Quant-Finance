# Convenções metodológicas — Frente E

Especificação formal das convenções estatísticas e da política de ausências. Data: 2026-11-06.

**Status por seção (atualizado em 2026-11-06, segunda sessão):**
- **§1 (DE-02, DE-02.a, DE-03.a, DE-04.c, DE-08): `DECIDED`.** O desenvolvedor delegou o fechamento dessas decisões ao agente, com base em `docs/DECISION_REVIEW_2026-11-06.md`. O registro formal está em `PROJECT_DECISIONS.md`.
- §2 (DD-05) e a parte CDI de §1.3 (DD-04): **proposta**, inalteradas. Não bloqueiam a Etapa A.
- §4 (N efetivo e V): protocolo `DECIDED`; o estimador principal e a escolha definitiva de V continuam **abertos**.
- §5 (CSCV/PBO, DE-05, S e DE-06): convenções `DECIDED`, **não implementadas**; o valor de S é fixado por regra no pré-registro; o embargo está em aberto, com critério.

O código em `src/frente_e/core/metrics.py`, `core/rates.py` e `core/sharpe_inference.py` implementa **exatamente** a §1. Qualquer mudança deve alterar este documento, o código e os testes juntos.

Categorias de estado usadas abaixo: *implementado* · *testado* · *validado matematicamente* · *validado por referência independente* · *pendente de decisão* · *pendente de dados*.

---

## 1. DE-02 — Sharpe, PSR e DSR

**DE-02 DECIDED (2026-11-06)** como especificado em §1.1–1.7, com duas ressalvas registradas como dependências explícitas, não resolvidas aqui:
1. **Excesso em carteira long-short.** "Excesso ao CDI" está bem definido para carteira comprada financiada por caixa. Numa long-short autofinanciada, o retorno já é um excesso, e subtrair o CDI seria errado. A definição do retorno da estratégia pertence a **DE-01**. DE-02 fixa o estimador e as unidades.
2. **Autocorrelação.** A variância assintótica usada no PSR supõe observações i.i.d. O artigo de 2012 afirma que ela também vale para retornos estacionários e ergódicos, citando Opdyke (2007), mas isso não foi conferido. **Nenhuma correção de autocorrelação é implementada.** O efeito é tratado em DE-06/DE-07: precisa ser medido, não suposto.

Justificativa, alternativas descartadas e fontes: DECISION_REVIEW_2026-11-06.md §2. Uma inconsistência conhecida e aceita: ŜR usa ddof = 1 e os momentos usam m₂ plug-in. A diferença é um fator √((T−1)/T), de ordem 1/T, irrelevante no T do estudo, e não afeta a garantia de radicando ≥ 0.

### 1.1 Frequência
- Retornos **diários** (pregões da B3). Todo cálculo estatístico (Sharpe, momentos, PSR, DSR, variância entre tentativas) é feito na frequência diária, **sem anualização prévia**.
- Convenção anual: **252** pregões, usada apenas para apresentar valores anualizados e para converter taxas anuais.
- Observação: o exemplo numérico de Bailey e López de Prado (2014) usa **250** observações por ano. Os testes que reproduzem o artigo usam 250, como no artigo; isso não muda a convenção do projeto.

### 1.2 Tipo de retorno
- **Retorno simples aritmético** de retorno total: `r_t = (P_t + D_t)/P_{t−1} − 1`, com eventos corporativos incorporados na data do evento (forma exata em DD-03, ainda pendente).
- Justificativa:
  1. **Agregação de carteira:** o retorno de uma carteira é linear nos retornos simples, `r_p = Σ wᵢ rᵢ`. Com log-retornos isso não vale.
  2. **CDI:** é uma taxa simples por dia útil, composta multiplicativamente; o excesso `r − r_f` fica na mesma unidade.
  3. **Métricas:** o PSR usa a média e os momentos da série que de fato compõe o PnL.
- Log-retornos não são usados em nenhum cálculo. Se algum dia forem necessários, a transformação `log(1+r)` precisa ser explícita e documentada.

### 1.3 Taxa livre de risco (CDI)
- Fonte: SGS/BCB, série 12 (CDI diário). Segundo fontes secundárias, a série é publicada em **% ao dia**; confirmar no portal do BCB na ingestão (DA-14). Conversão: `r_{f,d} = valor / 100`.
- Se for usada uma série anual (base 252): `r_{f,d} = (1 + r_{f,a})^{1/252} − 1`.
- **Alinhamento:** a taxa datada em *d* remunera de *d* até o próximo dia útil do CDI. O retorno do intervalo de pregões (t₋₁, t] é comparado com `r_f = Π_{d ∈ [t₋₁, t)} (1 + r_d) − 1`. Assim:
  - num fim de semana, a taxa de sexta remunera até segunda;
  - num feriado local da B3 em que o CDI corre, o intervalo acumula dois dias de CDI.
- Cada pregão que inicia um intervalo precisa ter taxa do CDI na mesma data, e deve existir uma data do CDI ≥ fim do intervalo. Caso contrário o erro é explícito (`DataValidationError`).
- Valores históricos são sempre os da própria data: nunca se usa a taxa atual para o passado, nem se preenche lacuna com zero ou com a taxa anterior.
- *Implementado e testado* (`core.rates`). Calendários reais ainda **pendentes de dados**.

### 1.4 Sharpe Ratio
| Item | Definição |
|---|---|
| Retorno excedente | `e_t = r_t − r_{f,t}` (diferença aritmética) |
| Média | `μ̂ = (1/T) Σ e_t` |
| Desvio-padrão | `σ̂ = sqrt( Σ (e_t − μ̂)² / (T − 1) )` (**ddof = 1**) |
| Sharpe diário | `ŜR = μ̂ / σ̂` |
| Sharpe anualizado | `ŜR_a = ŜR · √252`, **só para apresentação**. A relação √252 é exata apenas para retornos i.i.d.; sob autocorrelação, não (ver DE-06) |
| Série insuficiente | `T < 2` → `DegenerateSeriesError` |
| Variância zero | todos os valores idênticos (`ptp = 0`, teste exato, sem tolerância) → `DegenerateSeriesError`; nunca `±inf` ou `NaN` |
| Valores não finitos | rejeitados (`ContractViolationError`); ver §2 |

*Implementado e testado* (`core.metrics`): valores conhecidos, invariância de escala, antissimetria, consistência diário/anualizado e CDI variável. *Validado matematicamente* contra cálculos à mão. Convenção *confirmada* (DE-02 DECIDED, 2026-11-06).

### 1.5 Assimetria e curtose
- Momentos centrais amostrais `m_k = (1/T) Σ (e_t − μ̂)^k`.
- **Assimetria** `γ̂₃ = g₁ = m₃ / m₂^{3/2}`.
- **Curtose de Pearson** `γ̂₄ = b₂ = m₄ / m₂²` (normal = 3). **Não** é a curtose excedente.
- **Sem correção de viés** (estimadores *plug-in*). Justificativa: a variância assintótica do Sharpe, que dá origem ao PSR, é escrita em termos dos momentos populacionais, e os estimadores plug-in são consistentes para eles. As versões com correção de viés (G₁ e curtose ajustada) são alternativa possível; a diferença é O(1/T).
- Equivale a `scipy.stats.skew(bias=True)` e `scipy.stats.kurtosis(fisher=False, bias=True)`. Atenção: o padrão do SciPy é `fisher=True`, que dá excesso.
- Mínimo matemático: `T ≥ 2` e série não constante. ~~Pendente (DE-02.a)~~ **DE-02.a decidida (2026-11-06): não há tamanho mínimo de amostra.** Ver §1.6.1.
- *Implementado e testado*: casos à mão, comparação com o SciPy em dados gerados por propriedade e desigualdade de Pearson.

### 1.6 PSR
```
PSR(SR*) = Φ( (ŜR − SR*) · √(T − 1) / √(1 − γ̂₃·ŜR + ((γ̂₄ − 1)/4)·ŜR²) )
```
- ŜR, SR*, γ̂₃ e γ̂₄ vêm da **mesma série diária** de retornos excedentes, e T é o número de observações dessa série.
- **Denominador.** Como `b₂ ≥ g₁² + 1` (desigualdade de Pearson), `1 − γ̂₃ŜR + ((γ̂₄−1)/4)ŜR² ≥ (1 − γ̂₃ŜR/2)² ≥ 0`. A igualdade só ocorre numa distribuição de dois pontos com `ŜR = 2/γ̂₃`. A implementação deve gerar `DegenerateSeriesError` se o radicando for ≤ 0, nunca dividir por zero.
- A forma (√(T−1), curtose de Pearson no termo (γ₄−1)/4) foi confirmada indiretamente pela reprodução do exemplo do artigo (§1.8), e depois visualmente nas eq. 11/41 de 2012 e na eq. 2 de 2014.
- **Momentos impossíveis:** γ₄ < 1 + γ₃² viola a desigualdade de Pearson e gera `ContractViolationError`, cuja mensagem lembra que a curtose é a bruta. Esse é o erro típico de passar o excesso de curtose.
- *Implementado e testado* em `core.sharpe_inference.probabilistic_sharpe_ratio` (2026-11-06). Aceita escalares ou arrays.

#### 1.6.1 DE-02.a — domínio, precisão, poder e MinTRL (DECIDED)
São quatro conceitos distintos, e só o primeiro é imposto pelo código:

| Conceito | Pergunta | Tratamento |
|---|---|---|
| Domínio matemático | a fórmula está definida? | T > 1, entradas finitas, γ₄ ≥ 1 + γ₃², radicando > 0. Fora disso, erro explícito |
| Precisão da aproximação | a normal assintótica descreve bem ŜR com este T e estas caudas? | questão empírica: simulação de cobertura do PSR sob H₀ (DE-07). **Nenhum limiar de T é imposto** |
| Poder | o histórico basta para rejeitar SR ≤ SR*? | é o que o PSR e o MinTRL respondem, *supondo* a aproximação válida |
| MinTRL | quantas observações bastariam? | **informação reportada**, nunca bloqueio |

- **T real:** o PSR aceita T real > 1, porque o artigo de 2012 avalia o PSR em T = 59,895. Dados reais sempre têm T inteiro ≥ 2.
- **Alternativas descartadas:**
  - limiar fixo (30, 252, …): a regra "T > 30" citada em 2012 (p. 11) é de livro-texto, não considera caudas pesadas e não tem base para retornos diários;
  - limiar derivado agora por simulação: depende do T e das caudas do estudo, que ainda não são conhecidos.

#### 1.6.2 MinTRL (DECIDED, eq. 13 de 2012)
```
MinTRL = 1 + (1 − γ̂₃·ŜR + ((γ̂₄ − 1)/4)·ŜR²) · (Φ⁻¹(c) / (ŜR − SR*))²
```
- c = nível de confiança, com 0,5 < c < 1. A unidade é o **número de observações** na frequência de ŜR: pregões, no projeto.
- Com ŜR ≤ SR*, o PSR fica ≤ 0,5 < c para todo T, o conjunto é vazio e o resultado é `+inf` (ínfimo do vazio, documentado). Não é erro, porque é um valor matematicamente determinado.
- Identidade testada: `PSR(SR*; T = MinTRL) = c`.
- *Implementado e testado* em `core.sharpe_inference.min_track_record_length`.

### 1.7 DSR
```
SR₀ = √V · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ],   γ ≈ 0,5772156649 (Euler–Mascheroni)
DSR = PSR(SR₀)
```
- `V` é a variância, **entre tentativas**, dos ŜR diários. O escopo de N (DE-04.a) e a política provisória para V (DE-04.b) estão em §4.
- `N` é o número efetivo de tentativas (DE-03). **DE-03.a (DECIDED):** N é real e entra na fórmula sem arredondamento. **DE-04.c (DECIDED):** N ≥ 2, e N < 2 gera `ContractViolationError`. Fundamento: o fator `(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))` é negativo para N < 1,2836 e vale −∞ em N = 1, e o "máximo entre tentativas" exige ao menos duas. Se um estimador produzir N efetivo < 2, isso é um resultado a reportar (a família é, na prática, uma tentativa), não um valor a corrigir.
- **Três N distintos, sem arredondamento silencioso:**
  1. N efetivo contínuo, saída do estimador;
  2. N usado na fórmula, que é o mesmo valor sem arredondar;
  3. `floor(N)`, usado **apenas** em teste, para comparar com o `poolr`.
- V = 0 é aceito e dá SR₀ = 0, ou seja, DSR = PSR(0). V < 0 é erro.
- Todas as entradas ficam em unidades diárias. Se V vier anualizado, divide-se pelo número de períodos por ano: 250 no exemplo do artigo e 252 no projeto. O mesmo vale para ŜR, com √períodos.
- *Implementado e testado* em `core.sharpe_inference`, com as funções `expected_max_factor`, `deflated_benchmark` e `deflated_sharpe_ratio`. N e V são argumentos nomeados obrigatórios, para que quem chama os registre.
- O DSR foi validado contra as três âncoras publicadas (§1.8), e as propriedades contra testes sintéticos (§1.9).
- **Precisão da aproximação de E[máx]** (cálculo exato por integração, `core.null_models.expected_max_iid_normal`): o erro relativo é −7,9% em N = 2, +2,3% em N = 10, +0,9% em N = 100 e +0,4% em N = 1000. Mantém-se a aproximação do artigo (fidelidade à fonte e às âncoras). O erro entra na avaliação de §4.

### 1.8 Referência para o DSR (DE-08)
Fonte: Bailey e López de Prado, *The Deflated Sharpe Ratio*, preprint em davidhbailey.com/dhbpapers/deflated-sharpe.pdf, seção "A numerical example". O texto foi extraído do PDF em 2026-11-06. Os símbolos matemáticos ficaram ilegíveis (fonte sem mapeamento Unicode), mas o texto corrido não.

| Item | Valor | Origem |
|---|---|---|
| Sharpe anualizado | 2,5 | texto legível |
| Amostra | diária, 5 anos, T = 1250, "250 observations per year" | texto legível |
| N, V, γ₃, γ₄ | N = 100, V_a = 1/2, γ₃ = −3, γ₄ = 10 | **ilegíveis na extração**; valores conhecidos da literatura, confirmados indiretamente abaixo |
| Âncora 1 | DSR ≈ 0,90 ("only a 90% chance") | texto legível |
| Âncora 2 | N = 46 → DSR = **0,9505** | texto legível |
| Âncora 3 | retornos normais (γ₃ = 0, γ₄ = 3) → DSR = 0,95 com N = 88 | texto legível |

Cálculo independente (SciPy, ambiente efêmero, fora do projeto) com os valores acima e V diário = 0,5/250:
- N = 100: SR₀ = 0,1132 (diário), DSR = **0,9004**;
- N = 46: DSR = **0,9505**;
- normal, N = 88: DSR = **0,9505**, e N = 89 dá 0,9498, ou seja, 88 é o último N com DSR ≥ 0,95.

As três âncoras batem em 4 casas decimais. Isso confirma ao mesmo tempo os parâmetros e a forma das fórmulas §1.6–1.7.

**Conferência visual (2026-11-06):** as páginas 9–10 foram renderizadas como imagem e os parâmetros da tabela acima estão confirmados. Detalhe: V = 1/2 é dado em unidades anualizadas e o próprio artigo o converte para `1/(2·250)`. O artigo de 2012 traz quatro âncoras adicionais do PSR, também reproduzidas. A lista completa (A1–A7), com páginas e SHA-256 dos PDFs, está em [DECISION_REVIEW_2026-11-06.md](DECISION_REVIEW_2026-11-06.md) §0.1. O congelamento como fixture aguarda a aprovação de DE-08.

**DE-08 (DECIDED, 2026-11-06):** as sete âncoras A1–A7 foram congeladas em `tests/fixtures/published/sharpe_anchors.json`, com os valores **como impressos** (texto), página, URL e SHA-256 dos PDFs. Elas são testadas em `tests/reference/test_published_anchors.py`.
- **Tolerância.** A proposta de usar só a meia unidade da última casa impressa mostrou-se **incompleta**: A6 (PSR = 0,95351 contra 0,953) a excede em 1e-5. A causa é que A4–A6 usam entradas que o artigo imprimiu arredondadas (ŜR = 0,458, γ₃ = −2,448, γ₄ = 10,164). A tolerância adotada é a meia unidade da saída somada à propagação de primeira ordem da meia unidade de cada entrada impressa arredondada, calculada no teste. As entradas hipotéticas (A1–A3) e as usadas literalmente no código do artigo (A7) são tratadas como exatas.
- O pacote `pbo` **não** é referência para o DSR.

### 1.9 Testes exigidos antes de declarar PSR/DSR validados
Estado em 2026-11-06: itens 1–4 *implementados e passando*; item 5 *pendente* (DE-07).
1. Reprodução das sete âncoras A1–A7, com a tolerância acima.
2. Sob normalidade (γ₃ = 0, γ₄ = 3), o PSR se reduz a `Φ((ŜR − SR*)·√(T−1)/√(1 + ŜR²/2))`.
3. Monotonia: PSR decrescente em SR*, afastando-se de 1/2 conforme T cresce, e com PSR(SR*) = 1/2; antissimetria `PSR(ŜR, SR*, γ₃) + PSR(−ŜR, −SR*, −γ₃) = 1`; SR₀ crescente em N e em V.
   - **Correção:** a versão anterior deste item dizia "PSR crescente em ŜR", o que é **falso** em geral. Sob normalidade, z(ŜR) decresce quando 1 + SR*·ŜR/2 < 0 (por exemplo SR* = 1 e ŜR ≈ −3). A monotonia em ŜR só é testada para SR* ≥ 0 sob normalidade.
4. Fronteiras com erro explícito: radicando nulo, momentos impossíveis, T ≤ 1, N < 2, V < 0, entradas não finitas ou booleanas, confiança fora de (0,5; 1).
5. Calibração por simulação: sob H₀ (SR verdadeiro = SR*), PSR ~ U(0,1) aproximadamente para T grande, inclusive com caudas pesadas. Fica para o desenho de DE-07 (semente fixa e tolerância de Monte Carlo declaradas antes de rodar).

---

## 2. Política de retornos ausentes (DD-05, proposta)

**Princípio:** o núcleo matemático só recebe séries finitas (`NaN`/`inf` são rejeitados, *implementado e testado*). Toda ausência é tratada **antes** do núcleo, com motivo explícito. Nada é preenchido com zero, nada recebe *forward fill* sem decisão, e nenhuma linha ou ativo é descartado em silêncio.

### 2.1 Estados por (data, ativo) — `frente_e.data.observation_status.ObservationStatus`
| Estado | Situação | Retorno no dia |
|---|---|---|
| `traded` | houve negócio e há preço anterior válido | definido |
| `no_trade` | listado, pregão aberto, sem negócio | ausente (motivo registrado) |
| `not_yet_listed` | antes da primeira negociação | ausente; fora do universo |
| `delisted` | depois da saída definitiva | ausente; fora do universo |
| `suspended` | negociação suspensa, se identificável na fonte | ausente |
| `source_gap` | o dado deveria existir e falta (coleta ou fonte) | ausente; **falha de qualidade** |
| `no_prior_price` | há preço hoje, mas não há preço anterior válido (1º dia, ou depois de uma lacuna) | ausente |

"Período sem observação válida": o calendário do estudo é o de pregões da B3 derivado dos próprios dados. Um pregão sem nenhum dado para nenhum ativo é `source_gap` no nível do mercado e aparece no relatório de qualidade.

### 2.2 Comunicação entre camadas
- A camada de dados publica uma tabela longa `(date, asset_id, status, price_raw, total_return)`, com `total_return` preenchido só quando `status = traded`, e um relatório de qualidade com contagens por estado.
- A construção das carteiras (DE-01) decide o efeito de cada estado sobre as posições e produz, para cada configuração, uma série de retornos **definida em todos os pregões** do calendário comum. Estar fora do mercado é um valor legítimo (retorno = r_f, excesso = 0), não um preenchimento.
- O núcleo recebe a matriz T × N (`ReturnsMatrix`), finita e com o mesmo calendário para todas as configurações.

### 2.3 Impacto em T × N e na CSCV
- Todas as N configurações compartilham as mesmas T linhas. Remover uma data para algumas configurações quebra o alinhamento IS/OOS e é proibido.
- Remover uma data para **todas** as configurações desloca as fronteiras dos blocos da CSCV. Só é aceitável com motivo registrado (ex.: `source_gap` de mercado) e decisão explícita (**pendente**).
- O pacote de referência `pbo` exige T divisível por S. A regra para T não divisível (truncar início/fim, ou blocos desiguais) é parte de DE-05 (**pendente**).

### 2.4 Pendentes de decisão (afetam resultados)
- **Retorno de deslistagem:** o último preço negociado subestima perdas em falências. A regra é **pendente** (DE-01/DD-02), e essa é uma fonte conhecida de viés de sobrevivência.
- **Posição mantida num dia `no_trade`:** (a) valorizar pelo último preço (equivale a um *forward fill* de preço, com retorno 0 no dia), (b) excluir o ativo da carteira, ou (c) outra regra. **Pendente** (DE-01).
- `source_gap` em ativo mantido: interromper o estudo (recomendado) ou excluir o ativo com registro. **Pendente.**

---

## 3. Convenções das referências independentes (evidência documental)

Lidas no código-fonte dos pacotes instalados em `validation/r/` (2026-11-06). Elas determinam como comparar as implementações próprias; não são decisões do projeto.

### 3.1 `pbo` 1.3.5 (referência de CSCV/PBO, relevante para DE-05)
- `f` recebe o bloco (linhas × configurações) e devolve um vetor de N métricas.
- Blocos de tamanho `T/S` sem tratamento de resto: **T precisa ser divisível por S**.
- `n* = which.max(R)`: empate resolvido pelo **primeiro** índice.
- `ω̄ = rank(R̄)[n*] / N`, com empates pela **média**. A melhor configuração OOS dá `ω̄ = 1` e `λ = +∞`, que vira `inf_sub = 6`. **O artigo define `rank/(N+1)`** (p. 12, conferido visualmente em 2026-11-06), então o pacote diverge do original. **O projeto adota `rank/(N+1)`** (DE-05.a DECIDED, §5.1).
- `PBO = #{λ ≤ 0} / C(S, S/2)`.
- Os campos `slope` e `intercept` do objeto de retorno saem **trocados** e arredondados (`signif(…, 5)`); use `rn_pairs`.

### 3.2 `poolr` 1.2-0 (referência de N efetivo por autovalores, relevante para DE-03)
- Métodos: `nyholt`, `liji`, `gao` (C = 0,995 por padrão), `galwey`, `chen`.
- **Todos os resultados passam por `floor()`.** O projeto **não** arredonda (DE-03.a DECIDED). A comparação é `floor(nosso) == poolr` mais autovalores com rtol 1e-9 (`tests/reference/test_effective_trials_reference.py`, passando).
- **Li-Ji é inteiro por construção** (verificado em 2026-11-06): como o traço é M, `Σ f(λ) = M − Σ_{λ≥1}(⌊λ⌋ − 1)`. Somar as frações dá 6,9999999999998 em vez de 7, e o `floor` daria 6. O `√eps` do `poolr` mascara esse ruído; o projeto usa a identidade exata. Gao também é uma contagem inteira. O `floor` só faz diferença para Nyholt, Galwey e A.3.
- `liji` soma `√eps` aos autovalores absolutos; `galwey` zera autovalores negativos.
- Observação sobre a referência, não resultado do projeto: no caso sintético de 4 blocos de 5 séries com correlação ≈ 0,96, os métodos discordam muito (Nyholt 16, Gao 17, Li e Ji 7, Galwey 7, contra 4 grupos verdadeiros). A escolha em DE-03 pesa no DSR.

---

## 4. N efetivo, contagem de tentativas e V (DE-03, DE-04)

### 4.1 DE-04.a — o que conta como tentativa (DECIDED, 2026-11-06)
- **Três identidades, que não se confundem:**
  - `run_id` = uma execução computacional;
  - fingerprint = configuração científica completa, inclusive sementes;
  - **tentativa** = configuração distinta da família avaliada nos dados do estudo.
- N efetivo é uma **estimativa** (§4.2), não uma contagem.
- Regras para M, o número de tentativas:

| Caso | Conta em M? | Entra em V? |
|---|---|---|
| Concluída | sim | sim |
| Falhou ou foi abortada e depois concluiu com a mesma configuração | uma vez | sim (a concluída) |
| Falhou ou foi abortada e nunca concluiu | **sim**, reportada à parte | não (não há ŜR) |
| Repetição idêntica | uma vez; resultados devem ser bit a bit iguais, e divergência é erro de integridade | sim |
| Sementes diferentes, regra determinística | uma vez | sim |
| Sementes diferentes, procedimento estocástico | cada semente | sim |
| Exploração com dados reais antes do pré-registro | **sim** | sim, se houver ŜR |
| Execuções sintéticas (`RunKind.SYNTHETIC`/`DEVELOPMENT` sem dados reais) | não | não |

- Excluir falhas e abortos que nunca concluíram abriria a brecha de abortar o que parece ruim. Por isso eles contam.
- Execuções com dados reais antes do pré-registro ficam **proibidas**, de modo que a linha "exploração" é vazia por construção. Em 2026-11-06 nenhum dado real foi lido.
- **Aberto (fixar no pré-registro):** se a seleção é feita dentro de cada faixa de liquidez, M é por faixa; se se reporta a melhor entre faixas, M é o total.

### 4.2 DE-03 — protocolo de N efetivo (protocolo DECIDED; estimador principal ABERTO)
1. **Candidatos pré-registrados** (`core.effective_trials.ESTIMATORS`): `raw` (M, cota superior), `nyholt`, `liji`, `gao` (C = 0,995), `galwey` e `average_correlation` (Bailey e López de Prado 2014, A.3). O ONC/clustering fica fora: a fonte original não foi conferida, ele introduz aleatoriedade e parâmetros, e só entra se todos os candidatos falharem no critério abaixo.
2. **Reporte:** o DSR da melhor configuração é reportado para **todos** os candidatos, como análise de sensibilidade.
3. **Estimador principal**, escolhido por simulação **antes de qualquer dado real**, com este critério (fixado em 2026-11-06, antes de executar qualquer simulação):
   - **Estimando.** Sob a nula global, ŜR ≈ N(0, C/(T−1)). Para o DSR estar correto com a variância marginal certa, o fator precisa ser `f(N*) = E[máxᵢ Zᵢ]`, com Z ~ N(0, C). N* é o "N implícito".
   - **Erro do estimador k na estrutura s:** `e_k(s) = f(N̂_k(C_s)) / E[máx Z]_s − 1`, o erro relativo de SR₀ quando V está correto. Inclui o erro da própria aproximação f.
   - **Regra:** principal = `argmin_k max_s |e_k(s)|` sobre o conjunto final de estruturas.
     - Empate: diferença dos piores casos menor que 2 erros-padrão de Monte Carlo.
     - Desempate: prefere-se o estimador com e ≥ 0 (conservador) em mais estruturas.
   - **Conjunto final de estruturas:** as genéricas de §4.4 **mais** as correlações induzidas pela grade de momentum aplicada a preços sintéticos sem sinal. Estas exigem DE-01 e usam só dados sintéticos.
   - **Proibido:** escolher pelo DSR, por dados reais ou pelo resultado mais favorável.
   - **Secundário (descritivo):** média e desvio de N̂ calculado sobre Ĉ (correlação estimada com T observações), para medir a sensibilidade ao ruído de estimação.
4. **Estado:** a execução preliminar (§4.4) usa só estruturas genéricas e **não escolhe** o estimador principal. Ela ordena os candidatos e mostra o tamanho dos erros.

### 4.3 DE-04.b — V (política PROVISÓRIA)
> **Superado pela emenda 1 (§4.5).** O texto abaixo é a política original, fixada antes da simulação, e fica preservado como histórico.

- **Primário provisório (original, superado):** V = variância amostral (ddof = 1) dos ŜR diários das tentativas distintas concluídas. É a leitura literal do artigo: "the variance across the trials' estimated SR".
- **Sensibilidade:** V_H0 = 1/(T−1), a variância assintótica de ŜR com SR verdadeiro = 0 (eq. 2 com ŜR = 0).
- **Problema conhecido (testado em `tests/unit/test_null_models.py`):** com tentativas correlacionadas, a variância entre tentativas encolhe. Na equicorrelação, E[s²] = (1 − ρ)·Var(ŜR). Combinar esse V com um N efetivo desconta a correlação **duas vezes**, o que é anticonservador.
- **Por que a escolha não fica decidida:** sob a nula global, V_H0 é exato por construção, então uma simulação só da nula global favorece V_H0 trivialmente e não testa o papel de V em capturar a dispersão de SR verdadeiros. A decisão exige um desenho **não nulo**, com dispersão de SR verdadeiros (entra em DE-07). Até lá, o estudo reporta o DSR com os dois V.

### 4.4 Simulação preliminar (especificação fixada antes da execução)
- Módulo: `frente_e.experiments.neff_study`. Execução registrada no ledger como `RunKind.SYNTHETIC`, `experiment_id = sim-neff-preliminary`, com semente raiz nova registrada.
- **Hipóteses de modelagem:**
  - nula global;
  - ŜR ≈ N(0, C/(T−1)), aproximação assintótica;
  - retornos normais (γ₃ = 0, γ₄ = 3) na taxa de rejeição;
  - sem autocorrelação temporal (DE-06) e sem caudas pesadas.
- **Estruturas genéricas, todas com M = 100.** São hipóteses de trabalho, não a grade real:
  - independente;
  - equicorrelação com ρ ∈ {0,5; 0,9};
  - blocos 10 × 10 com ρ = 0,9 e 4 × 25 com ρ = 0,95;
  - AR(1) com ρ ∈ {0,9; 0,99}, uma grade 1-D de janelas vizinhas;
  - grade separável 10 × 10 com (ρ_f, ρ_h) ∈ {(0,9; 0,9), (0,97; 0,8)}, um proxy de formação × manutenção.
- **Parâmetros:** 200.000 réplicas por estrutura, em blocos de 20.000; T = 1250, o comprimento do exemplo do DSR; 20 sorteios de Ĉ por estrutura.
- **Saídas por estrutura:**
  - E[máx Z] com erro-padrão de Monte Carlo;
  - N* implícito;
  - E[s²(Z)], que mede o encolhimento de V;
  - para cada candidato: N̂(C), e_k(s), N̂(Ĉ) com média e desvio, o erro conjunto `E[√V̂_cross]·f(N̂) / E[máx Z] − 1`, e a fração de réplicas com DSR > 0,95 usando V̂_cross e V_H0.
  - Essa fração é **descritiva para DE-07**. O limiar 0,95 é o do exemplo do artigo, não uma decisão de DE-09.
- Também é saída a tabela determinística do erro da aproximação f(N) contra o E[máx] exato.

### 4.5 Resultados preliminares e emenda 1 (2026-11-06)
**Execução:** `run-de53b36114814d0398437a14fd8d100f`, `RunKind.SYNTHETIC`, primeira entrada do ledger do projeto. Artefato JSON com SHA-256 `608be6f5fe181768e8a29f7af4d4118d858231f92f897b777317c9303c93a9da`, em `experiments/artifacts/`, fora do Git. Os números abaixo foram copiados desse artefato. Erro-padrão de Monte Carlo de E[máx Z] ≤ 0,0021, ou seja, ≤ 0,3% relativo.

**Tabela 1 — erro relativo de SR₀ com V marginal correto, `e = f(N̂)/E[máx Z] − 1`** (critério original de §4.2; N̂ calculado sobre C verdadeira):

| Estrutura | E[máx Z] | N* | E[s²] | M bruto | Nyholt | Li-Ji | Gao | Galwey | A.3 |
|---|---|---|---|---|---|---|---|---|---|
| independente | 2,506 | 93,3 | 1,000 | +1,0% | +1,0% | +1,0% | +1,0% | +1,0% | +1,0% |
| equi 0,5 | 1,772 | 15,1 | 0,500 | +42,8% | +37,0% | +28,9% | +42,6% | +32,1% | +28,7% |
| equi 0,9 | 0,794 | 2,8 | 0,100 | +218,7% | +138,8% | +104,3% | +216,9% | +129,0% | +103,7% |
| blocos 10×10, 0,9 | 1,974 | 23,7 | 0,918 | +28,2% | +26,8% | −3,7% | +27,5% | +7,8% | +26,7% |
| blocos 4×25, 0,95 | 1,453 | 7,9 | 0,774 | +74,2% | +68,2% | +14,6% | +71,9% | +25,4% | +67,8% |
| AR(1) 0,9 | 2,007 | 25,6 | 0,838 | +26,1% | +24,6% | +7,0% | +24,4% | +9,4% | +23,0% |
| AR(1) 0,99 | 0,964 | 3,5 | 0,267 | +162,6% | +131,4% | +72,7% | +124,0% | +64,7% | +111,4% |
| grade 0,9 × 0,9 | 1,515 | 8,9 | 0,475 | +67,1% | +58,6% | +31,9% | +51,3% | +27,8% | +49,3% |
| grade 0,97 × 0,8 | 1,453 | 7,9 | 0,513 | +74,1% | +65,8% | +33,6% | +52,6% | +27,3% | +57,5% |
| **pior \|e\|** | | | | 218,7% | 138,8% | **104,3%** | 216,9% | 129,0% | 111,4% |

**Tabela 2 — erro relativo de SR₀ com V entre tentativas (ddof = 1), `E[s(Z)]·f(N̂)/E[máx Z] − 1`**, e fração de réplicas com DSR > 0,95:

| Estrutura | M bruto | Nyholt | Li-Ji | Gao | Galwey | A.3 |
|---|---|---|---|---|---|---|
| independente | +1% (0,06%) | +1% (0,06%) | +1% (0,06%) | +1% (0,06%) | +1% (0,06%) | +1% (0,06%) |
| equi 0,5 | +1% (1,55%) | −3% (1,95%) | −9% (2,66%) | +1% (1,56%) | −7% (2,36%) | −9% (2,68%) |
| equi 0,9 | +1% (4,22%) | −25% (6,49%) | −36% (7,74%) | −0% (4,26%) | −28% (6,84%) | −36% (7,75%) |
| blocos 10×10 | +20% (0,00%) | +19% (0,00%) | −10% (0,12%) | +19% (0,00%) | +1% (0,03%) | +19% (0,00%) |
| blocos 4×25 | +43% (0,00%) | +38% (0,01%) | −6% (0,17%) | +41% (0,00%) | +3% (0,08%) | +38% (0,01%) |
| AR(1) 0,9 | +13% (0,03%) | +12% (0,03%) | −4% (0,21%) | +12% (0,03%) | −2% (0,17%) | +10% (0,04%) |
| AR(1) 0,99 | +28% (1,41%) | +12% (2,06%) | −16% (4,04%) | +9% (2,24%) | −20% (4,44%) | +3% (2,61%) |
| grade 0,9 × 0,9 | +12% (0,69%) | +6% (0,94%) | −12% (2,34%) | +2% (1,22%) | −14% (2,68%) | +0% (1,30%) |
| grade 0,97 × 0,8 | +20% (0,44%) | +14% (0,60%) | −8% (1,81%) | +5% (0,98%) | −12% (2,23%) | +9% (0,82%) |

Com V_H0 e N efetivo, a fração de DSR > 0,95 ficou entre 0,02% e 0,75% em todas as estruturas: conservador em todas.

**Leituras.** [Fatos] são os números acima. [Inferências] estão marcadas.
1. **[Fato]** Pelo critério original, **nenhum candidato é adequado**: o menor pior caso é o de Li-Ji, com 104%. Isso não autoriza escolher o estimador principal, que continua **aberto**.
2. **[Fato]** Sob a nula global, o fator comum de tentativas correlacionadas desloca todas juntas e quase não altera o máximo centrado. Por isso N* fica pequeno (2,8 em equi 0,9) e V entre tentativas encolhe (E[s²] = 0,10 = 1 − ρ). **[Inferência]** Na equicorrelação, √V_cross·f(M) acerta E[máx] por construção (E[máx] = √(1−ρ)·E[máx de M i.i.d.]). Isso explica o +1%.
3. **[Fato]** Os pares (N efetivo, V entre tentativas) foram **anticonservadores** em várias estruturas, até −36%, e a fração de DSR > 0,95 chegou a 7,7% contra 4,2% do par (M, V entre tentativas). É o "desconto duplo" previsto em §4.3.
4. **[Fato]** O par (M bruto, V entre tentativas) ficou entre +1% e +43%: conservador nas 9 estruturas e o mais preciso no pior caso entre os pares avaliados.
5. **[Fato, descritivo para DE-07]** Mesmo com tentativas independentes, DSR > 0,95 ocorreu em só 0,06–0,13% das réplicas. **[Inferência]** SR₀ = E[máx] não é o quantil de 95% do máximo, então o DSR com limiar 0,95 não é um teste de nível 5%. O critério de sucesso 2 do projeto ("taxa próxima do nível nominal") precisa ser reformulado em DE-07.
6. **[Fato]** N̂ sobre Ĉ (T = 1250) ficou próximo de N̂ sobre C, exceto Gao (blocos 4×25: 91 → 84,3 ± 0,5) e Li-Ji (blocos 10×10: 20 → 23,9 ± 0,7), que são sensíveis ao ruído por serem contagens ou descontínuos.
7. **[Fato]** O erro da aproximação f(N) (tabela determinística) coincide com §1.7. No caso independente com M = 100, E[máx] por Monte Carlo deu 2,5062 ± 0,0010, contra 2,5076 exato.

**Emenda 1 ao protocolo (2026-11-06; registrada depois desta simulação e antes de qualquer dado real):**
- **DE-04.b (provisório):** par primário passa a ser **(N = M bruto, V = variância entre tentativas, ddof = 1)**. Sensibilidade obrigatória: cada N efetivo com V_H0. O par (N efetivo, V entre tentativas) é **descartado por evidência**, porque é anticonservador.
- **DE-03:** o critério de seleção passa a avaliar **pares (N, V)** pelo erro conjunto de SR₀ (Tabela 2) e não só N com V marginal (Tabela 1). A regra continua: menor pior |erro|, desempate pelo conservador, conjunto final de estruturas incluindo a grade de momentum. Continua **aberto** até essas estruturas existirem (DE-01).
- **Limite desta evidência:** sob a nula global, V_H0 é exato por construção. O papel de V em capturar a dispersão de SR **verdadeiros**, que é o que o artigo pretende, não foi testado. Isso exige um desenho não nulo (DE-07).
- **Candidato não adotado:** estimar E[máx] diretamente por Monte Carlo a partir de Ĉ, sem N efetivo. Seria um método novo, não encontrado na literatura conferida. Registrado como ideia, não como decisão.

---

## 5. CSCV e PBO (DE-05, S, DE-06): convenções decididas, **não implementadas**

### 5.1 DE-05.a — rank relativo (DECIDED)
- `ω̄_c = r̄_c(n*) / (N + 1)` e `λ_c = ln(ω̄_c / (1 − ω̄_c))`, como no artigo (p. 12, conferido visualmente). Ranks maiores são melhores. `φ = #{λ_c ≤ 0} / C(S, S/2)`.
- **Alternativa descartada:** rank/N do pacote `pbo`. Ela diverge do original e exige substituir λ = +∞ por 6. As diferenças:
  - com N par e sem ranks fracionários, φ coincide nas duas convenções;
  - com N ímpar, a configuração exatamente na mediana conta como sobreajuste em rank/(N+1) (λ = 0) e não em rank/N;
  - a distribuição de λ difere sempre.
- **Validação contra o `pbo`:** comparar o que não depende da convenção (`n*`, `os_rank`, `rn`, `rbn`), aplicar a transformação de cada convenção e comparar φ onde elas coincidem. É preciso um golden novo com **N ímpar**, gerado pelo pipeline em R, para exibir a divergência.

### 5.2 DE-05.b — T não divisível por S (DECIDED)
- O núcleo **rejeita** T % S ≠ 0, porque o artigo exige submatrizes de dimensões iguais.
- A camada de experimento aplica a regra pré-registrada: **descartar as T mod S observações mais antigas** e registrar as datas descartadas no ledger.
- **Descartadas:**
  - blocos desiguais: afastam-se do artigo e criam assimetria IS/OOS;
  - descartar as mais recentes: perde o período mais informativo;
  - escolher S que divida T: faz S depender dos dados.

### 5.3 DE-05.c — duplicatas e empates (DECIDED)
- Colunas de M **idênticas** são erro de configuração. Duplicatas também inflariam N.
- Empate no máximo IS: vence a primeira configuração na ordem canônica da grade (determinístico, igual ao `pbo`), e o relatório conta os empates.
- Empates no ranking OOS: posto médio.

### 5.4 Métrica
Sharpe na convenção DE-02, calculado sobre J (IS, T/2 linhas) e J̄ (OOS, T/2 linhas). Bloco ou conjunto com variância nula é erro explícito.

### 5.5 Escolha de S (procedimento DECIDED; valor fixado no pré-registro)
**Fatos que orientam a escolha [verificado no artigo, p. 11–12 e 21–22]:**
- Para qualquer S, IS e OOS têm **T/2 linhas cada**. S **não** muda o tamanho da amostra em que a métrica é estimada; muda em quantos pedaços (S/2) ela se divide.
- S muda o número de partições C(S, S/2): 70 com S = 8, 924 com S = 12, 12.870 com S = 16. O texto impresso do artigo diz "12, 780", erro tipográfico, já que C(16, 8) = 12.870.
- Como a CSCV é exaustiva, φ não tem erro de amostragem das partições; a resolução de φ é 1/C(S, S/2). O desvio-padrão binomial do artigo (p. 22) trata os logits como independentes, e eles não são, porque as partições se sobrepõem [inferência].
- O artigo sugere S = 16 para cerca de 4 anos diários (blocos trimestrais, "preserve daily, weekly and monthly effects") e alerta que S grande demais destrói a estrutura temporal.
- **Custo:** com somas por bloco pré-computadas, cada partição custa O(S·N), e S ≤ 16 é trivial nesta máquina. Por isso o custo não é critério de escolha [inferência; medir no benchmark B4].

**Regra:**
- S = **o maior par ≤ 16** tal que o comprimento do bloco `⌊T/S⌋ ≥ max(63, H_max)`.
  - 63 pregões ≈ um trimestre, conforme a justificativa do artigo.
  - H_max é a janela de manutenção mais longa da grade, para que um ciclo de posição não se fragmente entre muitos blocos.
- O teto 16 é o valor do exemplo do artigo, uma **escolha operacional**.
- Sensibilidade reportada com S ∈ {8, 12, 16}, se admissíveis.
- Depende de T (período, DD-01) e de H_max (DE-01). Por isso o valor é calculado e congelado no pré-registro, antes de olhar resultados.

### 5.6 DE-06 — dependência, sobreposição, vazamento, purga e embargo (análise; IN PROGRESS)
Distinções [inferência, a partir da construção do estudo]:

| Conceito | No estudo |
|---|---|
| **Sobreposição de observações** | ocorre quando uma observação é um retorno de horizonte longo (rótulo que cobre vários dias). **Não ocorre** se M contém retornos **diários realizados**, atribuídos ao dia em que ocorrem |
| **Vazamento de informação** | o desempenho IS usa informação do período OOS, ou o contrário. Com posições definidas só com informação até t−1, o retorno do dia t não usa o futuro. Num bloco IS que segue um OOS, o sinal usa preços do OOS, mas sob a nula esses preços não predizem os retornos seguintes |
| **Dependência temporal** | autocorrelação dos retornos das estratégias e agrupamento de volatilidade. Sob a nula com retornos independentes, o retorno de uma estratégia é uma diferença de martingal (sem correlação serial), mas não é independente |
| **Purga** | remove observações cujo rótulo cruza a fronteira. **Desnecessária por construção** com retornos diários realizados |
| **Embargo** | remove observações logo após a fronteira para conter dependência serial. Só se justifica se houver dependência relevante |

**Decisões:**
- **Regra de construção (DECIDED):** a matriz M tem retornos **diários realizados**, posições definidas com informação até t−1 e um calendário comum que começa **depois da janela de formação mais longa da grade**. Assim todas as configurações têm sinal em todas as linhas. Isso torna a purga desnecessária.
- **Embargo: não adotado, em aberto com critério.** Será adotado se, na simulação da grade de momentum com preços sintéticos sem sinal (após DE-01), a PBO média se afastar de forma sistemática do valor obtido com retornos de configurações independentes, além do erro de Monte Carlo. Esse afastamento é a consequência observável de dependência IS/OOS. O tamanho do embargo seria então o maior lag de autocorrelação relevante.
- **Fora do escopo:** correções de autocorrelação no PSR (ver a ressalva 2 de DE-02).
