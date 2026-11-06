# Convenções metodológicas — Frente E

Especificação formal das convenções estatísticas e da política de ausências. Data: 2026-11-06.

**Status por seção (atualizado em 2026-11-06, segunda sessão):**
- **§1 (DE-02, DE-02.a, DE-03.a, DE-04.c, DE-08): `DECIDED`.** O desenvolvedor delegou o fechamento dessas decisões ao agente, com base em `docs/DECISION_REVIEW_2026-11-06.md`. O registro formal está em `PROJECT_DECISIONS.md`.
- §2 (DD-05) e a parte CDI de §1.3 (DD-04): **proposta**, inalteradas. Não bloqueiam a Etapa A.
- §4 (N efetivo e V): protocolo `DECIDED`; o estimador principal e a escolha definitiva de V continuam **abertos**.
- §5 (CSCV/PBO, DE-05, S e DE-06): convenções `DECIDED`. **Implementadas e validadas em 2026-11-06 (Etapa B; §5.7).** O valor de S é fixado por regra no pré-registro; o embargo está em aberto, com critério.

- §6 (DE-01, família de momentum): regra **DECIDED** (2026-11-06), **ainda não implementada**; universo (DD-01) é proposta.

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
5. Calibração por simulação: sob H₀ (SR verdadeiro = SR*), PSR ~ U(0,1) aproximadamente para T grande, inclusive com caudas pesadas. Fica para o desenho de DE-07 (semente fixa e tolerância de Monte Carlo declaradas antes de rodar). **DECIDED (2026-11-06, D1):** critério 2b de [DE07_PROTOCOL.md](DE07_PROTOCOL.md) §5.1, com a configuração (6, 6) pré-especificada. O DSR **não** deve ter nível exato (§2.2 do protocolo), então este item vale para o PSR de uma configuração única.

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

## 5. CSCV e PBO (DE-05, S, DE-06): convenções decididas; implementação em §5.7

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
- **Custo:** com somas por bloco pré-computadas, cada partição custaria O(S·N), e S ≤ 16 seria trivial nesta máquina. Por isso o custo não é critério de escolha [inferência; medir no benchmark B4].
  - **Correção (2026-11-06, medido):** a implementação adotada recalcula o Sharpe sobre J e J̄ em cada partição, que é a forma direta da definição, e não usa somas por bloco. Medições (dados aleatórios, só engenharia, fora do ledger): S = 12, T = 4536, N = 144 → 924 partições em 9,5 s; S = 16, T = 4528, N = 144 → 12.870 partições em 131 s.
  - Uma execução do estudo real cabe no orçamento (≤ 2 h). Simulações com centenas de réplicas não cabem.
  - **Gatilho de otimização:** antes de qualquer simulação de CSCV com muitas réplicas (critério do embargo, desenho não nulo), implementar somas por bloco com combinação exata de médias e variâncias (fórmula de combinação de momentos por grupos), validada contra a implementação direta.
  - Mesmo com o custo maior, ele não é critério de escolha de S, porque a escolha precede a medição de desempenho.

**Regra:**
- S = **o maior par ≤ 16** tal que o comprimento do bloco `⌊T/S⌋ ≥ max(63, H_max)`.
  - 63 pregões ≈ um trimestre, conforme a justificativa do artigo.
  - H_max é a janela de manutenção mais longa da grade, para que um ciclo de posição não se fragmente entre muitos blocos.
- O teto 16 é o valor do exemplo do artigo, uma **escolha operacional**.
- Sensibilidade reportada com S ∈ {8, 12, 16}, se admissíveis.
- Depende de T (período, DD-01) e de H_max (DE-01). Por isso o valor é calculado e congelado no pré-registro, antes de olhar resultados.

### 5.7 Implementação e validação (2026-11-06, Etapa B)
**Componentes:**
- `core.cscv`:
  - `in_sample_block_sets` gera as partições, em ordem lexicográfica (= `combn`);
  - `partition_performance` calcula o Sharpe DE-02 em J e J̄ e rejeita T % S ≠ 0;
  - `rank_partitions` determina n*, o tamanho do empate IS e o posto médio OOS;
  - `cscv_pbo` aplica os contratos (N ≥ 2, colunas idênticas proibidas), calcula ω̄ = r̄/(N+1) e λ, e devolve `CSCVResult`, com φ, o nº de empates IS e a probabilidade de perda OOS.
- `experiments.cscv_input`:
  - `trim_oldest_to_multiple` aplica DE-05.b e devolve os índices descartados, para o ledger;
  - `cscv_result_to_json` grava o resultado por partição.
- φ é decidido por `2·r̄ ≤ N + 1`, uma comparação exata entre postos, equivalente a λ ≤ 0.

**Validação:**
- **Contra o `pbo` 1.3.5, 4 casos e 11 combinações (caso, S), incluindo `pbo_odd`:**
  - blocos IS, n*, posto OOS e posto médio em empates **idênticos**;
  - R[n*] e R̄[n*] dentro de rtol 1e-12 / atol 1e-15, a tolerância pré-registrada no manifesto;
  - ω̄, λ e φ do pacote reconstruídos a partir do posto com a regra rank/N, o que confirma a leitura da referência.
- **Contra a definição do artigo:**
  - o φ do projeto é igual a #(r̄/(N+1) ≤ 1/2)/C calculado com os postos do R, em todos os casos;
  - no `pbo_odd`, ele difere do pacote em S = 8 (0,1571 contra 0,1429) e S = 12 (0,1050 contra 0,0942), e a diferença é exatamente a fração de partições com r̄ = (N+1)/2;
  - o posto máximo dá λ = ln N, finito.
- **Comportamento conhecido:**
  - sob a nula i.i.d., E[φ] = ⌊(N+1)/2⌋/N **exato**: 400 conjuntos com N = 5 reproduzem 0,6 dentro de 4 erros-padrão e ficam a mais de 10 erros-padrão de 0,4, o valor com rank/N;
  - sinal forte → φ = 0;
  - invariância a escala positiva e a permutação de colunas.
- **`pbo_ties`:** as funções de nível baixo reproduzem o pacote, inclusive os postos médios; `cscv_pbo` rejeita o caso, porque tem colunas idênticas (DE-05.c).
- **Ponta a ponta:** matriz sintética → descarte das 3 observações mais antigas (T = 243, S = 8) → núcleo → ledger → reconstrução com bytes idênticos (`tests/integration/test_cscv_run.py`).

**Não implementado, por decisão de escopo:** a regressão de degradação de desempenho e a dominância estocástica do §3 do artigo. Os pares (R[n*], R̄[n*]) gravados por partição permitem calculá-las no relatório.

**Limitação:** o artigo da PBO não traz exemplo numérico reproduzível. As figuras vêm de simulações sem semente publicada. A validação contra o artigo é **pela definição**, não por valores publicados.

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

---

## 6. DE-01 — Família de momentum (formalização, 2026-11-06)

**Estado:** regra (A–D) **DECIDED**. Universo (E, DD-01) é **proposta** com valores concretos, a fechar depois da auditoria do COTAHIST (DD-02). Essa auditoria usa só presença e volume, nunca retornos. Período e deslistagem dependem da mesma auditoria. **Nenhum dado real foi lido**: o ledger tem uma única execução, sintética.

**Marcas de evidência:**
- [lit.] literatura;
- [lit., não conferido] atribuição conhecida, sem leitura do texto integral nesta sessão;
- [operacional] escolha do projeto, sem fonte;
- [hipótese] a validar.

### 6.0 Fontes e o que elas sustentam
| Fonte | O que se usa | Status |
|---|---|---|
| Jegadeesh e Titman (1993), JF 48(1) | momentum **cross-sectional**; grade de formação J e manutenção K de 3, 6, 9 e 12 meses; carteiras **sobrepostas** (K coortes); variante com intervalo de uma semana | [lit., não conferido]: a busca confirmou o artigo e o desenho por decis, não a grade J × K nem a sobreposição |
| Moskowitz, Ooi e Pedersen (2012), JFE 104(2) | momentum de **série temporal**: 12 meses, manutenção de 1 mês, ponderação por volatilidade; persistência de 1 a 12 meses com reversão parcial depois | [lit.] resumo e descrições secundárias |
| Kim, Tse e Wald (2016) | o resultado do momentum de série temporal se deve sobretudo à ponderação por volatilidade | [lit.] resumo secundário |
| Brasil: Bonomo e Dall'Agnol (2003), Fonte e Carmona (2005), Mussa et al. (2008), Civiletti, Campani e Roquete (carteiras long-only igualmente ponderadas de vencedoras) e outros | evidência **mista**: o efeito aparece ou some conforme período e método | [lit., não conferido]: só resumos via busca (DE-10) |

A evidência brasileira mista **reforça** a pergunta do projeto: resultados que dependem do período e do método são o cenário em que a seleção entre variantes infla o desempenho.

### 6.1 Escolha da variante (DECIDED)
**Momentum cross-sectional, só comprado (long-only), carteira de vencedoras com pesos iguais, financiada pelo CDI.**

Alternativas descartadas:
- **Série temporal (MOP):**
  - foi documentada para futuros de várias classes de ativos, não para ações individuais;
  - depende de ponderação por volatilidade, que acrescenta parâmetros (janela de volatilidade, alvo) e, segundo Kim, Tse e Wald, é o que gera o resultado;
  - com posições vendidas, exige aluguel de ações na B3.
- **Long-short vencedoras − perdedoras (JT):**
  - na B3, a venda a descoberto tem custo de aluguel e restrições não modeladas;
  - o retorno autofinanciado já é um excesso, o que reabriria a ressalva 1 de DE-02;
  - no desenho adotado, essa ressalva fica **resolvida**: carteira comprada e financiada por caixa ⇒ `e_t = r_p,t − r_f,t`.

Consequências e limitações:
- O retorno mistura prêmio de mercado e momentum [inferência].
- As configurações terão um fator comum forte (o mercado), e portanto correlação alta entre si. Isso é relevante para N efetivo e para V (§4.5 mostrou que um fator comum encolhe a variância entre tentativas). É uma característica real do problema, não um defeito.

### 6.2 A — Sinal (DECIDED)
- **Data de formação t:** último pregão de cada mês do calendário da B3.
- **Sinal:**
  - `s_i(t) = TR_i(t) / TR_i(t − J) − 1`, com TR o índice de retorno total (DA-10/DD-03) no fechamento e J em **pregões** (`J = 21·j`, j em meses);
  - a data t − J é o pregão J posições antes de t.
- **Elegibilidade em t:** ação no universo em t (DD-01, ponto no tempo) **e** com negócio em t **e** em t − J.
  - O sinal só usa preços de negócios reais, sem preço defasado (DE-01.e).
  - Não se cria parâmetro de idade máxima de preço: no universo líquido, a falta de negócio é exceção, e cada exclusão é contada no relatório de qualidade.
- **Intervalo entre formação e manutenção (*skip*): 0** [operacional].
  - A variante com intervalo ("12−1") é conhecida [lit., não conferido], mas não entra como dimensão da grade, para não multiplicar N.
  - Limitação: a reversão de curto prazo pode afetar J = 1 mês.
- **Sem look-ahead:**
  - sinal e universo usam dados até o fechamento de t;
  - a ordem é executada no **fechamento de t+1** (1 pregão de defasagem), e o retorno da posição começa no intervalo (t+1, t+2];
  - isso fecha a parte de *timing* de DD-04 e evita executar no mesmo preço que gerou o sinal.

### 6.3 B — Grade (DECIDED)
- **Família completa F144:** j ∈ {1, …, 12} meses de formação × k ∈ {1, …, 12} meses de manutenção.
  - Mês = 21 pregões na formação; na manutenção, k fechamentos de mês.
- **Subfamílias aninhadas, para a variação com N** (pergunta de pesquisa):

| Família | j, k | N | Fundamento |
|---|---|---|---|
| F4 | {6, 12} × {6, 12} | 4 | o mínimo de uma "grade" |
| F16 | {3, 6, 9, 12} × {3, 6, 9, 12} | 16 | a grade de JT (1993) [lit., não conferido] |
| F144 | {1, …, 12} × {1, …, 12} | 144 | todos os meses do horizonte de persistência de 1 a 12 meses (MOP) |

  F4 ⊂ F16 ⊂ F144. Cada família é tratada como um pesquisador que testou N variantes; M = N da família (DE-04.a).
- **Por que parar em 12 meses:** além desse horizonte a literatura descreve reversão [lit.]; seria outra família.
- **Por que passo mensal:** é a granularidade do rebalanceamento. Passos menores criariam configurações quase idênticas, o que infla M sem informação nova.
- **Dimensões fixas** (fora da grade, para N ser interpretável):
  - quantil: tercil superior;
  - pesos: iguais;
  - skip: 0;
  - defasagem: 1 pregão;
  - custos: 0 (§6.5).
- A grade foi fixada **antes** de qualquer dado e não pode ser ampliada depois (CLAUDE.md §0).

### 6.4 C — Formação e manutenção (DECIDED)
- **Coorte.** Em cada data de formação, forma-se uma coorte com o tercil superior dos elegíveis por `s_i(t)`.
  - Tamanho: `⌈n_elegíveis/3⌉` ações.
  - Empate na fronteira: resolvido pelo identificador do ativo, de forma determinística e contada.
- **Manutenção.** A coorte entra no fechamento de t+1 e sai no fechamento do pregão seguinte ao k-ésimo fim de mês após t.
- **Sobreposição, como em JT [lit., não conferido]:**
  - a carteira mantém k coortes simultâneas, cada uma com 1/k do capital;
  - dentro da coorte, pesos iguais;
  - peso agregado `w_i,t = (1/k)·Σ_coortes 1{i ∈ c}/|c|`;
  - a mesma ação em várias coortes soma pesos.
- **Pesos constantes entre rebalanceamentos**, ou seja, rebalanceamento diário implícito para o alvo [operacional].
  - Alternativa descartada: comprar e manter dentro da coorte. É mais realista quanto ao giro, mas exige rastrear frações por coorte.
  - A diferença afeta o nível do retorno, não a mecânica de seleção estudada [inferência].
- **Aquecimento.** A matriz M começa no primeiro pregão em que **todas** as 144 configurações têm k coortes completas: 12 meses de janela de liquidez/formação mais 12 meses de manutenção, ou seja, 24 meses depois do início dos dados. Todas as configurações compartilham o mesmo calendário (DE-06, regra de construção).

### 6.5 D — Retornos da estratégia (DECIDED)
- `r_p,t = Σ_i w_i,t · r_i,t`, com `r_i,t` o retorno total diário (DA-10). Os pesos vigentes no dia t são os definidos na execução anterior.
- **Excesso:** `e_t = r_p,t − r_f,t`, com o CDI do intervalo (§1.3). A carteira está sempre totalmente investida, com caixa só no caso abaixo.
- **Dias sem negócio** (DE-01.e da revisão, agora DECIDED):
  - ativo mantido sem negócio no dia: avaliado pelo último preço (retorno 0 no dia), com o retorno acumulado realizado no próximo negócio; sinalizado e contado por configuração;
  - compra de ativo sem negócio no dia da execução: o peso fica em caixa (rende CDI) até o primeiro negócio;
  - venda de ativo sem negócio: o ativo continua na carteira até o primeiro negócio.
  - Proibido: preço defasado para sinal ou para execução.
- **Deslistagem** (DE-01.d, procedimento DECIDED, regra final depois da contagem em DD-02):
  - o valor terminal documentado por evento (preço da OPA, relação de troca, liquidação) entra na data efetiva, pela tabela de eventos (DA-10);
  - casos sem documentação são reportados com os limites "último preço" e "−100%" como sensibilidade.
- **Custos: zero no resultado primário** [operacional].
  - Justificativa: a pergunta é a deflação estatística por seleção, não a lucratividade líquida.
  - Limitação: configurações com k curto giram mais e ficam favorecidas sem custos. O **giro** de cada configuração é reportado.
  - Uma sensibilidade com custo proporcional simples pode ser pré-registrada, mas não muda N, porque não é uma escolha entre variantes.
- **Liquidez e impacto:** não modelados. A restrição ao universo líquido é a mitigação.

### 6.6 E — Universo (DD-01: PROPOSTA)
- **Instrumentos:**
  - ações (ON, PN, classes PN, units) do mercado à vista, lote padrão;
  - excluídos: fracionário, BDR, fundos (FII, ETF), direitos, opções e termo.
  - Os códigos do COTAHIST (TPMERC, CODBDI, ESPECI) serão fixados na auditoria contra o layout oficial.
- **Uma classe por emissor:** a mais negociada em cada data de formação, para não ter PETR3 e PETR4 na mesma coorte. O emissor é identificado pelo ISIN, a confirmar na auditoria.
- **Medida de liquidez:** volume financeiro médio diário nos 252 pregões anteriores a t, com presença em ≥ 95% deles.
  - É baseada em **posto** na data, o que neutraliza inflação e mudanças de escala nominal.
  - Os valores 252 e 95% são [operacional].
- **Liquidez como variável de análise:** dois universos disjuntos em cada t.
  - **U1:** emissores com postos 1–30.
  - **U2:** postos 31–60.
  - Cada família é estudada em cada universo **separadamente**, e M é contado por família e universo. Não há seleção entre universos (fecha o "aberto" de §4.1).
- **Sobrevivência:**
  - o universo é reconstruído **ponto no tempo** a partir de todos os arquivos anuais do COTAHIST;
  - ativos deslistados entram enquanto foram negociados [hipótese: confirmar em DD-02 que os arquivos históricos incluem ativos hoje inexistentes];
  - a saída segue §6.5.
- **Período:** posterior ao Plano Real, evitando mudanças de unidade monetária. O início candidato é 2000 e o fim é o último ano completo; fixar em DD-01.
- **Fechar DD-01 exige:**
  - contagem anual de emissores elegíveis, que precisa permitir 60 em todas as datas;
  - estabilidade dos postos;
  - códigos do layout.
  - Se houver menos de 60 elegíveis em algum período, a alternativa pré-definida é encurtar o período, **nunca** reduzir U2 para caber.
- **Limitações do COTAHIST:**
  - preços brutos (DD-03);
  - sem eventos corporativos;
  - sem causa nem valor de deslistagem;
  - mudanças de ticker;
  - ausência de negócio é inferida, não registrada (§2).

### 6.7 O que esta definição destrava (sem implementar)
| Decisão futura | Insumo agora disponível | Próximo passo |
|---|---|---|
| **S** (§5.5) | H_max = 12 meses = 252 pregões ⇒ bloco ≥ max(63, 252) = 252. Ilustração, não decisão: dados de 2000 a 2025 (≈ 6550 pregões) menos 24 meses de aquecimento ≈ 6050 ⇒ S = 16 dá blocos de ≈ 378 ≥ 252 | fixar no pré-registro com o T real |
| **N efetivo** (§4.2) | a estrutura de correlação da família pode ser gerada **sem dados reais**: aplicar F144 a preços sintéticos sem sinal para 60 ativos (com fator de mercado comum, coerente com §6.1) e calcular a correlação entre as 144 séries | implementar a regra de carteira e rodar o critério de §4.2 e §4.5 com essas estruturas |
| **Embargo** (§5.6) | com pesos definidos até t e coortes sobrepostas, sob a nula o retorno da estratégia é diferença de martingal [inferência] | simular a PBO da grade sem sinal e comparar com configurações independentes |
| **Desenho não nulo** (DE-07, DE-04.b) | o sinal plantável é a persistência cross-sectional de retornos | desenhar o processo gerador e o critério de "taxa nominal" (§4.5, leitura 5) |
| **Pré-registro** | regra, grade, famílias, universos e timing definidos | fechar DD-01, DD-02, DD-03, DD-05 e o valor de S |
| **Custo computacional** | 2 universos × 3 famílias de CSCV; F144 com S = 16 leva ≈ 131 s (§5.5) | cabe no orçamento |

---

## 7. Carteira de momentum sintética e estudos sobre a grade (ciclo de 2026-11-06)

### 7.1 Etapa A — regra implementada (`core.momentum`) e lacunas de DE-01
**Implementado exatamente como §6.2–6.5**, com os seguintes pontos de leitura explicitados:
- **Linhas:** a linha t de retornos é o intervalo (t − 1, t]. Uma coorte formada no fechamento de f e executada no fechamento de f + 1 rende a partir da linha f + 2. Ela sai no fechamento de x = (k-ésimo fim de mês seguinte) + 1, e a linha x ainda conta.
- **Fins de mês:** o último pregão da série nunca é data de formação, porque não haveria pregão seguinte para executar. Uma coorte cujo k-ésimo fim de mês cai fora da série fica ativa até o fim dos dados.
- **Calendário comum:** a primeira formação usada é o primeiro fim de mês com ≥ 21·12 pregões de histórico; a matriz começa na linha seguinte à execução da 12ª formação. Por construção, F4 e F16 são **colunas idênticas** às de F144 (testado).
- **Pesos:** o motor verifica em todas as linhas do calendário comum que Σ pesos + caixa = 1 (k coortes de 1/k encadeadas: a execução de uma coorte coincide com a saída de outra).
- **Validação:** uma implementação de referência independente, um laço dia a dia em `tests/unit/test_momentum.py`, reproduz o motor com atol 1e-15, inclusive com 5% de dias sem negócio e compras adiadas.

**Lacunas encontradas:**

| ID | Lacuna | Decisão |
|---|---|---|
| DE-01.g | §6.4 diz "empate resolvido pelo identificador", sem a direção | **DECIDED (2026-11-06):** vence o menor índice na ordem canônica dos identificadores; as formações com empate na fronteira são contadas. Alternativas: sorteio com semente (acrescenta aleatoriedade sem ganho) ou incluir todos os empatados (tamanho de coorte variável, contraria ⌈n/3⌉). Com preços contínuos, empates exatos são raros |
| DE-01.h | §6.5 manda manter o ativo cuja venda não pôde ser executada, mas com coortes sobrepostas a fatia que expira já foi ocupada pela coorte nova (peso total > 1). Execução de uma coorte = saída de outra | **PENDING.** O motor gera `PendingDecisionError("DE-01.h")`. Alternativas: (a) excesso temporário financiado a CDI; (b) a coorte nova entra só com o capital liberado; (c) renormalizar a carteira. Não ocorre nos cenários sintéticos completos. Deve ser decidido antes do estudo real, com a contagem de casos da auditoria |
| — | coorte sem nenhum elegível | **erro explícito**, sem convenção. U1/U2 garantem ≥ 30 elegíveis |
| DE-01.i | §6.5 manda reportar o **giro**, mas não o define (pesos constantes implicam rebalanceamento diário) | **PENDING**, porque só é necessário no relatório real. Não implementado |

### 7.2 Cenários sintéticos sem sinal (fixados em 2026-11-06, antes dos estudos da Etapa B)
`experiments.momentum_scenarios`:
- 60 ativos, T = 6550 pregões, r_f = 0,04%/dia.
- Média de **todos** os ativos = r_f, então o SR verdadeiro de qualquer configuração é zero (nula global exata).
- Os parâmetros cross-sectionais vêm de fluxo fixo e são iguais em todas as réplicas.

| Cenário | σ mercado | β | σ idiossincrático | volatilidade estocástica (φ, σ_h) |
|---|---|---|---|---|
| base | 1,2% | U(0,7; 1,3) | U(1,2%; 2,5%) | — |
| low_common | 0,5% | U(0,7; 1,3) | U(1,2%; 2,5%) | — |
| vol_clustering | 1,2% | U(0,7; 1,3) | U(1,2%; 2,5%) | (0,98; **0,06**), emenda 3 |
| heterogeneous | 1,2% | U(0,3; 1,7) | U(0,5%; 4%) | — |

O gerador separa a dependência (fator comum, heterogeneidade, aglomeração de volatilidade) da previsibilidade, que é nula: retornos independentes no tempo na média. Testes em `tests/unit/test_null_models.py`.

**Emenda 3 (2026-11-06): parâmetro de `vol_clustering` corrigido.**
- **Valor original e falha:** com σ_h = 0,15, o desvio estacionário de log σ era 0,15/√(1 − 0,98²) ≈ 0,75, e a volatilidade chegava a ~10 vezes o nível médio. Na execução registrada `run-7dd0ad1965784ee8ae05681dc337c168` (`sim-grid-neff-embargo`), o gerador recusou um retorno ≤ −100% e a execução **falhou**. A falha está no ledger.
- **Exposição a resultados:** nenhum resultado foi produzido nem visto, porque a falha ocorreu antes da agregação.
- **Novo valor:** σ_h = 0,06, com desvio estacionário ≈ 0,30 e multiplicadores de volatilidade em [0,4; 2,5] em 99,7% do tempo [operacional].
- **Execução de mecânica da Etapa A:** `run-de18681b…` usou o valor antigo, registrado nos seus parâmetros, numa réplica sem falha. O resumo em §7.2 refere-se a esse valor.
- **Prevenção:** `tests/integration/test_momentum_scenarios.py` gera todos os cenários com 20 sementes.

**Execução de mecânica** (`run-de18681ba37247d788bc0db5233746e1`, `sim-momentum-mechanics`, `RunKind.SYNTHETIC`; artefato `10e56474…`): F144 em cada cenário, uma réplica.
- Em todos os cenários: 6049 linhas (2001-12-04 a 2025-02-07) e 290 formações; zero empates na fronteira, compras adiadas e dias de preço defasado (painel completo).
- Correlação entre configurações (fora da diagonal, mín / mediana / máx):

| Cenário | Mín | Mediana | Máx |
|---|---|---|---|
| base | 0,935 | 0,985 | 0,9998 |
| low_common | 0,753 | 0,938 | 0,9991 |
| vol_clustering | 0,950 | 0,988 | 0,9998 |
| heterogeneous | 0,877 | 0,969 | 0,9995 |

- **[Fato]** A carteira só comprada herda o fator de mercado, e as 144 configurações são fortemente correlacionadas, como antecipado em §6.1. **[Inferência]** Isso coloca a grade na região de §4.5 em que o fator comum domina a correlação.

### 7.3 Etapa B — protocolo de N efetivo e V na grade (fixado antes da execução registrada)
**Experimento:** `experiments.grid_study`, `sim-grid-neff-embargo`, `RunKind.SYNTHETIC`.
- 50 réplicas × 4 cenários (§7.2), com F4, F16 e F144 extraídos da mesma grade F144.
- Por réplica, dois conjuntos independentes (A, B) e uma permutação de linhas de A.

**Grandezas, todas separadas:**

| Grandeza | Definição |
|---|---|
| Correlação entre configurações | Ĉ = correlação de Pearson das séries de excesso da família em A |
| Autocorrelação dentro de uma configuração | média entre configurações da ACF nos lags 1, 5, 21 e 63 (F144) |
| V entre tentativas | variância amostral (ddof = 1) dos ŜR da família numa réplica |
| V marginal | média, entre configurações, da variância de ŜR_i entre réplicas: a variância verdadeira do estimador |
| V teórica sob H₀ | 1/(T − 1), com T = linhas do calendário comum |
| Alvo | E[máx ŜR] da família, média entre réplicas |
| N* de referência | f⁻¹(E[máx]/√V_marginal) (None se < 2) |

**Avaliação de pares (N̂, V)** (emenda 1 de §4.5):
- erro relativo `E[√V·f(N̂)]/E[máx] − 1`, com N̂ calculado sobre Ĉ **em cada réplica** (uso realista), com erro-padrão pelo método delta pareado;
- cada um dos 6 estimadores × {V entre tentativas, V_H0};
- N̂ < 2 em alguma réplica torna o par indefinido na estrutura (DE-04.c).

**Regra de seleção** (§4.2, emenda 1 e acréscimo da emenda 2, registrado agora e antes da execução):
1. estruturas = 9 genéricas (`run-de53b361…`, Tabelas 1 e 2 de §4.5, carregadas pelo SHA-256 do artefato) + 12 da grade (4 cenários × 3 famílias);
2. **admissível** = par definido em todas as estruturas e **nunca anticonservador** além de 2 erros-padrão (erro < −2·EP). Este requisito é **novo**: a regra original (argmin do pior |erro|) sempre escolheria algo, inclusive um par anticonservador, o que inflaria falsas descobertas;
3. principal = admissível com menor pior |erro|; empate (diferença ≤ 2 EP) → mais estruturas com erro ≥ 0 → menor pior erro;
4. sem admissível → **em aberto**;
5. **alcance:** a seleção vale **sob a nula global**. A confirmação depende do desenho não nulo de DE-07, porque V_H0 é exato sob a nula por construção (§4.5).

### 7.4 Etapa B — protocolo do embargo (emendas 2 e 2b ao critério de §5.6)
**Problema do critério original de §5.6 (identificado antes de qualquer execução registrada):**
- §5.6 compara a PBO da grade com "o valor obtido com configurações independentes", E[φ] = ⌊(N+1)/2⌋/N.
- Esse valor exato depende de as configurações serem permutáveis. Configurações correlacionadas e heterogêneas mudam E[φ] **sem nenhum vazamento**.
- O critério confundiria estrutura transversal com dependência IS/OOS.

**Emenda 2 — metades independentes:** φ_indep usa o desempenho IS de A e o OOS do mesmo bloco complementar de B. Preserva a estrutura transversal e elimina a dependência entre metades.

**Emenda 2b — permutação de linhas** (motivo registrado com transparência):
- **O que motivou:** a réplica usada para cronometrar o estudo (não registrada, não reportada como resultado) mostrou, em F144 / base, φ = 0,95 sem embargo, 0,66 com metades independentes e 0,94 / 0,91 com embargo de 21 / 63.
- **Leitura [inferência]:** a diferença grande não vem de dependência temporal: o embargo quase não a altera e a ACF é pequena. Vem da **complementaridade** das metades numa mesma amostra: m_IS + m_OOS = 2·m_total para a média do fator comum. Configurações com exposição diferente ao fator comum trocam de posição entre metades.
- **Consequência:** as metades independentes eliminam a complementaridade **e** a dependência temporal, então não isolam o que o embargo pode corrigir.
- **O que foi acrescentado:** φ_perm é a CSCV sobre as linhas de A **permutadas**, com a mesma permutação para todas as configurações. Isso destrói a dependência temporal e preserva a complementaridade e a correlação transversal.

**Regra do embargo:**
- adotar se, em alguma estrutura (cenário × família), |E[φ − φ_perm]| > 2 EP (efeito temporal detectável) **e** algum embargo h ∈ {21, 63} reduzir |E[φ_h − φ_perm]| em mais de 2 EP;
- caso contrário, não adotar, com a evidência registrada;
- embargo = descartar as h primeiras linhas de todo bloco cujo bloco anterior pertence ao outro conjunto (`core.cscv.block_partition_performance`, testado contra a definição);
- S = 16, que é o valor da regra de §5.5 para T = 6048 e H_max = 252 (bloco de 378 ≥ 252).

**Medida descritiva, sem regra:** E[φ − φ_indep] (efeito de complementaridade). Se for grande, é um achado relevante para DE-09: a PBO sob a nula desta família não seria 0,5, e "PBO material" precisaria ser interpretada contra a distribuição nula da própria família.

### 7.5 Otimização da CSCV (gatilho de §5.5 atingido)
- **Medição:** no tamanho da simulação, a forma direta levou 23 s (F16) e cerca de 131 s (F144) por CSCV com S = 16. O estudo exige 4 CSCVs por família e réplica.
- **Implementação:** `block_partition_performance`, com estatísticas por bloco (n, média, M2 centrado) e agregação exata de grupos. Resultado: F144 com S = 16 em **1,1 s** e F16 em 0,17 s.
- **Equivalência (testada):**
  - blocos e postos idênticos;
  - Sharpes dentro de rtol 1e-12 / atol 1e-15 frente à forma direta, ao valor exato (aritmética racional) e aos 11 golden do R;
  - no extremo |μ|/σ ≈ 1e5 (fora do uso), o erro relativo observado é ≈ 2e-12 e a forma direta é mais precisa; a fórmula ingênua Σx² − nμ² colapsaria.
- `cscv_pbo` passou a usar a versão por blocos; a forma direta permanece como referência dos testes.

### 7.6 Resultados da Etapa B (`run-36aa72b3fe09461b8fe423cc7451d134`)
**Execução e falha anterior:**
- `sim-grid-neff-embargo`, 200 réplicas (50 × 4 cenários), ≈ 47 min (≈ 14 s por réplica; CSCV F144 com S = 16 ≈ 1,1 s).
- Artefato `c21b4ee1…` (1 MB, fora do Git). Código registrado como *dirty*; o commit da Etapa B contém exatamente esse código.
- A tentativa anterior `run-7dd0ad19…` falhou (emenda 3) e está no ledger.

**Tabela 3 — estrutura da grade sob a nula** (médias sobre 50 réplicas; V em unidades de SR diário²; V_H0 = 1/(T−1) = 1,65e-4):

| Cenário / família | corr. média entre configurações | E[máx ŜR] ± EP | V marginal | V entre tentativas | N* (V marginal) |
|---|---|---|---|---|---|
| base / F4 | 0,989 | −0,0003 ± 0,0015 | 1,10e-4 | 2,2e-6 | — (< 2) |
| base / F16 | 0,987 | 0,0013 ± 0,0014 | 1,07e-4 | 2,6e-6 | — |
| base / F144 | 0,983 | 0,0032 ± 0,0014 | 1,06e-4 | 3,3e-6 | — |
| heterogeneous / F4 | 0,979 | 0,0044 ± 0,0015 | 1,14e-4 | 4,1e-6 | — |
| heterogeneous / F16 | 0,974 | 0,0063 ± 0,0015 | 1,17e-4 | 5,3e-6 | 2,15 |
| heterogeneous / F144 | 0,967 | 0,0089 ± 0,0015 | 1,16e-4 | 6,3e-6 | 2,87 |
| low_common / F4 | 0,954 | 0,0021 ± 0,0020 | 1,98e-4 | 8,6e-6 | — |
| low_common / F16 | 0,943 | 0,0054 ± 0,0019 | 1,89e-4 | 1,21e-5 | — |
| low_common / F144 | 0,928 | 0,0092 ± 0,0019 | 1,83e-4 | 1,42e-5 | 2,40 |
| vol_clustering / F4 | 0,990 | 0,0019 ± 0,0018 | 1,67e-4 | 1,4e-6 | — |
| vol_clustering / F16 | 0,987 | 0,0031 ± 0,0018 | 1,63e-4 | 1,8e-6 | — |
| vol_clustering / F144 | 0,984 | 0,0049 ± 0,0017 | 1,62e-4 | 2,3e-6 | — |

- **Autocorrelação dentro das configurações (F144):** |ACF| ≤ 0,003 nos lags 1, 5, 21 e 63, em todos os cenários, contra 2/√T ≈ 0,026. Sem dependência serial detectável na média, coerente com §5.6 (diferença de martingal sob a nula).
- **V marginal frente a V_H0:** de −36% a +20%. Com 50 réplicas, o erro relativo de uma variância é ≈ 20% e as configurações quase não diversificam esse erro (correlação ≈ 0,98). A diferença é compatível com ruído de Monte Carlo; **não** é evidência contra V_H0 como variância marginal.

**N efetivo e V — leituras:**
1. **[Fato]** Com correlação de 0,93 a 0,99 (fator de mercado numa carteira só comprada), **E[máx ŜR] é estatisticamente indistinguível de zero** em 8 das 12 estruturas. N* só é definido em 3 delas.
2. **[Fato]** O critério pré-registrado (§4.2/§7.3: erro relativo `E[pred]/E[máx] − 1`) divide por esse alvo quase nulo e ruidoso, produzindo erros de −647% a +5384%. Aplicada mecanicamente, a regra "seleciona" `gao + V_H0`, com pior erro de 2710%. **Essa seleção não é adotada: o critério é não informativo nesta estrutura. Nenhum estimador foi validado.** A saída mecânica fica registrada no artefato, sem edição.
3. **[Fato]** V entre tentativas = 2–8% de V marginal (≈ 1 − ρ̄). Nyholt, Galwey e A.3 dão N̂ < 2 (fora do domínio, DE-04.c) em F4/F16 de vários cenários. Os estimadores por autovalores enxergam o fator comum e concluem que a família vale cerca de uma tentativa.
4. **[Post hoc, descritivo, não usado para decidir]** Na escala do PSR, z = (E[pred] − E[máx])/√V_H0:
   - (M, V entre tentativas): |z| ≤ 0,20 em todas as 12 estruturas; taxa de DSR > 0,95 sob a nula entre 0 e 0,08 (média 0,04; EP ≈ 0,03 com 50 réplicas);
   - (N efetivo, V entre tentativas): até 0,10 (anticonservador, como previsto em §4.3);
   - V_H0: conservador demais (taxas ≤ 0,02, |z| até 2,4).
   - Isso **corrobora** o primário provisório da emenda 1, que foi registrado antes destes dados.
5. **[Inferência]** Com um fator comum dominante, ŜR_máx = componente comum + máximo dos componentes específicos. O componente comum tem variância próxima da variância de estimação usada no PSR, então PSR(SR₀) com SR₀ ≈ E[máx] tende a ficar próximo da calibração nominal. No caso independente (§4.5), a taxa foi 0,06–0,13%. **A calibração do DSR sob a nula depende da estrutura de correlação** (entrada direta para DE-07).

**Embargo — leituras:**
- A regra (emenda 2b) dá **não adotar**. Em nenhuma das 12 estruturas |E[φ − φ_perm]| > 2 EP (maior: base/F144, −0,018 ± 0,009, 1,9 EP), e o embargo de 63 nunca reduziu a diferença.
- Efeito mínimo detectável ≈ 0,02 de PBO.
- **[Fato]** A "complementaridade" inferida da réplica de cronometragem (§7.4) **não se confirmou em média**: E[φ − φ_indep] oscila entre −0,10 e +0,08, com sinal variável, quase sempre dentro de 2 EP. A réplica de cronometragem (φ = 0,95) estava na cauda da distribuição.
- **[Fato, relevante para DE-09]** A distribuição nula da PBO **num único conjunto de dados** é muito larga: percentis 5% / 50% / 95% entre [0,02–0,21] / [0,36–0,61] / [0,76–0,96], conforme a estrutura. Uma PBO de 0,9 ocorre sob a nula. A média nula também não é exatamente 0,5 (base/F16: 0,403 ± 0,033), porque as configurações não são permutáveis.

**Proposta para o próximo protocolo (emenda 4, PENDENTE, não executada):**
- trocar o erro relativo por **erro padronizado na escala do PSR** (z acima) e pela **calibração** P(DSR > 0,95) sob a nula;
- usar ≥ 500 réplicas por estrutura (EP da taxa ≈ 1%), sem CSCV, para conter o custo (≈ 4 s por réplica);
- considerar estimar N efetivo sobre a correlação dos **componentes específicos** (desvios da média transversal), uma ideia **sem fonte conferida**.
- Tudo isso a pré-registrar antes de executar, de preferência junto com o desenho não nulo de DE-07.

**Atualização (2026-11-06, DE-07 Etapa A):** a emenda 4 foi incorporada ao protocolo v1 em [DE07_PROTOCOL.md](DE07_PROTOCOL.md) (proposto na Etapa A; D1–D7 aprovadas na Etapa B1, 2026-11-06). Achados que revisam leituras desta seção, sem apagá-las:
- **leitura 4:** a taxa "0–8%" não tem código versionado que a produza (o `grid_study` não calcula DSR); fica como exploratória;
- **leitura 5:** a inferência foi confirmada analiticamente. Na equicorrelação, P(DSR > 0,95) com (M, E[V entre tentativas]) cresce de 0,12% para 4,98% com ρ, sem passar de α (protocolo §2.2, Tabela A, testada);
- **leitura 1:** com ρ ≥ 0,95, E[máx Z] < f(2), de modo que o N implícito não existe por construção, e não só por ruído (protocolo §2.4);
- a "ideia sem fonte" de N efetivo sobre os componentes específicos **não** entrou no protocolo.

