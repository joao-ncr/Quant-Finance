# Revisão de decisões pendentes — 2026-11-06

> Documento de análise. **Nenhuma decisão aqui está aprovada.** O status oficial de cada item fica em [PROJECT_DECISIONS.md](../PROJECT_DECISIONS.md). Quando o usuário aprovar, rejeitar ou modificar uma recomendação, o resultado é registrado lá, e este documento continua sendo a fundamentação histórica.

Legenda de evidência usada no texto:
- **[verificado]**: conferido nesta sessão na fonte original (imagem da página do PDF) ou por cálculo reproduzível.
- **[literatura, não conferido]**: atribuído a uma fonte que não foi lida no original nesta sessão.
- **[inferência]**: raciocínio próprio, sem fonte direta.

## 0. Fontes conferidas no original

Os PDFs foram baixados de davidhbailey.com em 2026-11-06 e renderizados como imagem para leitura (pypdfium2 em ambiente efêmero, fora do projeto).

| Artigo | Arquivo | SHA-256 |
|---|---|---|
| Bailey & López de Prado (2014), *The Deflated Sharpe Ratio* | `dhbpapers/deflated-sharpe.pdf` (22 p.) | `ca4a1e834a2954b84eb560776e903329cd9cef318d458aadf6d67bc8166d10b5` |
| Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting* | `dhbpapers/backtest-prob.pdf` (34 p.) | `d8bfbadaaedb430d9dec929646d92f3580fdf18238f34121ae7738fa1c69a63d` |
| Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier* | `dhbpapers/sharpe-frontier.pdf` (46 p.) | `7a756315dc645e343f9b619a727adbd10fde7347af14ae7000bdf3c99397b0e5` |

São preprints. A paginação citada é a desses arquivos, não a da versão publicada.

### 0.1 Âncoras numéricas publicadas, reproduzidas por cálculo independente [verificado]

| # | Fonte | Entradas | Publicado | Reproduzido |
|---|---|---|---|---|
| A1 | DSR 2014, p. 9–10 | SR_a = 2,5; 250 obs/ano; T = 1250; N = 100; V_a = 1/2; γ₃ = −3; γ₄ = 10 | SR₀ ≈ 0,1132; DSR ≈ 0,9004 | 0,1132; 0,9004 |
| A2 | DSR 2014, p. 10 | idem, N = 46 | 0,9505 | 0,9505 |
| A3 | DSR 2014, p. 10 | γ₃ = 0, γ₄ = 3, N = 88 | 0,9505 | 0,9505 |
| A4 | SR Frontier 2012, p. 10 | SR = 0,458 (mensal); n = 24; γ₃ = 0; γ₄ = 3 | PSR(0) = 0,982 | 0,9817 |
| A5 | SR Frontier 2012, p. 10 | SR = 0,458; n = 24; γ₃ = −2,448; γ₄ = 10,164 | PSR(0) = 0,913 | 0,9134 |
| A6 | SR Frontier 2012, p. 10 | idem, n = 36 | PSR(0) = 0,953 | 0,9535 |
| A7 | SR Frontier 2012, p. 11 e 20 | SR = 2/√12; SR* = 1/√12; γ₃ = −0,72; γ₄ = 5,78; 95% | MinTRL = 59,895; PSR(n = 59,895) = 0,95 | 59,895; 0,9500 |

Fatos confirmados visualmente que não estavam claros antes:
- **DSR 2014, p. 10:** V[{ŜRₙ}] = 1/2 é dado em unidades **anualizadas**, e o artigo o converte para unidades diárias como `1/(2·250)`. ŜR entra como `2,5/√250`, e o denominador usa `√1249 = √(T−1)`.
- **SR Frontier 2012, eq. 41 (p. 21):** γ₃ = E[(r−μ)³]/σ³ e γ₄ = E[(r−μ)⁴]/σ⁴, ou seja, curtose bruta (normal = 3). O código do Apêndice A.3 (p. 20) usa `stats[3]-1` com `5.78`. As entradas são "stats on excess returns", "non-annualized".
- **SR Frontier 2012, p. 11:** "Eqs. (11) and (13) are built upon Eq. (8), which applies to an asymptotic distribution. CLT is typically assumed to hold for samples in excess of 30 observations (Hogg and Tanis (1996))… the moments inputted in Eq. (13) must be computed on longer series for CLT to hold."
- **PBO, p. 11–13:** a partição é "an even number S of disjoint submatrices **of equal dimensions** … of order (T/S × N)". O rank relativo é **ω̄_c := r̄_c(n*)/(N + 1) ∈ (0, 1)** e o logit é λ_c = ln(ω̄_c/(1 − ω̄_c)). A PBO é definida como φ = ∫₋∞⁰ f(λ) dλ e descrita como a taxa em que a configuração ótima IS fica abaixo da mediana OOS. Ranks maiores são melhores. O artigo não diz como desempatar.
- **DSR 2014, Apêndice A.3 (p. 14–15):** N̂ = ρ̂ + (1 − ρ̂)·M, com ρ̂ a correlação média fora da diagonal entre as M tentativas (eq. 8–9). O próprio artigo aponta duas limitações: a correlação capta só dependência linear, e com `T < M(M−1)/2` a matriz fica mal condicionada e ρ̂ "may itself be overfit". Como alternativa, sugere redução de dimensão ou medidas de teoria da informação.

### 0.2 Cálculos de apoio [verificado: script reproduzível, valores abaixo]

Precisão da aproximação de E[max] usada no DSR, com N variáveis normais padrão i.i.d., comparada com o valor exato por integração numérica:

| N | aproximação | exato | erro relativo |
|---|---|---|---|
| 2 | 0,5198 | 0,5642 | −7,9% |
| 3 | 0,8528 | 0,8463 | +0,8% |
| 5 | 1,1926 | 1,1630 | +2,6% |
| 10 | 1,5746 | 1,5388 | +2,3% |
| 20 | 1,9007 | 1,8675 | +1,8% |
| 100 | 2,5306 | 2,5076 | +0,9% |
| 1000 | 3,2551 | 3,2414 | +0,4% |

Comportamento da expressão fora do domínio usual: o fator `(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))` é **negativo para N < 1,2836** (−0,78 em N = 1,01; −0,32 em N = 1,1) e vale 0,216 em N = 1,5. Com N = 1, Φ⁻¹(0) = −∞.

---

## 1. DA-03 — Ledger de tentativas

1. **Problema.** O N do DSR e a auditoria do estudo dependem de um registro completo, em que nenhuma tentativa possa sumir ou mudar sem deixar rastro.
2. **Alternativas.**
   - (a) Aprovar a implementação atual.
   - (b) Aprovar com uma regra operacional de âncora externa.
   - (c) Acrescentar assinatura criptográfica ou carimbo de tempo externo.
   - (d) Trocar por um banco de dados (ex.: SQLite).
3. **Referências.** Um log encadeado por hash detecta alteração, reordenação e remoção no meio do arquivo, mas não detecta truncamento do fim nem reescrita completa sem um compromisso externo. É a limitação clássica de logs seguros em máquina não confiável [literatura, não conferido: Schneier & Kelsey, *Secure audit logs*, 1999]. A suíte do projeto testa exatamente essa limitação.
4. **Consequências.**
   - O modelo de ameaça é corrupção acidental e autoengano do próprio pesquisador, não um adversário.
   - Quem tem acesso de escrita sempre pode reescrever o arquivo; o que se ganha é tornar isso **detectável** sempre que existir uma âncora publicada anteriormente.
   - (c) agrega pouco para um pesquisador único e acrescenta dependência ou infraestrutura.
   - (d) perde a legibilidade linha a linha e o diff no Git sem ganho de integridade.
5. **Recomendação: (b).** Aprovar a implementação atual e adotar como regra que o `ledger head` (`N:SHA256`) seja registrado:
   - em todo commit que altere o `experiments/ledger.jsonl`, que é versionado, de modo que o próprio histórico do Git funcione como âncora;
   - no documento de pré-registro, fixando o estado do ledger antes do estudo;
   - em cada relatório de resultados.
6. **Para aprovar:** DA-03 → DECIDED, com a regra de ancoragem acima.

## 2. DE-02 — Convenções do Sharpe (consistência com as fórmulas originais)

1. **Problema.** O PSR e o DSR só são interpretáveis se ŜR, γ₃, γ₄ e T estiverem na mesma frequência e com as mesmas definições usadas nas fórmulas.
2. **Proposta e alternativas, item a item.**

| Item | Proposta | Alternativa | O que o original diz [verificado] |
|---|---|---|---|
| Frequência | diária, sem anualizar | anualizada | SR "non-annualized", na frequência das observações; T conta observações (A1–A7) |
| Retorno | simples | log | não especifica |
| Excesso | r − CDI | bruto | "stats on excess returns" (2012, A.3) |
| σ̂ do SR | ddof = 1 | ddof = 0 | define populacionalmente (σ = √E[(r−μ)²]); não prescreve estimador |
| γ̂₃, γ̂₄ | Pearson plug-in, bruta, sem correção de viés | G₁/G₂ corrigidos | definições populacionais (eq. 41); curtose bruta |
| Anualização | √252, só para apresentação | — | o exemplo usa 250; anualização não entra na fórmula |

3. **Consequências.**
   - ddof = 1 vs 0 altera o SR por um fator √((T−1)/T), que vale 0,9996 com T = 1250. Isso é irrelevante na prática, mas precisa ser fixado.
   - Misturar σ̂ (ddof = 1) no SR com m₂ (ddof = 0) nos momentos é uma inconsistência de ordem 1/T [inferência]. Ela não afeta a garantia de radicando ≥ 0, que vale para qualquer ŜR desde que os momentos sejam os de Pearson plug-in.
   - O retorno simples é o que se agrega entre ativos de uma carteira; o log não se agrega. Isso pesa a favor do simples para carteiras de momentum [inferência].
   - **Ressalva 1:** o artigo de 2012 (p. 4 e 8) afirma, citando Opdyke (2007), que a distribuição limite vale para retornos estacionários e ergódicos, não só i.i.d. Não conferi Opdyke. Lo (2002) mostra que autocorrelação altera a variância do estimador do SR [literatura, não conferido]. Para estratégias com janelas sobrepostas, isso fica em DE-06 e deve ser medido na simulação (DE-07), não suposto.
   - **Ressalva 2:** "excesso ao CDI" está bem definido para carteira comprada e financiada por caixa. Para carteira long-short autofinanciada, o retorno já é um excesso e subtrair o CDI seria errado. A definição do retorno da estratégia pertence a DE-01; DE-02 fixa só o estimador [inferência].
4. **Recomendação.** Aprovar DE-02 como proposta, registrando as duas ressalvas como dependências explícitas (de DE-01 e de DE-06/DE-07).
5. **Para aprovar:** DE-02 → DECIDED (estimador e unidades), com as ressalvas.

## 3. DD-04 (parte CDI) — Alinhamento temporal

1. **Problema.** Comparar o retorno entre dois pregões com o CDI acumulado exatamente no mesmo intervalo, sem look-ahead e sem preencher lacunas em silêncio.
2. **Proposta atual (implementada em `core.rates`).** A taxa datada em d remunera até o próximo dia útil do CDI. O intervalo (t₋₁, t] compõe as taxas das datas em [t₋₁, t). É erro se t₋₁ não for data do CDI, ou se não houver data do CDI ≥ t.
3. **Casos de divergência [inferência sobre o mecanismo; as ocorrências precisam ser medidas nos dados].**

| Caso | Exemplo plausível | Comportamento atual | Avaliação |
|---|---|---|---|
| Dia útil bancário sem pregão | feriado municipal de São Paulo em que a B3 fecha | o intervalo compõe duas taxas | correto, sem erro |
| Pregão sem data do CDI | pregão em dia que não é útil bancário, **ou** lacuna da série | erro | a regra está certa para lacuna da série; para dia não útil legítimo, o CDI acumulado correto é o produto das datas em [t₋₁, t), que pode ser vazio (r_f = 0) |
| Fim da amostra | último pregão sem taxa posterior | erro | correto |
| Unidade da série | SGS 12 em % a.d. (fontes secundárias) | conversão /100 | conferir na ingestão contra a série anualizada correspondente |

4. **Consequência central.** Sem um calendário bancário **independente**, um dia não útil legítimo é indistinguível de uma falha da série. Relaxar o erro sem esse calendário transformaria falhas de fonte em r_f = 0 silencioso.
5. **Recomendação.** Aprovar a convenção como proposta, **mantendo a rejeição explícita**. O caso "pregão sem data do CDI" passa a ser resolvido só quando houver um calendário de dias úteis bancários de fonte declarada (decisão de ingestão, pendente). Cada ocorrência é contada e reportada.
6. **Para aprovar:** DD-04 (parte CDI) → DECIDED. O timing do sinal continua em DE-01.

## 4. DD-05 — Retornos ausentes

1. **Problema.** O núcleo exige matriz finita, mas os dados reais têm ausências com causas diferentes. É preciso representá-las sem fabricar retornos e sem perder auditoria.
2. **Avaliação da proposta [inferência].** Os 7 estados cobrem as causas relevantes. O ponto fraco é a **evidência**: o COTAHIST só registra negociações, então a ausência de um ativo num pregão é inferida, não observada. Sem uma regra de classificação, a escolha entre `no_trade`, `suspended` e `source_gap` vira julgamento caso a caso.
3. **Alternativas.**
   - (a) Manter os 7 estados e classificar caso a caso.
   - (b) Manter os 7 estados com **regras de classificação determinísticas e campo de evidência**.
   - (c) Reduzir a "observado / ausente".
4. **Regras propostas para (b):**

| Estado | Atribuído quando | Evidência exigida |
|---|---|---|
| `traded` | há registro válido no COTAHIST | o registro |
| `no_trade` | pregão em que outros ativos têm registro, e o ativo está dentro do intervalo de listagem, mas não tem registro | padrão para ausência sem outra evidência |
| `suspended` | só com fonte externa documentada (comunicado da B3) | referência da fonte |
| `source_gap` | pregão sem nenhum registro no arquivo, registro corrompido ou falha de checagem de integridade | resultado da checagem |
| `not_yet_listed` | antes do primeiro registro | primeiro registro; aproximação documentada da data de listagem |
| `delisted` | depois da saída definitiva, **com evidência** (ver DE-01) | evento documentado; sem evidência, o ativo continua `no_trade` |
| `no_prior_price` | há preço, mas não há preço válido anterior | estado da data anterior |

   Também propõe-se guardar a **idade do último preço** (pregões desde a última negociação), necessária para as regras de DE-01.
5. **Consequências.**
   - (c) perde a distinção entre falha de qualidade e fato de mercado, que o relatório de qualidade precisa.
   - (a) é exatamente o "ajuste caso a caso" que o projeto proíbe.
6. **Recomendação: (b).**
7. **Para aprovar:** DD-05 → DECIDED com as regras de classificação, o campo de evidência e a idade do preço. Como é aditivo, o enum atual não muda.

## 5. DE-05 — PBO e CSCV

### 5.1 rank/N ou rank/(N+1)
- **O original define rank/(N+1)** [verificado, p. 12]. O `pbo` 1.3.5 usa rank/N e por isso precisa substituir λ = +∞ por 6. É um desvio do pacote em relação ao artigo.
- **Consequências [verificado por álgebra e nos golden files]:**
  - φ conta λ ≤ 0, isto é, ω̄ ≤ 1/2.
    - Com rank/N: rank ≤ N/2.
    - Com rank/(N+1): rank ≤ (N+1)/2.
    - Para **N par sem ranks fracionários**, as duas regras coincidem.
    - Diferem para **N ímpar**: a configuração exatamente na mediana conta como sobreajuste em rank/(N+1) (λ = 0) e não conta em rank/N.
    - Diferem também quando um empate produz rank fracionário igual a (N+1)/2.
  - Nos 8 golden files (N = 20), φ é **idêntico** nas duas convenções.
  - A **distribuição dos logits** difere em todos os casos. Por exemplo, no `pbo_signal` todo λ vale 6 no pacote e ln 20 ≈ 2,996 no artigo.
  - rank/(N+1) dispensa o substituto arbitrário para +∞.
  - Degradação de desempenho e probabilidade de perda não dependem do rank.
- **Recomendação:** adotar **rank/(N+1)**, a definição do artigo.
  - A validação contra o `pbo` compara grandezas que não dependem da convenção: `n*`, `os_rank`, rn e rbn por partição.
  - Depois aplica a transformação de cada convenção e compara φ nos casos em que as convenções coincidem.
  - É preciso acrescentar um golden com **N ímpar**, para mostrar onde as convenções divergem.

### 5.2 T não divisível por S
- **O original exige blocos de dimensões iguais** e usa isso para justificar a simetria IS/OOS [verificado, p. 11 e 13].
- **Alternativas:**

| Alternativa | Consequência |
|---|---|
| (a) Núcleo rejeita T % S ≠ 0 | contrato explícito, igual ao do artigo e do pacote; a escolha fica para quem chama |
| (b) Descartar as T mod S observações mais antigas, por regra pré-registrada | perde ≤ S−1 observações (com S = 16, ≤ 15 pregões em milhares); a data final é preservada; determinístico |
| (c) Descartar as mais recentes | o mesmo, mas perde o período mais informativo para OOS |
| (d) Blocos desiguais (`array_split`) | usa todos os dados, mas IS e OOS ficam com tamanhos diferentes conforme a partição; afasta-se do artigo e cria uma assimetria pequena e difícil de justificar |
| (e) Escolher S que divida T | S passaria a depender dos dados: **rejeitada**, porque S deve ser pré-registrado |

- **Recomendação: (a) + (b).** O núcleo rejeita. A camada de experimento aplica a regra pré-registrada de descartar as T mod S observações mais antigas e registra no ledger as datas descartadas. Isso não é "ajustar janela para caber": a regra é fixada antes dos dados e é auditável.

### 5.3 Empates
- **Empates no ranking OOS:** posto médio, que é o padrão e coincide com o `pbo`.
- **Empate no máximo IS:** com retornos contínuos, só acontece se houver séries idênticas, o que indica configurações duplicadas. Essas duplicatas também inflariam N.
- **Recomendação:** detectar colunas idênticas em M antes da CSCV e tratar como erro de configuração (a grade é corrigida e registrada). Empates residuais IS são resolvidos pela primeira configuração na ordem canônica da grade (determinístico, igual ao `pbo`) e contados no relatório.

### 5.4 Métrica por bloco
- Sharpe com a convenção DE-02. Bloco com variância nula é erro explícito.

**Continua aberto em DE-05/DE-06:** o valor de S (o artigo discute S = 16 para cerca de 4 anos diários, p. 22) e o embargo nas fronteiras.

**Para aprovar:** DE-05.a (rank/(N+1)), DE-05.b ((a) + (b)), DE-05.c (empates).

## 6. DE-03 e DE-03.a — Número efetivo de tentativas

### 6.1 Estimadores

| Método | Definição | Fundamento e limitações | No caso de referência (4 blocos × 5, ruído 0,002) |
|---|---|---|---|
| Nyholt (2004) | 1 + (M−1)(1 − Var(λ)/M), com λ os autovalores da correlação | criado para SNPs em genética, mirando correções de Bonferroni/Šidák; usa só a variância dos autovalores, o que foi criticado em estruturas de blocos [literatura, não conferido: Li & Ji 2005] | 16 |
| Li & Ji (2005) | Σ f(\|λᵢ\|), com f(x) = 1{x ≥ 1} + (x − ⌊x⌋) | também genético; conta autovalores ≥ 1 mais as partes fracionárias | 7 |
| Gao et al. (2008) | nº de componentes que explicam C = 99,5% da variância | sensível ao ruído idiossincrático: componentes pequenos entram até atingir 99,5%; C é um parâmetro escolhido | 17 |
| Galwey (2009) | (Σ√λᵢ⁺)² / Σλᵢ⁺ | contínuo; zera autovalores negativos | 7 |
| DSR 2014, A.3 | ρ̂ + (1−ρ̂)M | do próprio artigo do DSR; resume a estrutura numa média; o artigo aponta as limitações | estrutura idealizada sem ruído: ρ̂ = 0,2105 → **16,0** [verificado por cálculo] |
| Clustering (ONC; López de Prado) | N = nº de clusters | [literatura, não conferido]; envolve k-means com reinicializações, portanto aleatoriedade e mais parâmetros | (esperado ≈ 4 [inferência]) |

### 6.2 Por que eles divergem tanto [inferência]
- Os métodos genéticos foram feitos para que uma correção de **Šidák** com M_eff controle a FWER, ou seja, a **cauda** da distribuição do máximo num α pequeno.
- O DSR precisa de outra coisa: o N tal que E[máximo de M correlacionados] ≈ E[máximo de N independentes], isto é, o **centro** da distribuição do máximo.
- São alvos diferentes, e nenhum desses estimadores foi validado para o segundo.
- No caso de referência, o número de fontes independentes "verdadeiro" é perto de 4. Li-Ji e Galwey (7) ficam mais perto disso; Nyholt, Gao e a fórmula do artigo (16–17) ficam longe.
- Esse caso é uma única estrutura sintética e **não** é evidência geral.

### 6.3 Influência no DSR [verificado]
O fator de SR₀ vale 1,387 com N = 7 e 1,801 com N = 16: o limiar de rejeição fica cerca de 30% mais alto. A escolha do estimador pode inverter a conclusão de "sobrevive / não sobrevive".

### 6.4 DE-03.a: `floor`
- No DSR, Φ⁻¹(1−1/N) está definido para todo N real > 1, então não há motivo estatístico para arredondar.
- O `floor` do `poolr` vem do uso em Bonferroni, que conta testes inteiros.
- Arredondar para baixo **sempre** reduz SR₀, ou seja, é anticonservador: de N = 7,99 para 7 o fator cai de 1,458 para 1,387 (−4,9%).
- **Recomendação:** não arredondar. A validação contra o `poolr` exige `floor(N_próprio) == N_poolr` exatamente, e os autovalores com rtol de 1e-9 (já congelados).

### 6.5 Recomendação para DE-03 (protocolo, não método único)
1. Pré-registrar um **conjunto** de estimadores: Li-Ji, Galwey, Nyholt, Gao e a fórmula A.3, mais M bruto como limite superior.
2. Reportar o DSR da melhor configuração para **todos** eles: é a análise de sensibilidade.
3. Escolher o estimador **principal** por critério fixado antes dos dados reais: o que melhor reproduz, **em simulação** com estruturas de correlação semelhantes às da grade de momentum, o E[máximo] observado e a taxa nominal de falsa descoberta (DE-07). A escolha é feita só com dados sintéticos, antes de qualquer dado real.
4. Clustering (ONC) fica como candidato adicional se a simulação mostrar que todos os estimadores por autovalores falham. Exige conferir a fonte original antes.

**Para aprovar:** DE-03.a (sem `floor`) e o protocolo de DE-03. O método principal **continua pendente** até a simulação.

## 7. DE-04 — O que entra em N e em V

### 7.1 Unidade de contagem [inferência a partir das definições do projeto]
- `run_id` identifica uma **execução**, e uma mesma configuração pode ter várias execuções.
- O fingerprint identifica a **configuração científica completa**, inclusive as sementes.
- A **tentativa**, que é o que N conta, é uma configuração distinta da família avaliada nos dados do estudo. Para uma regra determinística (momentum com dados reais), a semente não altera o retorno, então sementes diferentes não criam tentativas diferentes. Só criam quando o resultado depende da semente.
- N efetivo é uma **estimativa** a partir das tentativas (DE-03), não uma contagem.

### 7.2 Regras propostas

| Caso | Conta em M (tentativas)? | Entra em V? | Justificativa |
|---|---|---|---|
| Concluída | sim | sim | — |
| Falhou e depois concluiu com a mesma configuração | uma vez | sim (a concluída) | é a mesma tentativa |
| Falhou ou foi abortada e nunca concluiu | **sim** | não (não há SR) | excluir abriria uma brecha (abortar o que parece ruim); fica reportada à parte |
| Repetição idêntica | uma vez | sim | os resultados devem ser bit a bit iguais; divergência é erro de integridade |
| Sementes diferentes, regra determinística | uma vez | sim | não é tentativa nova |
| Sementes diferentes, procedimento estocástico | cada uma | sim | cada sorteio é uma oportunidade de seleção |
| Exploração com dados reais antes do pré-registro | **sim** | sim, se houver SR | é exatamente o viés de seleção estudado |
| Execuções com dados sintéticos | não | não | não informam a seleção nos dados reais |

- **Estado atual [verificado]:** o ledger tem 0 registros e nenhum dado real foi tocado.
- **Recomendação adicional:** proibir qualquer execução com dados reais antes do pré-registro. Assim a linha "exploração prévia" fica vazia por construção, e não por declaração.
- **Pendente:** se o estudo seleciona a melhor configuração **dentro** de cada faixa de liquidez, N é por faixa. Se compara faixas e reporta a melhor, N é o total. Isso deve ser fixado no pré-registro.

### 7.3 Variância V
- **O original [verificado, p. 8]** diz "the variance across the trials' estimated SR", sem ddof.
- ddof = 1 vs 0: com M = 100, V muda 1% e SR₀ cerca de 0,5%. A recomendação é **ddof = 1**, documentado.
- **Problema maior [inferência, com álgebra simples]:**
  - Com tentativas correlacionadas, a variância **entre tentativas** subestima a variância marginal de cada ŜR.
  - Num modelo equicorrelacionado, E[s²] = (1 − ρ)σ².
  - Se N já foi reduzido para N efetivo e V também vem reduzido pela correlação, a correlação é descontada duas vezes, o que é anticonservador.
- **Alternativas para V:**
  - (a) entre todas as M tentativas concluídas (leitura literal do artigo);
  - (b) entre representantes de clusters (só existe com clustering);
  - (c) V teórico sob H₀: com SR verdadeiro = 0, a variância assintótica do PSR (eq. 2) reduz-se a 1/(T−1).
- **Recomendação:**
  - Primário (a), fiel ao artigo, com ddof = 1.
  - (c) como sensibilidade.
  - A combinação (estimador de N, V) é avaliada na mesma simulação da §6.5, e o critério de sucesso 2 do projeto decide.
  - Não aprovar uma correção de V por (1 − ρ̂), porque seria fórmula inventada.

### 7.4 Domínio de N
- A expressão fica **negativa para N < 1,2836** e é indefinida em N = 1.
- O erro da aproximação em N = 2 é de −7,9%.
- **Recomendação:** o núcleo rejeita N < 2 com erro explícito. Abaixo disso, "máximo entre tentativas" não tem conteúdo e a aproximação não foi validada.
- Se um estimador produzir N efetivo < 2 para uma família, isso é um resultado a reportar (a família é, na prática, uma única tentativa), não um valor a corrigir.

**Para aprovar:** DE-04.a (tabela de contagem), DE-04.b (V: (a) primário com ddof = 1, (c) como sensibilidade, combinação decidida em simulação), DE-04.c (rejeitar N < 2).

## 8. DE-02.a — Amostra mínima para o PSR
1. **Problema.** O PSR usa uma aproximação assintótica, e com T pequeno o valor pode ser enganoso.
2. **O que a literatura oferece [verificado, 2012, p. 11]:**
   - O MinTRL (eq. 13) responde quantas observações são necessárias para **rejeitar** SR ≤ SR* com confiança 1−α, dadas as estimativas. É uma pergunta de **poder**, não de validade da aproximação.
   - A regra "T > 30" que o artigo cita é uma regra de livro-texto para o TLC. Ela não leva em conta caudas pesadas: a variância de b₂ depende do 8º momento, então a curtose estimada converge devagar [inferência].
3. **Alternativas.**
   - (a) Limiar fixo, por exemplo 30 ou 252. Seria arbitrário.
   - (b) Nenhum limiar além do domínio matemático, mais reportar MinTRL e calibrar por simulação.
   - (c) Limiar derivado de simulação para o T e os momentos do estudo.
4. **Consequências.**
   - No estudo, o PSR e o DSR são calculados sobre a amostra inteira, com milhares de pregões. A CSCV usa só ranks de SR, não PSR.
   - Por isso o risco prático de T pequeno é baixo. Ele aparece em sensibilidade e em simulação.
5. **Recomendação: (b).**
   - O núcleo exige T ≥ 2, série não constante e radicando > 0, com erro explícito em cada caso.
   - Calcular e reportar o **MinTRL** junto com PSR e DSR. A fórmula foi verificada e tem a âncora A7.
   - A adequação em amostra finita é verificada na simulação (DE-07): cobertura do PSR sob H₀ com caudas pesadas, no T do estudo.
   - Não há limiar inventado.
6. **Para aprovar:** DE-02.a fechada como (b), com o MinTRL incluído na Etapa A.

## 9. DE-01 (sub-decisões) — Deslistagem e dias sem negócio

Não bloqueia a Etapa A: o PSR e o DSR operam sobre matrizes de retornos já construídas. Bloqueia o motor de backtest e o pré-registro.

### 9.1 Retorno de deslistagem
- **Literatura (EUA) [literatura, não conferido]:** Shumway (1997) mostra que o CRSP omite retornos de deslistagem, e que as deslistagens por desempenho têm média próxima de −30% (NYSE/AMEX). Shumway & Warther (1999) reportam perdas maiores na Nasdaq. Esses valores **não** se transferem à B3.
- **B3 [inferência]:** as causas típicas são cancelamento de registro por OPA (preço conhecido), incorporação ou fusão (relação de troca), falência ou liquidação (valor próximo de zero) e mudança de ticker ou ISIN (não é deslistagem: é identidade, DD-02). O COTAHIST não informa causa nem valor terminal.
- **Alternativas e viés:**

| Alternativa | Viés |
|---|---|
| (A) Último preço negociado (retorno 0 até a saída) | para cima, sobretudo em falências |
| (B) −100% | para baixo, extremo |
| (C) Imputar a média por categoria, ao estilo Shumway | exige estimativas para a B3 que não temos |
| (D) Valor terminal documentado por evento (preço da OPA, relação de troca, liquidação) na tabela de eventos (DA-10), com (A) e (B) como limites de sensibilidade para casos sem documentação | mínimo, com custo manual |

- **Recomendação: (D)**, porque o universo é pequeno e líquido. O número de casos sai da auditoria DD-02 **antes** de fixar a regra. O valor terminal entra na data efetiva da saída, e entre a última negociação e a saída a posição fica marcada a preço defasado (9.2).

### 9.2 Dias sem negócio
- A distinção entre falta de negócio e falha de fonte segue as regras de DD-05 (seção 4).
- **Avaliação de posição mantida:** não existe preço. A marcação pelo último preço (preço defasado) é a convenção contábil usual e é a única que não fabrica informação.
  - O custo é conhecido: retornos zero seguidos de um salto reduzem a volatilidade medida e induzem autocorrelação [literatura, não conferido: Scholes & Williams 1977; Lo & MacKinlay 1990, negociação não sincronizada].
  - Isso infla o SR e afeta o PSR.
- **Recomendação:**
  1. Preço defasado **só para avaliar posições já mantidas**, sempre marcado e com a idade do preço.
  2. **Proibido** para sinal e para execução: não se entra, sai ou rebalanceia sem negócio no dia. A ordem é adiada para o próximo pregão com negócio daquele ativo, e essa regra entra na definição da estratégia.
  3. Contar e reportar a fração de posição-dias com preço defasado por configuração.
  4. O limiar de presença em pregões que torna um ativo elegível é DD-01 (pendente) e deve fazer desse caso uma exceção.
- O cálculo do sinal com lacunas (último preço ≤ data e idade máxima do preço) introduz um parâmetro, que é decisão de DE-01/DD-01, **pendente**.

**Para aprovar:** DE-01.d (deslistagem: (D) com limites de sensibilidade, condicionada à contagem de DD-02) e DE-01.e (dias sem negócio: regras 1–3).

## 10. DE-08 — Referência de validação do DSR
- Os parâmetros do exemplo de 2014 foram **conferidos visualmente** (seção 0.1) e o resultado foi reproduzido.
- Há ainda quatro âncoras do PSR no artigo de 2012 (A4–A7), vindas de outra fonte publicada.
- **Recomendação:**
  - Congelar A1–A7 como fixture de **constantes publicadas**, com página, PDF e SHA-256. Esse arquivo é diferente dos golden files gerados por R; é digitado, com citação, e revisado.
  - Tolerância = meia unidade da última casa publicada: 5e-5 para 0,9004 e 0,9505; 5e-4 para 0,982, 0,913 e 0,953; 5e-4 para 59,895.
  - É o erro de arredondamento do próprio artigo, não uma folga escolhida.
- **Para aprovar:** DE-08 → DECIDED.

## 11. Ordem de implementação condicionada
- **Etapa A** (PSR, DSR, MinTRL) depende de: DE-02, DE-02.a, DE-08, DE-03.a e DE-04.c.
  - A função do DSR recebe N e V como entradas, então não depende do estimador de N efetivo (DE-03) nem do escopo de N (DE-04.a).
  - Os estimadores de N efetivo, validados contra os golden do `poolr`, podem vir na sequência, porque DE-03.a basta para implementá-los.
- **Etapa B** (CSCV, PBO) depende de: DE-05.a, b e c. S e o embargo continuam abertos e entram como parâmetros obrigatórios, sem padrão.


---

## Adenda — consolidação (2026-11-06, 2ª sessão)

O desenvolvedor delegou ao agente o fechamento das decisões da Etapa A e das convenções futuras. O texto acima é preservado como estava. Abaixo estão o que foi decidido e o que **divergiu** da análise original.

| Item | Recomendação original | Resultado | Onde |
|---|---|---|---|
| DE-02 | aprovar com 2 ressalvas | **DECIDED**, como recomendado | PROJECT_DECISIONS.md; METHODOLOGY_CONVENTIONS.md §1 |
| DE-02.a | sem limiar; MinTRL reportado | **DECIDED**, como recomendado | §1.6.1–1.6.2 |
| DE-03.a | sem `floor` | **DECIDED**. Achado: Li-Ji é inteiro por construção (traço = M), então somar frações gerava 6,9999999999998 e `floor` daria 6, contra 7 do `poolr`. Implementado pela identidade exata | §3.2; `core.effective_trials` |
| DE-03 | conjunto pré-registrado; principal por simulação | protocolo **DECIDED**; principal **aberto**. Nenhum candidato passou no critério original (pior caso ≥ 104%) | §4.2, §4.5 |
| DE-04.a | tabela de contagem | **DECIDED**, como recomendado | §4.1 |
| DE-04.b | V entre tentativas (primário) + V_H0 (sensibilidade) | **divergiu**: a simulação mostrou que (N efetivo, V entre tentativas) é anticonservador (até −36%). Emenda 1: primário provisório (M bruto, V entre tentativas); sensibilidade (N efetivo, V_H0) | §4.5 |
| DE-04.c | rejeitar N < 2 | **DECIDED**, como recomendado | §1.7 |
| DE-05.a/b/c | rank/(N+1); núcleo rejeita T % S; descarte das mais antigas; duplicatas = erro | **DECIDED**, não implementado | §5.1–5.4 |
| S | (não analisado) | procedimento **DECIDED**; valor no pré-registro | §5.5 |
| DE-06 | (não analisado) | regra de construção **DECIDED** (purga desnecessária); embargo aberto com critério | §5.6 |
| DE-08 | tolerância = meia unidade impressa | **divergiu**: A6 a excede (0,95351 contra 0,953), porque as entradas foram impressas arredondadas. Tolerância = meia unidade da saída + propagação do arredondamento das entradas | §1.8; `tests/reference/test_published_anchors.py` |
| DA-03, DD-04 (CDI), DD-05, DE-01.d/e | — | **inalterados** (IN PROGRESS/PENDING): não são necessários para a Etapa A | — |

Outras correções desta sessão:
- **METHODOLOGY_CONVENTIONS.md §1.9** afirmava "PSR crescente em ŜR". Isso é falso em geral: z decresce em ŜR quando 1 + SR*·ŜR/2 < 0. O texto foi corrigido e há teste do contraexemplo.
- **Erro tipográfico no artigo da PBO:** a p. 11 diz que S = 16 gera "12, 780" combinações; o correto é C(16, 8) = 12.870.
- **Reprodutibilidade do estudo de simulação:** o fluxo aleatório de cada estrutura dependia da ordem do dicionário, e o JSON canônico do ledger reordena as chaves. O teste de reconstrução a partir do ledger detectou o defeito, que foi corrigido usando a ordem canônica.
- **Galwey** não é Lipschitz em λ = 0: autovalores nulos de arredondamento (≈1e-16) viram ≈1e-8 no resultado. Isso está documentado e tem teste com tolerância derivada.

---

## Adenda 2 — Etapa B (2026-11-06)
- **DE-05.a validada por caso discriminante:** o golden `pbo_odd` (N = 9) separa as convenções. Em S = 8 e 12, o φ do artigo (0,1571 e 0,1050) difere do φ do `pbo` (0,1429 e 0,0942), e a diferença é exatamente a fração de partições com rank OOS = (N+1)/2. A implementação própria segue o artigo (METHODOLOGY_CONVENTIONS.md §5.7).
- **Correção da §5.5:** a afirmação de que o custo da CSCV seria "trivial" supunha somas por bloco. A implementação direta mede 131 s com S = 16 e N = 144, e um gatilho de otimização ficou registrado.
- **DE-01 formalizada** (§6): resolve a ressalva 1 de DE-02 (long-only) e o *timing* de DD-04. Os insumos para S, N efetivo, embargo e o desenho não nulo estão em §6.7.

