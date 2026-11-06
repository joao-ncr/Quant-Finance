# Avaliação de stack — Frente E

Comparação de linguagens, bibliotecas e ferramentas para o projeto *Validação estatística de regras simples na B3*.
Data: 2026-11-06. Decisões derivadas: [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md).

**Legenda de evidência** (usada em todos os documentos de arquitetura):
- **[E]** evidência documental (documentação oficial, artigo, repositório, consultados nesta etapa ou na anterior);
- **[M]** prática de mercado observada (observação geral, não medida sistematicamente);
- **[R]** recomendação técnica (julgamento de engenharia, com justificativa);
- **[H]** hipótese que exige benchmark ou verificação antes de ser tratada como fato.

Nenhum número de desempenho neste documento foi medido. Estimativas de ordem de grandeza são **[H]** e estão especificadas para teste na seção 6.

---

## 1. Requisitos computacionais derivados do problema

### 1.1 Grandezas de referência (todas [H], a confirmar)
| Símbolo | Significado | Ordem de grandeza assumida | Depende de |
|---|---|---|---|
| A | ativos no universo | dezenas a ~100 | DD-01 |
| T | pregões na amostra | ~6.000–7.000 (pós-Plano Real até hoje) | DD-02, escolha de início |
| N | configurações da grade | 10² a 10³ | DE-01 |
| S | blocos CSCV | 8 a 16 → C(S,S/2) até 12.870 partições | DE-05 |
| R | réplicas de Monte Carlo | 10² a 10³ | DE-07 |

Consequências diretas (aritmética sobre as grandezas acima, não medição):
- Matriz de retornos T × N em float64: 7.000 × 1.000 × 8 B ≈ **56 MB**. Cabe em memória com folga.
- Matriz de correlação N × N: 1.000² × 8 B ≈ **8 MB**; decomposição espectral O(N³) ≈ 10⁹ operações.
- Painel de preços A × T: 100 × 7.000 ≈ 7 × 10⁵ valores. Pequeno.

### 1.2 Classificação das cargas
| Carga | Gargalo dominante | Observação |
|---|---|---|
| W1. Leitura do COTAHIST (largura fixa, 245 bytes/registro [E]) | I/O + parsing de texto | Executada raramente; incremental por ano. Volume total a medir (B1). |
| W2. Normalização, eventos corporativos, universo point-in-time | **Complexidade lógica**, não CPU | Risco principal é de correção (sobrevivência, look-ahead). |
| W3. Geração dos retornos das N estratégias | CPU, vetorizável | Somas acumuladas tornam o custo por configuração ≈ O(A·T). |
| W4. Sharpe, PSR, DSR | Trivial (O(T·N)) | Risco é numérico/convencional (curtose bruta vs. excesso), não de desempenho. |
| W5. N efetivo | CPU: O(N³) para autovalores; clusterização (ONC) potencialmente mais cara | ONC exige muitas execuções de k-means + silhueta [E: descrição secundária]. Medir (B4). |
| W6. CSCV/PBO | CPU; **depende da métrica** | Se a métrica for Sharpe, estatísticas suficientes por bloco (Σx, Σx²) reduzem cada partição a O(S·N). Se exigir a série completa (ordem, Omega, momentos superiores), cada partição custa O(T·N), cerca de 10³ vezes mais. **A escolha de DE-05 muda o custo computacional.** |
| W7. Monte Carlo sob a nula | CPU, **embaraçosamente paralelo** entre réplicas | Custo ≈ R × (W3 + W5 + W6) se a nula for gerada em preços. É a carga mais pesada do projeto [H]. |
| W8. Relatórios e sensibilidade | Trivial | |

**Conclusão de requisitos [R]:** o problema é de **volume moderado e alta exigência de correção**. Não há carga que, pelas estimativas, exija computação distribuída. O único ponto quente plausível é W7, e eventualmente W5/W6 dependendo de DE-03 e DE-05. Ele é paralelizável por processos numa única máquina. A escolha da stack deve otimizar primeiro **verificabilidade e reprodutibilidade**, depois desempenho, com gatilhos medidos.

### 1.3 Requisitos não funcionais
- **Rastreabilidade de tentativas como requisito científico:** cada configuração avaliada é uma tentativa que entra em N. O registro deve existir *antes* do primeiro backtest com dados reais e ser append-only.
- **Reprodutibilidade bit a bit** das simulações, independentemente do número de processos (exige gestão explícita de fluxos aleatórios).
- **Validação independente:** implementações próprias comparadas com referência externa (pacote R `pbo` para PBO; `poolr::meff` para N efetivo por autovalores [E]; DSR sem pacote de referência, ver DE-08).
- **Point-in-time:** reconstruir a elegibilidade de cada ativo em cada data.
- **Projeto individual:** custo de manutenção e quantidade de ambientes contam como requisito.

---

## 2. Comparação de linguagens

Sem notas numéricas. Cada célula traz o julgamento e o motivo.

| Critério | Python | R | Julia | C++ | Rust | Java |
|---|---|---|---|---|---|---|
| 1. Bibliotecas estatísticas | Maduras e amplas (SciPy, statsmodels, scikit-learn) [E] | **As mais maduras** em estatística clássica; tem `pbo` e `rb3` [E] | Boas (StatsBase, Distributions), ecossistema menor [E] | Pouco além de álgebra linear; estatística seria implementada à mão | Imaturo (statrs, ndarray-stats) [R] | Limitado (Commons Math, Smile) |
| 2. Séries temporais financeiras | Boas (pandas, Polars) | Muito boas (xts, PerformanceAnalytics) | Razoáveis (TimeSeries.jl) | Nenhuma de alto nível | Polars (núcleo em Rust) | Fracas |
| 3. Processamento matricial | NumPy/BLAS: padrão de fato | Nativo, BLAS | Nativo, excelente | Eigen, excelente | ndarray, razoável | Fraco |
| 4. Desempenho em laços/simulação | Lento em Python puro; rápido se vetorizado ou com Numba | Lento em laços; Rcpp para velocidade (introduz C++) | **Rápido sem segunda linguagem** | Máximo | Máximo | Bom (JIT) |
| 5. Paralelismo | Processos (GIL); simples para tarefas independentes | `future`/`parallel`, adequado | Threads e processos nativos | Total, alto custo de engenharia | Rayon, seguro e eficiente | Maduro |
| 6. Memória | Adequada para a escala (1.1) | Cópias implícitas podem pesar | Boa | Total controle | Total controle | Overhead de JVM |
| 7. Ingestão de dados financeiros BR | Parser próprio simples (layout público) [E]; clientes SGS existem | **`rb3` lê COTAHIST diretamente** [E] | Parser próprio | Parser próprio | Parser próprio | Parser próprio |
| 8. Armazenamento (Parquet/Arrow/DuckDB) | Excelente (PyArrow, Polars, DuckDB) | Bom (arrow, duckdb) | Bom (Arrow.jl) | Bom (Arrow C++) | Excelente (Polars, arrow-rs) | Bom (Arrow Java) |
| 9. Reprodutibilidade | Boa com lockfile (uv, pixi, conda) | Boa com `renv` | Boa (Project/Manifest.toml nativos) | Depende do build | Boa (Cargo.lock) | Boa (Maven/Gradle) |
| 10. Testabilidade/engenharia | Muito boa (pytest, hypothesis, mypy, ruff) | Boa (testthat), tipagem fraca | Boa (Test stdlib), tooling menor | Boa, verbosa | **Excelente** (tipos, cargo test) | Muito boa |
| 11. Manutenção (1 pessoa) | Alta | Alta | Média (latência de compilação, ecossistema menor) | Baixa (custo alto por mudança) | Média-baixa para pesquisa iterativa | Média-baixa |
| 12. Aderência ao mercado quant | Pesquisa: dominante [M] | Comum em estatística/econometria, menor em quant dev [M] | Nicho [M] | Dominante em execução de baixa latência e bibliotecas de precificação [M] | Crescente em infraestrutura [M] | Comum em sistemas de trading/infra, raro em pesquisa [M] |
| 13. Documentação/comunidade | Muito ampla | Muito ampla (estatística) | Média | Ampla, menos para estatística | Média | Ampla, menos para estatística |
| 14. Integração com outras linguagens | Excelente (Rscript/rpy2, PyO3, pybind11, Arrow) | Boa (reticulate, Rcpp) | Boa (PythonCall, RCall) | Base de tudo | PyO3/maturin, boa | JNI, pesada |

### 2.1 Análise por linguagem
- **Python.** Ecossistema completo para os oito domínios do problema (dados, matrizes, estatística, clusterização para ONC, paralelismo por processos, testes por propriedades, gráficos). Fraqueza real: laços em Python puro, mitigável por vetorização (W3, W6 com estatísticas por bloco) e, se medido como necessário, Numba. **Não é escolhido por familiaridade**, mas porque é a única opção que cobre todos os domínios sem segunda linguagem *e* tem o melhor ferramental de engenharia para validação.
- **R.** Candidato sério: tem as duas peças específicas do domínio (`rb3`, `pbo`) e excelente estatística. Contra: (a) o núcleo do projeto (DSR, N efetivo, motor de momentum sobre grade, Monte Carlo) será implementação própria de qualquer forma, já que `pbo` não faz DSR; (b) laços rápidos exigiriam Rcpp, ou seja, C++; (c) ferramental de tipagem e testes por propriedade mais fraco; (d) usar `pbo` como implementação *e* como referência eliminaria a validação independente. **Papel recomendado: instrumento de validação** (gerar resultados de referência e conferir a ingestão), fora do caminho de execução.
- **Julia.** Melhor argumento técnico contra Python: velocidade sem segunda linguagem, ótimo para Monte Carlo. Contra: ecossistema de dados e estatística menor, sem `pbo` nem `rb3`, latência de compilação, menor aderência ao mercado [M]. Só se justificaria se os benchmarks mostrassem W7 inviável em Python + Numba, o que é [H] e não esperado pela seção 1.
- **C++.** Desempenho máximo, mas o custo de engenharia por mudança é incompatível com pesquisa iterativa individual. QuantLib é voltada a precificação de derivativos e curvas [E]; nada no escopo usa isso. Rejeitado como linguagem principal; possível apenas como núcleo de extensão se um gatilho de desempenho for atingido (improvável).
- **Rust.** Segurança e desempenho excelentes; Polars é escrito em Rust. Estatística imatura e ciclo de iteração mais lento para pesquisa. Mesmo papel potencial de C++ (extensão via PyO3), com menor risco de memória. Não justificado hoje.
- **Java.** Ecossistema científico fraco para este problema. Rejeitado.
- **TypeScript/JavaScript.** Só faria sentido para visualização interativa ou aplicação web, que não são requisito da primeira versão. Rejeitado.

---

## 3. Bibliotecas e ferramentas por ecossistema

Estado de manutenção e licença: **[E]** quando consultado nesta etapa; **[E-prévio]** quando é conhecimento consolidado não reconferido hoje (conferir o arquivo LICENSE ao adotar).

### 3.1 Python
| Biblioteca | Papel possível | Licença | Situação | Avaliação |
|---|---|---|---|---|
| NumPy | Matrizes, RNG (`Generator`, `SeedSequence`) | BSD [E-prévio] | Ativa | **Essencial.** `SeedSequence.spawn` resolve fluxos aleatórios independentes por réplica. |
| SciPy | `norm.cdf/ppf/logcdf`, momentos, `special` | BSD [E-prévio] | Ativa | **Essencial.** Armadilha: `scipy.stats.kurtosis` retorna **excesso** por padrão (`fisher=True`); o PSR usa curtose bruta. Testar explicitamente. |
| Polars | ETL do COTAHIST, joins as-of para point-in-time | MIT [E-prévio] | Ativa | **Recomendado para a camada de dados.** Lazy, multi-thread, `join_asof`. Histórico de mudanças de API antes da 1.0. |
| pandas | Alternativa de ETL | BSD [E-prévio] | Ativa | Viável; `read_fwf` tende a ser lento [H, B1]. Não usar duas bibliotecas de DataFrame. |
| PyArrow | Parquet/Arrow | Apache-2.0 [E-prévio] | Ativa | Dependência indireta (Polars lê/escreve Parquet nativamente). |
| DuckDB | SQL analítico sobre Parquet | MIT [E-prévio] | Ativa | **Opcional**, para exploração ad hoc. Não é necessário no caminho principal se Polars cobrir os joins. |
| Numba | JIT de laços quentes | BSD [E-prévio] | Ativa | **Adiado por gatilho** (B2/B3/B5). Atenção: o RNG dentro de kernels Numba tem estado próprio. Gerar aleatórios fora dos kernels para manter reprodutibilidade. |
| statsmodels | Erros HAC/Newey-West, autocorrelação | BSD [E-prévio] | Ativa | Opcional, condicionado a DE-06. |
| scikit-learn | k-means + silhueta (ONC) | BSD [E-prévio] | Ativa | Opcional, condicionado a DE-03 adotar ONC. |
| pytest + hypothesis | Testes unitários e por propriedade | MIT / MPL-2.0 [E-prévio] | Ativas | **Essenciais** para validar propriedades estatísticas. |
| matplotlib | Figuras de publicação | Licença própria (estilo PSF/BSD) [E-prévio] | Ativa | **Recomendado**; controle fino para figuras acadêmicas. |
| ruff, mypy | Lint/format, tipos | MIT [E-prévio] | Ativas | Recomendados. |
| Cliente SGS (`python-bcb`, `sgs`, `bcbpy`) | CDI (série 12 [E]) | a verificar | `python-bcb` **não confirmado** na busca de hoje; `bcbpy` relata limite de 10 anos por consulta no SGS desde mar/2025 [E: README de terceiro] | Decisão em DA-14: verificar cliente; alternativa é chamar a API SGS diretamente e paginar por janela. |
| pypbo, skepsis, fynance | PBO em Python | a verificar | `pypbo` fixa statsmodels 0.8.0 (antigo) [E] | **Não adotar** como dependência. Podem servir como terceira referência informal. |

### 3.2 R
| Pacote | Papel | Situação | Avaliação |
|---|---|---|---|
| `pbo` | Referência de PBO/CSCV | v1.3.5, MIT, última publicação 2022 [E] | **Usar como oráculo de validação.** Parado desde 2022, mas o algoritmo é estável. Conferir como trata T não divisível por S e empates. |
| `rb3` | Leitura do COTAHIST | Ativo; API mudou entre versões (`cotahist_get` vs. `fetch_marketdata`) [E] | **Usar como checagem cruzada** do parser próprio (contagens e preços em anos amostrados). |
| `poolr` (`meff`) | N efetivo por autovalores: Nyholt, Li e Ji, Gao, Galwey [E] | a verificar | Referência para DE-03 (métodos espectrais). |
| data.table, xts, quantmod, PerformanceAnalytics | ETL, séries, métricas | Maduros [E-prévio] | Não necessários se o R ficar restrito à validação. |

### 3.3 Julia, C++, Rust, Java (resumo)
- **Julia:** DataFrames.jl, TimeSeries.jl, StatsBase.jl e Distributions.jl são maduros o bastante [E-prévio]. JuMP.jl é para otimização matemática, **sem uso neste problema**.
- **C++:** Eigen (MPL-2.0) e QuantLib (BSD modificada) [E-prévio]. QuantLib não endereça nenhuma carga do escopo.
- **Rust:** Polars, ndarray, Rayon. Úteis apenas como extensão nativa via PyO3, se acionado gatilho.
- **Java:** Apache Arrow Java existe [E-prévio]; sem vantagem para o problema.

---

## 4. Arquiteturas híbridas: custo vs. benefício

| Combinação | Benefício concreto | Custo | Veredito [R] |
|---|---|---|---|
| R na ingestão (`rb3`) + Python no restante | Evita escrever parser | Dois ambientes no caminho crítico; acoplamento à API do `rb3`, que já mudou [E]; serialização intermediária | **Não.** Parser de largura fixa com layout público é pequeno e testável. `rb3` vira checagem cruzada. |
| R só para validação (`pbo`, `rb3`, `poolr`) | Validação independente por implementação de terceiros | Ambiente R isolado, executado raramente; troca por arquivos (Parquet/CSV) | **Sim.** Custo contido: os resultados de referência viram *golden files* versionados, e os testes do dia a dia rodam só em Python. |
| Python + Numba | Acelera laços sem trocar de linguagem | Restrições de tipos, depuração mais difícil, RNG separado | **Por gatilho** (benchmarks B2, B3, B5). |
| Python + Rust/C++ (PyO3/pybind11) | Máximo desempenho | Toolchain de compilação, build no Windows, depuração entre linguagens | **Não**, salvo se Numba não bastar após medição. |
| Python + Julia | Simulação rápida | Dois runtimes, ponte de dados, menor aderência | **Não**, salvo gatilho. |

---

## 5. Práticas de mercado e ferramentas de referência

### 5.1 Ferramentas avaliadas
| Ferramenta | O que é | Situação | Usar? | Conceito aproveitável |
|---|---|---|---|---|
| vectorbt | Backtester vetorizado em NumPy/Numba | Apache-2.0 **com Commons Clause** ("fair-code"); edição comunitária do vectorbt PRO; v0.28.0 no PyPI [E] | Não como dependência | Broadcast da grade de parâmetros sobre matrizes; é a mesma ideia do motor W3. |
| Backtrader | Backtester orientado a eventos | Manutenção não confirmada nesta busca | Não | Nada específico: o modelo orientado a eventos é lento para grades de 10³ configurações e esconde convenções de execução. |
| Zipline (zipline-reloaded) | Backtester orientado a eventos (ex-Quantopian) | Mantido por Stefan Jansen; v3.2.0 [E] | Não | *Pipeline API* (fatores calculados por data, sem look-ahead) e uso de calendário de pregões. |
| Qlib (Microsoft) | Plataforma de pesquisa quant orientada a ML | Ativa; licença MIT segundo agregador [E, não confirmado no repositório] | Não | Separação dados → modelo → backtest → análise; registro de experimentos. Pesada e orientada a ML/mercado chinês para este escopo. |
| mlfinlab (Hudson & Thames) | Implementações de López de Prado | **Proprietária ("all rights reserved"), não open source** [E] | **Não** (licença incompatível com publicação aberta) | Implementar a partir dos artigos e livros originais. |
| QuantLib | Precificação de derivativos, curvas, calendários | Ativa | Não | Fora do escopo. |
| DuckDB / Apache Arrow | Motor SQL embutido / formato colunar | Ativos | Arrow via Parquet sim; DuckDB opcional | Dados colunares imutáveis, consultados no lugar. |
| MLflow | Tracking de experimentos (params, métricas, artefatos, UI) | Ativo, Apache-2.0 [E-prévio] | Não no início; reavaliar por gatilho | Modelo run/params/metrics/artifacts e proveniência de código. |

### 5.2 Práticas institucionais vs. necessidade do projeto
| Prática institucional [M] | Necessária aqui? | Forma proporcional [R] |
|---|---|---|
| Data lake versionado, dados point-in-time, *security master* | Sim, em versão reduzida | Parquet imutável + manifesto de hashes + tabela de identidade de ativos (ISIN/ticker) + tabela de elegibilidade por data. |
| Registro de toda pesquisa e tentativa (López de Prado recomenda registrar todas as tentativas para corrigir por seleção [E]) | **Sim, é o núcleo científico** | Ledger append-only próprio, encadeado por hash. |
| Validação independente de modelos (equipe separada) | Sim, adaptada | Implementações de referência de terceiros (R) + testes de resposta conhecida. |
| Plataformas de tracking (MLflow, W&B), orquestradores (Airflow, Dagster, Prefect) | Não | CLI Python com etapas idempotentes e endereçadas por hash. |
| Clusters, filas, microsserviços | Não | Paralelismo por processos em uma máquina. |
| C++ para execução de baixa latência | Não (não há execução) | — |
| Revisão de código e CI | Sim, leve | Testes locais; CI opcional quando houver repositório remoto. |

---

## 6. Benchmarks propostos (não executados)

Objetivo: resolver incertezas que mudam decisões. Todos usam cargas reais ou sintéticas com as dimensões da seção 1.1. Registrar hardware, versões e semente. Cada benchmark tem um critério de decisão; se a dúvida não existir mais, o benchmark não é executado.

| ID | Pergunta que resolve | Operação | Variantes | Medidas | Decide |
|---|---|---|---|---|---|
| B1 | Parser próprio é correto e eficiente? | Ler 1 ano recente e o histórico completo do COTAHIST; filtrar ações à vista | Polars (parser próprio) vs. pandas `read_fwf` vs. `rb3` | Tempo, RAM de pico, **contagem de registros e preços idênticos entre implementações** | DA-09, DA-05 |
| B2 | NumPy vetorizado basta para gerar a grade? | Retornos de N configurações de momentum sobre o painel A × T | NumPy puro vs. Numba; N ∈ {100, 1.000} | Tempo, RAM, igualdade numérica | DA-07 (Numba) |
| B3 | Custo real da CSCV conforme a métrica | PBO sobre matriz T × N | Estatísticas por bloco vs. recálculo por linhas; S ∈ {8, 12, 16}; NumPy vs. Numba; **mesma matriz no R `pbo`** | Tempo, RAM, concordância com R | DA-07, DE-05 |
| B4 | Custo do N efetivo | Correlação + `eigh`; ONC (k-means + silhueta) | N ∈ {100, 500, 1.000} | Tempo, estabilidade entre sementes | DE-03, DA-07 |
| B5 | Monte Carlo cabe no orçamento? | Pipeline completo sob a nula, R pequeno (ex.: 20–50) e extrapolação linear | 1..k processos; com e sem Numba | Tempo por réplica, escala com processos, **resultado bit a bit igual para qualquer nº de processos** | DA-07, DA-11 |
| B6 | (Só se B5 estourar o orçamento após Numba) Outra linguagem compensa? | Kernel mais caro identificado em B5 | Numba vs. Julia ou Rust (PyO3) | Tempo, esforço de integração | Reabrir DA-02 |

O orçamento computacional aceitável (ex.: tempo máximo de um estudo completo) é **decisão do desenvolvedor** (DA-07) e deve ser fixado antes de B5.

---

## 7. Propostas de stack e decisão

| Dimensão | **A. Enxuta** | **B. Orientada a desempenho** | **C. Híbrida / orientada ao mercado** |
|---|---|---|---|
| Linguagem principal | Python | Julia (variante B′: Python + kernels Rust via PyO3) | Python |
| Complementares | R só para validação | R só para validação | R na ingestão (`rb3`) e validação; C++/Rust para kernels |
| Científicas | NumPy, SciPy; scikit-learn e statsmodels sob demanda | StatsBase, Distributions, LinearAlgebra | NumPy, SciPy, scikit-learn, statsmodels, Numba |
| Dados | Polars + Parquet | DataFrames.jl + Arrow.jl/Parquet | pandas/Polars + Arrow + DuckDB |
| Armazenamento | Parquet + manifestos SHA-256 | Idem | Parquet + DuckDB + DVC |
| Tracking | Ledger JSONL próprio, encadeado por hash | Ledger próprio | MLflow (+ ledger para N) |
| Testes | pytest + hypothesis + golden files do R | Test stdlib + golden files do R | pytest + hypothesis + CI |
| Relatórios | matplotlib + documento gerado por código | Makie/Plots.jl | Quarto + MLflow UI |
| Execução | Uma máquina, `ProcessPoolExecutor` | Uma máquina, threads/Distributed | Orquestrador (Prefect/Dagster), possível cluster |
| Vantagens | Menos peças; validação independente; maior aderência a pesquisa quant [M]; evolução por gatilho | Simulação rápida sem segunda linguagem | Parece com infraestrutura institucional [M]; UI de experimentos |
| Desvantagens | Laços lentos sem vetorização/Numba | Ecossistema menor; sem `pbo`/`rb3`; latência de compilação; menor aderência [M] | Muitas peças; dois runtimes no caminho crítico; a infraestrutura distrai da ciência |
| Manutenção | Baixa | Média | Alta |
| Limitações | Pode exigir Numba em W7 [H] | Sem ganho comprovado para cargas de volume moderado [H] | Complexidade sem gargalo que a justifique |

### 7.1 Recomendação principal: **Stack A (enxuta)**
- Atende a todas as cargas estimadas (§1) com uma linguagem de execução, mantém a validação independente via R e tem o menor custo de manutenção para um projeto individual.
- O desempenho fica tratado por gatilhos medidos (B2–B6), não por antecipação: Numba primeiro, linguagem nativa só se Numba não bastar.
- Conceitos da Stack C são adotados sem as ferramentas: proveniência e tracking (ledger próprio com campos inspirados no MLflow), versionamento por hash (equivalente leve ao DVC), formato colunar (Arrow/Parquet).

### 7.2 Não escolhidas e por quê
- **Julia (B):** vantagem real em laços numéricos, mas sem carga comprovadamente limitada por CPU que a justifique, e sem as referências de domínio.
- **C++, Rust:** custo de engenharia sem gargalo medido; reservadas para B6.
- **Java, TypeScript:** sem vantagem para o problema; não há interface a construir.
- **R como linguagem principal:** eliminaria a validação independente e exigiria Rcpp para laços.
- **MLflow, DVC, DuckDB obrigatório, orquestradores:** resolvem problemas de equipe e de escala que o projeto não tem; ficam como evolução por gatilho (ARCHITECTURE §6).
- **Frameworks de backtesting:** ver §5.1 e DA-12.

---

## 8. Fontes consultadas
- Layout COTAHIST (B3): https://www.b3.com.br/data/files/33/67/B9/50/D84057102C784E47AC094EA8/SeriesHistoricas_Layout.pdf
- Manual de Precificação de Eventos Corporativos (B3): https://www.b3.com.br/data/files/2C/B0/39/4D/AFA538101E311E28AC094EA8/Manual%20de%20Eventos%20Corporativos.pdf
- `rb3`: https://docs.ropensci.org/rb3/ · https://packages.oit.ncsu.edu/cran/web/packages/rb3/refman/rb3.html
- `pbo` (CRAN): https://search.r-project.org/CRAN/refmans/pbo/html/pbo.html · https://www.rdocumentation.org/packages/pbo/versions/1.3.5
- `poolr::meff`: https://rdrr.io/pkg/poolr/man/meff.html · Li e Ji (2005), *Heredity* 95(3): https://www.nature.com/articles/6800717.pdf
- ONC / False Strategy Theorem (fontes secundárias): https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio · López de Prado e Lewis (2019), *Quantitative Finance* 19(9)
- pypbo: https://github.com/esvhd/pypbo · skepsis: https://pypi.org/project/skepsis/
- vectorbt: https://vectorbt.dev/terms · https://pypi.org/project/vectorbt
- zipline-reloaded: https://pypi.org/project/zipline-reloaded/3.2.0/ · https://github.com/stefan-jansen/zipline-reloaded/releases
- mlfinlab (licença): https://awesome.ecosyste.ms/projects/github.com%2Fhudson-and-thames%2Fmlfinlab · https://pypi.org/project/famlafl/0.1.8/
- Qlib: https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/ · https://arxiv.org/pdf/2009.11189.pdf
- pixi (Python + R com lockfile único): https://pydevtools.com/handbook/reference/pixi/ · https://prefix.dev/blog/pixi_a_fast_conda_alternative
- Clientes SGS: https://pypi.org/project/bcbpy/ · https://pypi.org/project/sgs/2.0.3/
