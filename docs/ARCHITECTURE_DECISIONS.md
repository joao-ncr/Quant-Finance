# Decisões arquiteturais (ADRs) — Frente E

Registro detalhado das decisões arquiteturais. Usa os mesmos identificadores `DA-xx` de [PROJECT_DECISIONS.md](../PROJECT_DECISIONS.md), que mantém o resumo e as decisões estatísticas (`DE-xx`, `DD-xx`).

**Status:** `PENDING` (sem proposta) · `IN PROGRESS` (proposta feita, **aguardando aprovação do desenvolvedor**) · `DECIDED` (aprovada, com data).
Nesta etapa as propostas estão em `IN PROGRESS`: a escolha técnica foi feita e justificada, mas só vira `DECIDED` com aprovação.

Legenda de evidência: [E] documental · [M] prática de mercado · [R] recomendação técnica · [H] hipótese que exige benchmark (ver [STACK_EVALUATION.md](STACK_EVALUATION.md)).

---

## DA-01 — Organização: monólito modular em camadas
- **Status:** IN PROGRESS
- **Contexto:** Projeto individual e acadêmico; cargas de volume moderado (STACK_EVALUATION §1).
- **Problema:** Separar dados, núcleo matemático, orquestração, validação e relatório sem infraestrutura desproporcional.
- **Alternativas consideradas:** (a) scripts soltos + notebooks; (b) **pacote Python único em camadas com CLI**; (c) serviços separados por domínio; (d) orquestrador de workflows (Airflow/Dagster/Prefect).
- **Critérios:** testabilidade do núcleo, rastreabilidade, custo de manutenção individual, reversibilidade.
- **Decisão:** (b). Camadas `data`, `core`, `experiments`, `reporting`, mais `tests/` e `validation/r/`. Núcleo sem I/O.
- **Justificativa:** (a) dificulta testar e registrar tentativas de forma garantida; (c) e (d) resolvem problemas de escala e operação que o projeto não tem [R].
- **Consequências:** + um só ambiente e um só deploy local; + núcleo testável isoladamente. − A disciplina de camadas depende de revisão (não há fronteira física).
- **Reavaliar se:** surgirem execuções agendadas recorrentes ou múltiplos usuários.

**Registro de implementação (2026-11-06, Etapa B):** a CSCV/PBO seguiu as camadas propostas.
- `core.cscv` é puro: sem I/O nem RNG; o teste de arquitetura cobre o arquivo.
- `experiments.cscv_input` contém a regra que depende de pré-registro (descarte DE-05.b) e a serialização para o ledger.
- Nenhuma camada nova foi criada; `ReturnsMatrix` e `sharpe_ratio` foram reutilizados.
- **Controle de versão:** desde esta etapa, cada etapa concluída é commitada localmente, sem push (pedido do desenvolvedor). Commit de referência da fundação e da Etapa A: `d7ba543`.

## DA-02 — Linguagem principal e complementares
- **Status:** IN PROGRESS
- **Contexto:** Comparação completa em STACK_EVALUATION §2–4.
- **Problema:** Escolher a linguagem pela adequação ao problema, não pela familiaridade.
- **Alternativas consideradas:** Python; R; Julia; C++; Rust; Java; híbridos (R na ingestão, Python + Numba, Python + Rust, Python + Julia).
- **Critérios:** cobertura dos oito domínios do problema sem segunda linguagem; ferramental de validação; desempenho suficiente nas cargas estimadas; manutenção; aderência ao mercado.
- **Decisão:** **Python** como linguagem única do caminho de execução. **R apenas como instrumento de validação**, em ambiente isolado (`pbo`, `rb3`, `poolr`). Nenhuma linguagem compilada na primeira versão.
- **Justificativa:** Python cobre ETL colunar, álgebra linear, estatística, clusterização (ONC), paralelismo por processos, testes por propriedade e figuras [E]. As cargas estimadas não exigem linguagem compilada [H, B2–B5]. R tem os pacotes de referência, mas usá-los como implementação eliminaria a validação independente, e o núcleo (DSR, N efetivo, motor de momentum) seria escrito à mão de qualquer forma.
- **Consequências:** + um runtime de produção; + validação independente via R. − Dois ambientes para manter (o R é pequeno e raramente executado). − Laços quentes podem exigir Numba (DA-07).
- **Reavaliar se:** B6 indicar que Numba não basta para o orçamento definido em DA-07.
- **Estado do ambiente R (2026-11-06):**
  - R 4.6.2 instalado por usuário (instalador oficial do CRAN, SHA-256 conferido com o manifesto do winget).
  - `validation/r/` com `renv` 1.3.1 em modo `explicit`, `DESCRIPTION` e `renv.lock`; `pbo` 1.3.5 e `poolr` 1.2-0 instalados.
  - Golden files gerados e congelados com manifest SHA-256.
  - `rb3` só será instalado na etapa do parser.
  - Pipeline em três passos (entradas em Python → referência em R → congelamento em Python), descrito em `validation/r/README.md`. A suíte Python não executa R.

## DA-03 — Registro de tentativas (ledger)
- **Status:** IN PROGRESS. Formato adotado pelo desenvolvedor como proposta principal (2026-11-06); **implementado e testado** (`frente_e.experiments`); aguarda aprovação formal para `DECIDED`.
- **Contexto:** O log de tentativas é insumo do cálculo de N (requisito científico, não administrativo).
- **Problema:** Garantir que toda tentativa seja registrada, inalterável e reconstruível, inclusive falhas.
- **Alternativas consideradas:** (a) **JSONL append-only encadeado por SHA-256**, versionado no Git; (b) SQLite com gatilhos que impedem UPDATE/DELETE; (c) MLflow (store local); (d) DuckDB.
- **Critérios:** imutabilidade verificável, legibilidade, versionamento, simplicidade, recuperação após falha, consulta.
- **Decisão:** (a). Banco de dados descartado: não resolve nada que o JSONL não resolva para um escritor único.
- **Identidade** (`experiments/identity.py`):
  - `experiment_id`: nome estável do estudo lógico, escolhido por pessoa (`[a-z0-9._-]`, 3–80 caracteres).
  - `run_id`: `run-<uuid4>`, uma execução individual. Nunca é derivado da configuração.
  - `configuration_fingerprint`: SHA-256 da forma canônica de {versão do fingerprint, `parameters` (família, parâmetros da estratégia, períodos, métrica de seleção, configuração estatística), `universe_reference`, `preregistration_reference` (caminho, versão, hash), `random_seeds`}.
  - Ficam **fora** do fingerprint: horário, `experiment_id`, `run_kind`, ambiente, revisão do código e referências de dados de entrada (todos registrados à parte).
  - Mesma configuração com sementes diferentes gera fingerprints diferentes. Uma repetição idêntica tem o mesmo fingerprint e um `run_id` novo, e deve reproduzir o resultado quando dados, código e ambiente registrados também coincidirem.
- **Contrato** (`experiments/records.py`, schema v1): exatamente os campos `schema_version`, `sequence`, `experiment_id`, `run_id`, `run_kind`, `configuration_fingerprint`, `timestamp_utc`, `status`, `parameters`, `universe_reference`, `preregistration_reference`, `random_seeds`, `software_environment`, `code_revision`, `input_data_references`, `results_reference`, `error_information`, `previous_record_hash`, `record_hash`.
  - Uma execução = um registro `started`, com a configuração completa, e no máximo um registro terminal (`completed`, `failed` ou `aborted`).
  - O registro terminal repete só a identidade, com os campos de configuração em `null`, para manter o JSONL leve.
  - `run_kind` ∈ {development, synthetic, study}.
- **Serialização canônica** (`experiments/canonical.py`): JSON UTF-8, chaves ordenadas, sem espaços, sem NaN/Infinity, floats em `repr` (round-trip exato). Escalares NumPy inteiros/booleanos são rejeitados, e `1` ≠ `1.0`.
- **Integridade:**
  - `record_hash` = SHA-256 da forma canônica do registro sem o próprio hash.
  - `previous_record_hash` encadeia os registros; o primeiro usa `"0"×64`.
  - `sequence` = posição da linha, e cada linha precisa estar exatamente em forma canônica.
  - O verificador detecta: alteração, remoção ou reordenação no meio do arquivo; hash inválido; `previous_record_hash` inválido; `run_id` duplicado; registro terminal sem `started`; dois registros terminais; identidade divergente entre `started` e o terminal; versão de schema não suportada; JSON inválido; linha não canônica; última linha truncada.
- **Limitações (documentadas e testadas):**
  - Quem reescreve o arquivo inteiro e recalcula a cadeia passa pela verificação isolada.
  - A remoção das últimas linhas também passa.
  - As duas são detectadas comparando com uma **âncora externa** `N:SHA256` (`frente-e ledger head` / `verify --expected-head`), que deve ser publicada fora do arquivo (commit do Git, relatório). O hash detecta adulteração, mas não impede reescrita.
- **Escrita e recuperação:**
  - Trava do SO num arquivo `.lock` (escritor único; liberada se o processo morrer).
  - A cadeia inteira é verificada ao abrir.
  - Antes de cada append, o tamanho do arquivo é conferido, para detectar escrita externa.
  - Cada linha é escrita numa única chamada, seguida de `flush` + `fsync`.
  - **Última linha truncada:** o escritor se recusa a abrir. `ledger recover-tail` move os bytes parciais para `ledger.jsonl.torn-<timestamp>` e trunca, mas só se o prefixo for íntegro. Nunca é automático.
  - **Execução interrompida:** fica como `started` sem registro terminal; é listada por `ledger incomplete` e encerrada com `ledger abort <run_id> --reason`.
  - Exceção dentro de `LedgerWriter.run` vira `failed`, com tipo, mensagem e traceback; `KeyboardInterrupt`/`SystemExit` viram `aborted`. Em ambos os casos a exceção é repassada.
  - Uma execução que termina sem resultados vira `failed` (`MissingResults`).
- **Artefatos** (`experiments/artifacts.py`):
  - Resultados volumosos ficam fora do JSONL, num armazém endereçado por conteúdo (`<sha[:2]>/<sha><sufixo>`), com escrita atômica e verificação de hash na leitura.
  - Arrays são gravados em `.npy` sem pickle. Parquet entra com a camada de dados.
- **Reprodutibilidade:**
  - `code_revision` é obtido do Git sem inventar nada: sem commits, `commit = null` com o motivo.
  - O próprio ledger é excluído do critério de código "sujo".
  - `software_environment` registra Python, plataforma, versões das distribuições relevantes e o SHA-256 do `uv.lock`.
  - Execuções `study` são **bloqueadas** sem commit, com código sujo ou sem pré-registro.
- **Justificativa:** O texto append-only é diffável no Git e a cadeia torna edições retroativas detectáveis. (b) é igualmente robusto, mas binário. (c) é centrado em *runs*, permite apagá-los e não resolve a semântica de "tentativa conta para N" [R].
- **Consequências:** + auditável; + sem dependências novas. − Consultas carregam o arquivo inteiro (aceitável até ~10⁵ linhas [H]). − Verificar o arquivo inteiro ao abrir custa O(n): medir quando houver grades reais.
- **Dependências estatísticas:** a definição de "tentativa" usada em N (só a grade final ou todo o histórico) é DE-04. O ledger guarda tudo, então ambas as opções ficam possíveis.
- **Reavaliar se:** o arquivo passar de dezenas de MB, ou surgir necessidade de UI → exportar para MLflow, mantendo o JSONL como fonte de verdade.

## DA-04 — Versionamento de dados, parâmetros e resultados
- **Status:** IN PROGRESS
- **Problema:** Reconstruir qualquer resultado publicado.
- **Alternativas consideradas:** (a) **manifestos SHA-256 versionados + brutos imutáveis fora do Git + derivados endereçados por hash**; (b) DVC; (c) brutos no Git (LFS); (d) nada além de pastas.
- **Critérios:** reconstruibilidade, custo operacional, adequação a uma só máquina.
- **Decisão:** (a). Parâmetros em TOML versionados (`configs/`); estudos pré-registrados e commitados antes da execução; cada tentativa registra commit, estado *dirty*, hash do lockfile e hash do manifesto de dados.
- **Justificativa:** Entrega a rastreabilidade de DVC sem ferramenta extra enquanto houver uma máquina e um usuário [R]. A B3 pode alterar ou remover arquivos históricos; por isso o hash dos brutos efetivamente usados importa [R].
- **Consequências:** − Backup dos brutos é responsabilidade manual. − Execução com código *dirty* deve ser bloqueada para estudos reais (permitida só para exploração, marcada como tal).
- **Reavaliar se:** houver colaboradores, múltiplas máquinas ou armazenamento remoto → DVC.

## DA-05 — Armazenamento e estruturas de dados
- **Status:** IN PROGRESS
- **Alternativas consideradas:** (a) **Parquet + Polars** (ETL) e **NumPy** (núcleo); (b) pandas no lugar de Polars; (c) DuckDB como armazenamento principal; (d) SQLite/PostgreSQL; (e) CSV.
- **Critérios:** desempenho de ETL, joins as-of para point-in-time, tipagem de colunas, interoperabilidade, número de bibliotecas.
- **Decisão:** (a). Uma única biblioteca de DataFrame (Polars) na camada de dados; o núcleo recebe arrays NumPy. Tabelas canônicas: `prices_raw`, `assets` (ISIN × ticker × vigência), `corporate_events`, `returns_total`, `cdi_daily`, `eligibility`. DuckDB opcional para exploração ad hoc.
- **Justificativa:** Parquet é colunar, tipado e portável [E-prévio]; Polars oferece `join_asof` e leitura lazy [E-prévio]. Banco relacional servidor não se justifica para dados estáticos de volume moderado [R].
- **Consequências:** − Polars teve mudanças de API antes da 1.0; fixar a versão no lockfile. − Conversões Polars → NumPy explícitas na fronteira das camadas.
- **Benchmark:** B1 confirma Polars vs. pandas para o parser.

## DA-06 — Estratégia de testes
- **Status:** IN PROGRESS
- **Decisão:** pytest + hypothesis. Seis categorias (ARCHITECTURE §3.4): unitários, propriedade, resposta conhecida, referência (golden files gerados em R), vazamento, integração. Benchmarks fora da suíte padrão.
- **Alternativas consideradas:** chamar R a cada execução de teste (rpy2/Rscript) vs. **golden files versionados**. Escolhido golden: a suíte roda sem R e a referência fica congelada e auditável.
- **Critérios para considerar um teste de estatística válido:** verifica uma propriedade matemática ou um valor de referência independente, nunca apenas "executa sem erro".
- **Consequências:** − Golden files precisam ser regenerados se a entrada de referência mudar (script documentado em `validation/r/`).
- **Dependências estatísticas:** tolerâncias numéricas de comparação dependem de DE-05 (empates, truncamento de T em S blocos) e de DE-08 (referência do DSR).

## DA-07 — Desempenho e paralelismo
- **Status:** IN PROGRESS (orçamento **provisório** definido pelo desenvolvedor em 2026-11-06; ver abaixo)
- **Problema:** Executar grade, CSCV e Monte Carlo em tempo aceitável, com reprodutibilidade.
- **Alternativas consideradas:** NumPy vetorizado; Numba; `ProcessPoolExecutor`; joblib; Ray/Dask; extensões nativas.
- **Decisão:** NumPy vetorizado + `concurrent.futures.ProcessPoolExecutor` (paralelismo entre réplicas/tarefas independentes). RNG: `numpy.random.SeedSequence` raiz por estudo, `spawn` por réplica, e resultado idêntico para qualquer número de processos. CSCV com Sharpe usa estatísticas suficientes por bloco quando a métrica permitir. **Numba só por gatilho medido** (B2, B3, B5).
- **Justificativa:** As cargas são embaraçosamente paralelas no nível da réplica [R]; Ray/Dask resolvem distribuição entre máquinas, que não é necessária.
- **Consequências:** − O custo da CSCV depende da métrica escolhida em DE-05 (diferença de cerca de 10³ vezes, STACK_EVALUATION §1.2). − Se Numba for adotado, gerar números aleatórios fora dos kernels.
- **Orçamento provisório (desenvolvedor, 2026-11-06), ajustado à máquina medida:**

  Máquina de referência (medida): Intel i5-7200U, **2 núcleos físicos / 4 lógicos**, **7,88 GiB de RAM** com apenas **0,94–1,4 GiB disponíveis** em duas medições (outros aplicativos abertos), 706 GB livres em disco, Windows 10 64 bits, Python 3.13.5 (uv).

  | Recurso | Meta do desenvolvedor | Ajuste / estado |
  |---|---|---|
  | Memória | ≤ 60% da RAM disponível | Aplicada no início de cada execução (`runtime.plan_workers`). Com a disponibilidade medida, o orçamento fica em **~0,56–0,84 GiB**. A execução falha explicitamente se nem um processo couber. **Recomendação:** fechar aplicativos pesados antes de estudos completos. |
  | Paralelismo | Processos locais | `max_workers = 2` (núcleos físicos); 1 thread nativa por processo. Começar sequencial; paralelizar só após B5. No Windows os processos usam *spawn*: cada um reimporta NumPy (custo de memória a medir). |
  | Disco | ≤ 10 GB | Sem restrição prática (706 GB livres). |
  | Testes unitários | < 2 min | **Medido: ~10 s** (59 testes). |
  | Integração | < 10 min | **Medido: ~16 s** para a suíte completa (61 testes). |
  | Experimento sintético de referência | < 15 min | Não medido (não existe ainda). |
  | Estudo completo | ≤ 2 h | Não medido. **Risco [H]:** Monte Carlo com ONC num notebook de 2 núcleos pode exceder; decidir após B5. Não reduzir réplicas, configurações ou partições para caber (regra do desenvolvedor). |

  Numba e linguagens compiladas: **excluídos nesta fase** por instrução do desenvolvedor; só voltam à discussão com evidência de B2–B5.

## DA-08 — CLAUDE.md privado e fora do controle de versão
- **Status:** DECIDED (2026-11-06). Detalhes em PROJECT_DECISIONS.md.

## DA-09 — Ingestão do COTAHIST
- **Status:** IN PROGRESS
- **Alternativas consideradas:** (a) **parser próprio em Python/Polars** sobre o layout oficial de 245 bytes [E]; (b) `rb3` (R) no caminho de produção; (c) fonte de terceiros com preços já ajustados.
- **Critérios:** controle sobre campos (ISIN, código BDI, tipo de mercado, fator de cotação), independência de API externa, verificabilidade.
- **Decisão:** (a), preservando **todas** as colunas do layout no bruto tipado. `rb3` como checagem cruzada (B1). (c) rejeitado como fonte primária: ajuste opaco, risco de sobrevivência e de look-ahead não auditáveis.
- **Justificativa:** O layout é público e estável (revisões 2017 e 2020 [E]); um parser de largura fixa é pequeno e testável. A API do `rb3` mudou entre versões [E].
- **Consequências:** − Implementar e testar o parser contra o layout (incluindo o registro header/trailer). − Confirmar no layout a revisão vigente e o campo de fator de cotação.
- **Registro de implementação (2026-11-06):**
  - `frente_e.data.cotahist`: leitor em largura fixa com *dtype* estruturado do NumPy (sem dependência nova), leitura em blocos e validação de header, trailer, terminadores e tipos;
  - o layout v2.0 foi conferido visualmente;
  - obtenção imutável com manifesto (DA-04), sem dependência de `experiments`;
  - **desvio:** o bruto tipado em Parquet (todas as colunas) foi **adiado** para a normalização, porque a auditoria lê os zips diretamente e Parquet exigiria Polars/pyarrow sem necessidade atual;
  - o `rb3` ainda não foi usado como checagem cruzada (B1 pendente).

## DA-10 — Representação de eventos corporativos e retornos
- **Status:** IN PROGRESS (fonte dos eventos: **PENDING**, DD-03)
- **Alternativas consideradas:** (a) **tabela de eventos separada + retorno total diário calculado na data do evento**; (b) série de preços retroajustada; (c) preços ajustados de terceiros.
- **Decisão:** (a). Preço bruto nunca é alterado; o retorno de t−1 a t incorpora o evento que ocorre em t.
- **Justificativa:** O retroajuste (b) reescreve o passado a cada novo evento, muda resultados conforme a data de download e dificulta a auditoria [R]. O retorno *forward* é auto-consistente e não usa informação futura.
- **Consequências:** − Depende de uma fonte de eventos confiável com datas ex (DD-03). Fontes identificadas, ainda não avaliadas: Manual de Eventos Corporativos da B3 (metodologia [E]), serviços de terceiros [E, qualidade não verificada].

## DA-11 — Gestão de ambiente
- **Status:** IN PROGRESS
- **Alternativas consideradas:** (a) **uv** (Python, `pyproject.toml` + `uv.lock`) + **renv** (R de validação); (b) pixi (Python e R num só lockfile via conda-forge [E]); (c) conda/mamba; (d) pip + venv.
- **Critérios:** lockfile reproduzível, funcionamento no Windows, isolamento do R, simplicidade.
- **Decisão:** (a). R isolado em `validation/r/` com `renv.lock`.
- **Justificativa:** Mantém o ambiente principal só em Python. (b) seria superior se `r-pbo` e `r-poolr` existirem no conda-forge, o que **não foi confirmado** [H].
- **Consequências:** − Duas ferramentas de ambiente. Nenhuma versão está fixada neste documento; as versões ficam registradas no `uv.lock`.
- **Aprovações pontuais do desenvolvedor (2026-11-06):** instalar uv 0.13.7 via `pip install --user uv` (o executável fica fora do PATH; usar `python -m uv`); fixar **Python 3.13** gerenciado pelo uv (`.python-version`; 3.13.5 instalado), sem alterar o Python 3.14 do sistema; `git init` local. A parte R (renv) segue pendente: **R não está instalado**.
- **Reavaliar se:** a verificação mostrar os pacotes R necessários no conda-forge → considerar pixi.

## DA-12 — Frameworks de backtesting
- **Status:** IN PROGRESS
- **Alternativas consideradas:** vectorbt, Backtrader, zipline-reloaded, Qlib, mlfinlab, **motor vetorizado próprio**.
- **Decisão:** Motor próprio, mínimo, vetorizado sobre a grade. Nenhum framework como dependência.
- **Justificativa:** O objeto de estudo exige controle total sobre convenções (timing do sinal, sobreposição de coortes, custos), que precisam ser documentadas e testadas. Motores orientados a eventos são inadequados para 10³ configurações [R]. vectorbt tem licença com Commons Clause [E]; mlfinlab é proprietário [E]. Aproveitam-se conceitos (broadcast de parâmetros do vectorbt; pipeline sem look-ahead do Zipline).
- **Consequências:** − Mais código próprio para testar; compensado pelos testes de vazamento e pelo escopo pequeno (só momentum).

## DA-13 — Relatórios
- **Status:** IN PROGRESS (ferramenta de documento final: **PENDING**)
- **Decisão:** Tabelas e figuras geradas por código (`reporting/`, matplotlib) a partir do ledger e dos artefatos. Notebooks só para exploração. Ferramenta de documento final (Quarto vs. Markdown gerado por script vs. LaTeX) decidida na etapa de relatórios.
- **Justificativa:** Sem requisito de interface web ou dashboard; um relatório gerado por código é reproduzível e serve tanto à publicação acadêmica quanto ao LinkedIn (exportando figuras).

## DA-14 — Obtenção do CDI
- **Status:** IN PROGRESS
- **Alternativas consideradas:** `python-bcb` (citado no documento de partida; **não confirmado** na busca desta etapa); `sgs`, `bcbpy` [E]; chamada direta à API SGS.
- **Decisão:** Verificar `python-bcb` (manutenção, licença) na etapa de ingestão; se inadequado, chamada direta à API com paginação por janelas. Série 12 (CDI diário) [E]. Guardar o JSON bruto com hash.
- **Consequências:** − Respeitar o limite de janela por consulta relatado (10 anos desde mar/2025 [E: README de terceiro, confirmar]).

---

## Dependências e licenças (2026-11-06)

Todas gratuitas e de código aberto. As versões exatas estão em `uv.lock` (Python) e `validation/r/renv.lock` (R).

| Dependência | Tipo | Licença | Motivo |
|---|---|---|---|
| numpy | execução | BSD-3 | matrizes, RNG (`SeedSequence`/`PCG64`), artefatos `.npy` |
| pytest | desenvolvimento | MIT | testes |
| hypothesis | desenvolvimento | MPL-2.0 | testes por propriedade |
| ruff | desenvolvimento | MIT | lint/formatação |
| mypy | desenvolvimento | MIT | tipos (strict) |
| scipy | **execução** (desde 2026-11-06) | BSD-3 | `special.ndtr`/`ndtri` (Φ e Φ⁻¹ vetorizados e precisos nas duas caudas) no PSR, DSR e MinTRL; `integrate.quad` para o E[máx] exato; `optimize.brentq` para o N implícito. A alternativa da stdlib (`statistics.NormalDist`) é escalar, e seu `cdf` via `erf` perde precisão na cauda esquerda. Já estava no lockfile; a versão travada não mudou. Continua sendo referência independente nos testes de momentos |
| scipy-stubs | desenvolvimento | BSD-3 | tipos do SciPy para o mypy, em vez de silenciar o import |
| uv 0.13.7 | ferramenta | MIT/Apache-2.0 | ambiente e lockfile |
| R 4.6.2 | ferramenta (validação) | GPL-2/GPL-3 | só gera golden files |
| renv 1.3.1, pbo 1.3.5 | validação | MIT | ambiente R; referência de PBO |
| poolr 1.2-0 (+ transitivas) | validação | GPL (≥ 2) e outras do `renv.lock` | referência de N efetivo; usado localmente, sem distribuição de código |

Ferramentas usadas uma vez, em ambiente efêmero (`uv run --no-project --with`), fora do projeto: `pypdf` (BSD) e `pdfminer.six` (MIT), para extrair o texto do artigo do DSR.
