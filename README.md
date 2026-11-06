# Frente E: validação estatística de regras simples na B3

Pesquisa acadêmica: quanto do desempenho aparente de uma família de regras de momentum na B3 sobrevive às correções por seleção e múltiplos testes (Deflated Sharpe Ratio, Probability of Backtest Overfitting via CSCV), e como essa deflação varia com o tamanho da família e com a liquidez.

**Estado:** fundação do projeto. Nenhuma métrica estatística implementada e nenhum resultado empírico.

## Documentação
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): camadas, contratos e fluxos.
- [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md): decisões arquiteturais (ADRs).
- [docs/STACK_EVALUATION.md](docs/STACK_EVALUATION.md): avaliação de linguagens e ferramentas.
- [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md): etapas e critérios de conclusão.
- [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md): decisões estatísticas e de dados, com status.

## Requisitos
- [uv](https://docs.astral.sh/uv/) (gerencia o Python 3.13 e as dependências pelo `uv.lock`).
- R **não** é necessário para a suíte de testes; só para regenerar referências em `validation/r/`.

## Uso
```bash
uv sync                          # cria .venv com Python 3.13 e as dependências travadas
uv run python -m frente_e info   # ambiente, recursos da máquina e orçamento de execução
uv run pytest                    # suíte completa
uv run pytest -m "not integration"   # apenas testes unitários
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
Se o executável `uv` não estiver no PATH, use `python -m uv` no lugar de `uv`.

Para reproduzir uma falha de teste por propriedade de forma determinística: `HYPOTHESIS_PROFILE=repro uv run pytest`.

## Estrutura
```
src/frente_e/   data/ · core/ (puro, sem I/O) · experiments/ · reporting/ · cli.py
configs/        project.toml (caminhos e orçamento) · studies/ (pré-registros)
tests/          unit/ · integration/ · fixtures/golden/
validation/r/   ambiente R isolado que gera golden files
benchmarks/     B1–B6 (docs/STACK_EVALUATION.md §6)
```
