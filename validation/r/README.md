# Validação independente em R

Ambiente R **isolado** (`renv`), usado **apenas** para gerar resultados de referência (golden files) em `tests/fixtures/golden/`. A suíte Python não executa R: ela confere os hashes do `manifest.json` e usa os valores congelados.

| Pacote | Versão (renv.lock) | Licença | Referência para |
|---|---|---|---|
| `pbo` | 1.3.5 | MIT | CSCV / PBO. **Não implementa DSR**: nunca citar como validação do DSR |
| `poolr` | 1.2-0 | GPL (≥ 2) | N efetivo por autovalores (`meff`: nyholt, liji, gao, galwey) |
| `renv` | 1.3.1 | MIT | ambiente reproduzível |
| `rb3` | não instalado | MIT (a confirmar) | conferência do parser do COTAHIST; instalar na etapa do parser (B1) |

As dependências transitivas (lattice, foreach, Rcpp, RcppEigen etc.) estão listadas com versão e licença no `renv.lock`. O código GPL é usado só como ferramenta local; os golden files são dados de saída, e nenhum código R é distribuído junto com o pacote Python.

**Por que o `rb3` não está instalado agora:** ele só é necessário na etapa do parser do COTAHIST, e sua árvore de dependências é bem maior que a do `pbo` e do `poolr`. Instalar quando for usado evita manter pacotes sem uso no ambiente.

## Estado (2026-11-06)
- R 4.6.2 instalado por usuário em `%LOCALAPPDATA%\Programs\R\R-4.6.2` (sem administrador).
- `renv` inicializado em modo `explicit`: só entra no lockfile o que está declarado em `DESCRIPTION`.
- Golden files gerados e congelados: `pbo_noise`, `pbo_signal`, `pbo_ties`, `meff_independent`, `meff_blocks`.

## Instalação do zero (Windows, sem administrador)
1. **R**: baixe o instalador oficial do CRAN e confira o SHA-256 (o manifesto do winget publica o hash):
   ```powershell
   winget show --id RProject.R --exact          # mostra URL do instalador e SHA256
   # baixe a URL indicada e confira: Get-FileHash .\R-<versão>-win.exe -Algorithm SHA256
   .\R-<versão>-win.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CURRENTUSER `
       /DIR="$env:LOCALAPPDATA\Programs\R\R-<versão>" /COMPONENTS=main,x64 `
       /MERGETASKS="!desktopicon,!recordversion,!associate"
   ```
   O `winget install` falhou nesta máquina porque o manifesto não tem instalador por usuário; o instalador Inno Setup aceita `/CURRENTUSER`.
2. **Biblioteca de usuário e renv** (o diretório precisa existir antes do `install.packages`):
   ```bash
   RS="$LOCALAPPDATA/Programs/R/R-4.6.2/bin/Rscript.exe"
   "$RS" -e 'dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE); install.packages("renv", repos = "https://cloud.r-project.org", lib = Sys.getenv("R_LIBS_USER"))'
   ```
3. **Restaurar o ambiente travado** (a partir de `validation/r`, para o `.Rprofile` ativar o renv):
   ```bash
   cd validation/r
   "$RS" -e 'renv::restore(prompt = FALSE)'
   ```

## Gerar e congelar as referências
```bash
# na raiz do projeto
python -m uv run python validation/golden_inputs.py           # 1. entradas sintéticas (semente fixa)
(cd validation/r && "$RS" scripts/make_golden.R)               # 2. referências com pbo/poolr (~20 s)
python -m uv run python validation/freeze_golden.py            # 3. manifest.json com SHA-256
python -m uv run pytest tests/reference                        # 4. confere integridade
```
Regerar as referências muda os hashes. Registre o motivo no commit e nunca edite os CSV à mão.

## Estrutura de cada caso (`tests/fixtures/golden/<caso>/`)
| Arquivo | Conteúdo |
|---|---|
| `input.csv` | entrada T × N; números em `repr` (round-trip exato) |
| `input_generation.txt` | semente, fluxo e versão do NumPy que gerou a entrada |
| `expected_*.csv` | saída do pacote R em `%.17g` |
| `meta.csv` | versões do R e dos pacotes; função chamada |
| `manifest.json` | SHA-256 de tudo + tolerância numérica com justificativa |

## Adicionar pacote
Declare em `DESCRIPTION`, rode `renv::install("<pkg>")` e depois `renv::snapshot(type = "explicit")`, e registre a licença nesta página. Não use conda-forge.
