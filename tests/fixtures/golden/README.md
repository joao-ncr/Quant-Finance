# Golden files

Resultados de referência produzidos por implementações independentes (ver `validation/r/README.md`). Versionados e nunca editados à mão: regenere pelo script de origem e registre o motivo no commit. Casos (gerados em 2026-11-06; detalhes em `validation/golden_inputs.py`):

| Caso | Conteúdo | Para quê |
|---|---|---|
| `pbo_noise` | ruído i.i.d., N = 20, S ∈ {4, 8, 12} | CSCV/PBO sem sinal |
| `pbo_signal` | coluna 0 com +0,3%/dia, N = 20 | PBO = 0 |
| `pbo_ties` | colunas 1 e 2 idênticas, N = 20, S ∈ {4, 8} | regras de empate (o projeto rejeita colunas idênticas: DE-05.c) |
| `pbo_odd` | ruído i.i.d., **N = 9**, S ∈ {4, 8, 12} | **discrimina** rank/N (`pbo`) de rank/(N+1) (artigo): partições com rank OOS = 5 em S = 8 (1) e S = 12 (10); S = 4 é controle. φ do pacote 0,1429 e 0,0942; φ pela definição do artigo 0,1571 e 0,1050 |
| `meff_independent` | 20 séries independentes | N efetivo ≈ 20 |
| `meff_blocks` | 4 fatores × 5 séries + ruído | N efetivo com estrutura de blocos |
