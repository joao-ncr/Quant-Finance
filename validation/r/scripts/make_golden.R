# Passo 2 de 3: calcula referências independentes com os pacotes R travados no renv.lock.
#
# Uso (a partir de validation/r, para o .Rprofile ativar o renv):
#   Rscript scripts/make_golden.R
#
# Lê tests/fixtures/golden/<caso>/input.csv e grava expected_*.csv e meta.csv no mesmo
# diretório. Números em "%.17g" (round-trip exato de double).
# Nada aqui é reimplementação: só chamadas aos pacotes, mais extração dos campos.

suppressPackageStartupMessages({
  library(pbo)
  library(poolr)
})

golden_dir <- normalizePath(file.path("..", "..", "tests", "fixtures", "golden"), mustWork = TRUE)

fmt <- function(x) {
  if (is.numeric(x)) {
    out <- sprintf("%.17g", x)
    out[is.infinite(x) & x > 0] <- "Inf"
    out[is.infinite(x) & x < 0] <- "-Inf"
    out
  } else {
    as.character(x)
  }
}

write_exact <- function(df, path) {
  df[] <- lapply(df, fmt)
  write.table(df, path, sep = ",", row.names = FALSE, quote = FALSE, eol = "\n",
              fileEncoding = "UTF-8")
}

read_input <- function(case) {
  m <- read.csv(file.path(golden_dir, case, "input.csv"), header = TRUE,
                colClasses = "numeric")
  stopifnot(!anyNA(m))
  m
}

write_meta <- function(case, extra) {
  meta <- data.frame(
    key = c("r_version", "pbo_version", "poolr_version", "renv_version", names(extra)),
    value = c(R.version.string,
              as.character(packageVersion("pbo")),
              as.character(packageVersion("poolr")),
              as.character(packageVersion("renv")),
              unlist(extra)),
    stringsAsFactors = FALSE
  )
  write_exact(meta, file.path(golden_dir, case, "meta.csv"))
}

# Métrica de desempenho por coluna: Sharpe diário com sd amostral (ddof = 1),
# a mesma convenção proposta em DE-02.
sharpe_columns <- function(x) apply(as.matrix(x), 2, function(col) mean(col) / sd(col))

pbo_case <- function(case, s_values) {
  m <- read_input(case)
  for (s in s_values) {
    stopifnot(nrow(m) %% s == 0)
    r <- pbo(m, s = s, f = sharpe_columns, threshold = 0)
    n_combos <- ncol(r$combos)
    per_combo <- data.frame(
      combo = seq_len(n_combos),
      is_blocks = apply(r$combos, 2, paste, collapse = ";"),
      n_star = as.integer(unlist(r$results[, "n*"])),
      n_max_oos = as.integer(unlist(r$results[, "n_max_oos"])),
      os_rank = as.numeric(unlist(r$results[, "os_rank"])),
      omega_bar = as.numeric(unlist(r$results[, "omega_bar"])),
      lambda_raw = as.numeric(unlist(r$results[, "lambda"])),
      lambda = r$lambda,
      rn = r$rn_pairs$Rn,
      rbn = r$rn_pairs$Rbn
    )
    write_exact(per_combo, file.path(golden_dir, case, sprintf("expected_pbo_s%d.csv", s)))
    summary_df <- data.frame(
      key = c("s", "n_combos", "t", "n", "phi", "inf_sub", "threshold"),
      value = c(s, n_combos, nrow(m), ncol(m), r$phi, r$inf_sub, r$threshold)
    )
    write_exact(summary_df, file.path(golden_dir, case, sprintf("expected_pbo_summary_s%d.csv", s)))
  }
  write_meta(case, list(function_called = "pbo::pbo(m, s, f = sharpe_columns, threshold = 0)",
                        s_values = paste(s_values, collapse = ";")))
}

meff_case <- function(case) {
  m <- as.matrix(read_input(case))
  R <- cor(m)
  methods <- c("nyholt", "liji", "gao", "galwey")
  values <- sapply(methods, function(meth) poolr::meff(R, method = meth))
  write_exact(data.frame(method = methods, meff_floor = as.numeric(values)),
              file.path(golden_dir, case, "expected_meff.csv"))
  write_exact(data.frame(eigenvalue = base::eigen(R, symmetric = TRUE, only.values = TRUE)$values),
              file.path(golden_dir, case, "expected_eigenvalues.csv"))
  write_meta(case, list(function_called = "poolr::meff(cor(m), method); gao com C = 0.995 (padrão)",
                        note = "poolr aplica floor() ao resultado de todos os métodos"))
}

pbo_case("pbo_noise", c(4, 8, 12))
pbo_case("pbo_signal", c(4, 8, 12))
pbo_case("pbo_ties", c(4, 8))
pbo_case("pbo_odd", c(4, 8, 12))  # N ímpar: discrimina rank/N de rank/(N+1)
meff_case("meff_independent")
meff_case("meff_blocks")
cat("golden references written to", golden_dir, "\n")
