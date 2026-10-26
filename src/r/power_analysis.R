# Power by simulation on pilot E: resample participants with replacement at each candidate N, refit the
# descriptive model, apply the registered rule, and record how often each test comes out "supported".
# Input:  .../games/play_games.csv, .../descriptive/{contrast_weights,cell_means,contrasts}.csv (stages 4.1-4.5)
# Output: .../power/decision_rates.csv   N x effect x hypothesis x outcome: rate of "supported", n_sims, n_failed
# Effect levels: "pilot" = the resampled data as is; "half" = every outcome shrunk toward its grand mean by half
# of the pilot cell effect, which halves every contrast and keeps the participant structure and the noise.
# Run:    VERSION=pilot_E N_SIMS=500 N_CORES=8 Rscript src/r/power_analysis.R   (N_SIMS default 500, N_CORES 1)

suppressPackageStartupMessages({ library(here); library(dplyr); library(readr); library(lme4); library(lmerTest); library(emmeans) })
if (Sys.getenv("OUTPUT_DATA_DIR") == "") dotenv::load_dot_env(here::here(".env"))
experiment_name <- Sys.getenv("EXPERIMENT_NAME", unset = "transfer_bandits_choice_textbox")
version         <- Sys.getenv("VERSION")
if (version == "") stop("VERSION is not set (e.g. VERSION=pilot_E Rscript src/r/power_analysis.R)")
analysis_dir <- file.path(Sys.getenv("OUTPUT_DATA_DIR"), "analysis", experiment_name, version)
out_dir <- file.path(analysis_dir, "power"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
N_GRID <- c(200, 250, 300, 350, 400, 500, 600, 700, 800); N_SIMS <- as.integer(Sys.getenv("N_SIMS", unset = "500")); N_CORES <- as.integer(Sys.getenv("N_CORES", unset = "1"))
set.seed(20261025)

# 1) Inputs: the play games, the weight vectors, the cell effects for shrinking, the confirmatory tests.
games <- read_csv(file.path(analysis_dir, "games", "play_games.csv"), show_col_types = FALSE) %>% filter(!exploratory) %>%
  mutate(condition = factor(condition, levels = c("expert-first", "novice-first", "solo")), play = factor(play, levels = c("first", "second")), round = factor(round, levels = c(1, 2)))
weights <- read_csv(file.path(analysis_dir, "descriptive", "contrast_weights.csv"), show_col_types = FALSE)
tests   <- read_csv(file.path(analysis_dir, "descriptive", "contrasts.csv"), show_col_types = FALSE) %>% filter(role == "confirmatory") %>% select(hypothesis, outcome, expected_sign)
OUTCOMES <- unique(tests$outcome); HYPS <- unique(tests$hypothesis)
cell_effect <- read_csv(file.path(analysis_dir, "descriptive", "cell_means.csv"), show_col_types = FALSE) %>% group_by(outcome) %>%
  mutate(effect = emmean - mean(emmean)) %>% ungroup() %>% select(outcome, condition, play, round, effect) %>%
  tidyr::pivot_wider(names_from = outcome, values_from = effect, names_prefix = "effect_") %>% mutate(round = factor(round, levels = c(1, 2)))
games <- left_join(games, cell_effect, by = c("condition", "play", "round"))
options(contrasts = c("contr.sum", "contr.poly"))
emm_options(lmerTest.limit = 1e5)   # keep Satterthwaite df above emmeans' 3000-row default (the full experiment exceeds it)

# 2) One simulation: resample N participants (duplicates become distinct participants), refit, decide.
weight_list <- setNames(lapply(HYPS, function(h) weights[[h]]), HYPS)
decide <- function(N, effect) {
  ids <- sample(unique(games$participant), N, replace = TRUE)
  d <- bind_rows(lapply(seq_along(ids), function(i) mutate(games[games$participant == ids[i], ], participant = i)))
  if (effect == "half") for (y in OUTCOMES) d[[y]] <- d[[y]] - 0.5 * d[[paste0("effect_", y)]]
  p <- bind_rows(lapply(OUTCOMES, function(y) {
    fit <- lmer(reformulate("condition * play * round + (1 | participant)", response = y), data = d, REML = TRUE)
    est <- as.data.frame(contrast(emmeans(fit, ~ condition * play * round, lmer.df = "satterthwaite"), method = weight_list))
    data.frame(hypothesis = as.character(est$contrast), outcome = y, estimate = est$estimate, p.value = est$p.value)
  }))
  inner_join(tests, p, by = c("hypothesis", "outcome")) %>% group_by(hypothesis) %>% mutate(p_holm = p.adjust(p.value, "holm")) %>% ungroup() %>%
    mutate(supported = p_holm < 0.05 & sign(estimate) == expected_sign, N = N, effect = effect)
}
one_sim <- function(s) bind_rows(lapply(N_GRID, function(N) bind_rows(lapply(c("pilot", "half"), function(e) tryCatch(decide(N, e), error = function(err) NULL)))))

# 3) Rates over simulations (RNG streams per core are reproducible under mclapply with the L'Ecuyer kind).
RNGkind("L'Ecuyer-CMRG")
sims <- parallel::mclapply(seq_len(N_SIMS), one_sim, mc.cores = N_CORES, mc.set.seed = TRUE)
rates <- bind_rows(sims) %>% group_by(N, effect, hypothesis, outcome) %>%
  summarise(rate = mean(supported), mean_estimate = mean(estimate), n_sims = n(), n_failed = N_SIMS - n(), .groups = "drop") %>% arrange(effect, N, hypothesis, outcome)
write_csv(rates, file.path(out_dir, "decision_rates.csv"))
cat("N_SIMS =", N_SIMS, "\n"); print(as.data.frame(tidyr::pivot_wider(rates, id_cols = c(effect, hypothesis, outcome), names_from = N, values_from = rate)), digits = 2)
