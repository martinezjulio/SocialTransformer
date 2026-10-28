# Descriptive stage: the 12-cell mixed model per outcome on the per-play-game table.
# Input:  $OUTPUT_DATA_DIR/analysis/<version>/games/play_games.csv (src/analysis/games.py)
# Output: .../descriptive/cell_means.csv   emmeans grid, one row per outcome x condition x play x round
#         .../descriptive/contrasts.csv    H1, H2, H3, H5 (confirmatory: Holm over payoff and entropy within a
#                                          hypothesis, verdict), their reward rows (reported: interval, no rule;
#                                          advisor, 28 Oct) and the registered exploratory contrasts, as weight
#                                          vectors over the 12 cells. H4 is the partner choice (partner_choice.R).
#         .../descriptive/contrast_weights.csv  the 12-cell weight vector of every contrast (one column each)
#         .../descriptive/robustness.csv   the confirmatory contrasts under two sensitivity fits: random slopes for round and play
#                                          (reduced in a stated order if singular) and session order as a covariate
# Model:  outcome ~ condition * play * round + (1 | participant), sum-to-zero coding, REML,
#         Satterthwaite df (lmerTest). Solo slots 1 and 3 (exploratory) are left out.
# Run:    VERSION=pilot_E Rscript src/r/descriptive.R

suppressPackageStartupMessages({
  library(here)
  library(dplyr)
  library(readr)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
if (Sys.getenv("OUTPUT_DATA_DIR") == "") dotenv::load_dot_env(here::here(".env"))

# 1) Paths; VERSION has no default so a run can never target the wrong pilot silently.
version         <- Sys.getenv("VERSION")
if (version == "") stop("VERSION is not set (e.g. VERSION=pilot_E Rscript src/r/descriptive.R)")
analysis_dir <- file.path(Sys.getenv("OUTPUT_DATA_DIR"), "analysis", version)
out_dir      <- file.path(analysis_dir, "descriptive")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 2) Data: factors with sum-to-zero contrasts so main effects are grand-mean centred.
OUTCOMES <- c("payoff", "reward", "entropy")
games <- read_csv(file.path(analysis_dir, "games", "play_games.csv"), show_col_types = FALSE) %>%
  filter(!exploratory) %>%
  mutate(
    participant = factor(participant),
    condition   = factor(condition, levels = c("expert-first", "novice-first", "solo")),
    play        = factor(play, levels = c("first", "second")),
    round       = factor(round, levels = c(1, 2))
  )
stopifnot(nrow(games) == 12 * nlevels(games$participant), all(table(games$participant) == 12))
options(contrasts = c("contr.sum", "contr.poly"))
emm_options(lmerTest.limit = 1e5)   # keep Satterthwaite df above emmeans' 3000-row default (the full experiment exceeds it)

# 3) One model per outcome; the emmeans grid over the 12 cells.
fits <- lapply(setNames(OUTCOMES, OUTCOMES), function(y) {
  lmer(reformulate("condition * play * round + (1 | participant)", response = y), data = games, REML = TRUE)
})
emms <- lapply(fits, function(fit) emmeans(fit, ~ condition * play * round, lmer.df = "satterthwaite"))
cell_means <- bind_rows(lapply(OUTCOMES, function(y) mutate(as.data.frame(emms[[y]]), outcome = y, .before = 1)))
write_csv(cell_means, file.path(out_dir, "cell_means.csv"))
cat("Wrote", nrow(cell_means), "cell means for", length(OUTCOMES), "outcomes to", file.path(out_dir, "cell_means.csv"), "\n")

# 4) Contrasts as weight vectors over the 12 cells (same row order in every emmeans grid). The
#    demonstrator watched before a play game: expert-first first play and novice-first second play.
grid <- as.data.frame(emms$payoff)[, c("condition", "play", "round")]
grid$demo <- with(grid, ifelse(condition == "solo", "none", ifelse((condition == "expert-first") == (play == "first"), "expert", "novice")))
m <- function(sel) as.numeric(sel) / sum(sel)              # mean over the selected cells
PR <- c("payoff", "reward")
CONTRASTS <- with(grid, list(
  H1 = list(w = m(demo == "expert") - m(demo == "novice"), outcomes = OUTCOMES, sign = c(1, 1, -1), role = "confirmatory"),
  H2 = list(w = m(demo == "novice" & round == 2) - m(demo == "novice" & round == 1), outcomes = OUTCOMES, sign = c(1, 1, -1), role = "confirmatory"),
  H3 = list(w = m(demo != "none" & round == 2) - m(demo != "none" & round == 1), outcomes = "entropy", sign = -1, role = "confirmatory"),
  H5 = list(w = m(demo == "expert") - m(demo == "none" & play == "second"), outcomes = PR, sign = c(1, 1), role = "confirmatory"),
  # Exploratory contrasts from the registration's closed list: intervals reported, no verdict. The former H4
  # (after-novice improvement vs solo improvement) is exploratory on every outcome since 28 Oct.
  novice_vs_solo_improvement = list(w = (m(demo == "novice" & round == 2) - m(demo == "novice" & round == 1)) - (m(demo == "none" & round == 2) - m(demo == "none" & round == 1)),
                                    outcomes = OUTCOMES, sign = c(1, 1, -1), role = "exploratory"),
  expert_round_change = list(w = m(demo == "expert" & round == 2) - m(demo == "expert" & round == 1), outcomes = OUTCOMES, sign = c(1, 1, -1), role = "exploratory"),
  solo_round_change   = list(w = m(demo == "none" & round == 2) - m(demo == "none" & round == 1), outcomes = OUTCOMES, sign = c(NA, NA, NA), role = "exploratory"),
  H5_round2           = list(w = m(demo == "expert" & round == 2) - m(demo == "none" & play == "second" & round == 2), outcomes = PR, sign = c(1, 1), role = "exploratory"),
  social_vs_solo4     = list(w = m(demo != "none") - m(demo == "none" & play == "second"), outcomes = PR, sign = c(1, 1), role = "exploratory"),
  novice_vs_solo4     = list(w = m(demo == "novice") - m(demo == "none" & play == "second"), outcomes = PR, sign = c(NA, NA), role = "exploratory")
))
stopifnot(all(sapply(CONTRASTS, function(k) abs(sum(k$w)) < 1e-12)))
CONFIRMATORY <- names(CONTRASTS)[sapply(CONTRASTS, `[[`, "role") == "confirmatory"]
write_csv(bind_cols(grid, as.data.frame(lapply(CONTRASTS, `[[`, "w"))), file.path(out_dir, "contrast_weights.csv"))   # read by power_analysis.R

contrast_rows <- function(emm_list, names) bind_rows(lapply(names, function(h) {
  k <- CONTRASTS[[h]]
  bind_rows(lapply(seq_along(k$outcomes), function(i) {
    est <- summary(contrast(emm_list[[k$outcomes[i]]], method = list(w = k$w)), infer = TRUE)
    role <- if (k$role == "confirmatory" && k$outcomes[i] == "reward") "reported" else k$role   # reward carries no rule (28 Oct)
    data.frame(hypothesis = h, outcome = k$outcomes[i], role = role, expected_sign = k$sign[i], as.data.frame(est)[, -1])
  }))
}))
contrasts <- contrast_rows(emms, names(CONTRASTS)) %>%
  group_by(hypothesis) %>%
  mutate(p_holm = replace(rep(NA_real_, n()), role == "confirmatory", p.adjust(p.value[role == "confirmatory"], method = "holm"))) %>%
  ungroup() %>%
  mutate(verdict = case_when(role != "confirmatory" ~ NA_character_,
                             p_holm < 0.05 & sign(estimate) == expected_sign ~ "supported",
                             p_holm < 0.05 ~ "contradicted",
                             TRUE ~ "not supported"))
write_csv(contrasts, file.path(out_dir, "contrasts.csv"))
print(as.data.frame(contrasts[contrasts$role != "exploratory", c("hypothesis", "outcome", "estimate", "lower.CL", "upper.CL", "p.value", "p_holm", "verdict")]), digits = 3)

# 5) Robustness fits (exploratory sensitivity, the confirmatory contrasts only, no Holm or verdict). Random slopes for round and
#    play by participant are reduced in a fixed order when the fit is singular or does not converge:
#    correlated slopes -> uncorrelated slopes -> intercept only (the main model). The random part uses
#    +-0.5 codings of round and play (lme4's || would expand a factor into one column per level). Session
#    order (1..3, randomised per participant) enters as a numeric covariate; emmeans takes it at its mean.
games <- mutate(games, round_c = ifelse(round == "2", 0.5, -0.5), play_c = ifelse(play == "second", 0.5, -0.5))
FIXED <- "condition * play * round"
RANDOM <- c(slopes_correlated = "(1 + round_c + play_c | participant)", slopes_uncorrelated = "(1 + round_c + play_c || participant)",
            intercept_only = "(1 | participant)")
fit_ok <- function(fit) !isSingular(fit) && is.null(fit@optinfo$conv$lme4$messages)
fit_reduced <- function(y) {
  for (spec in names(RANDOM)) {
    fit <- suppressWarnings(suppressMessages(lmer(reformulate(paste(FIXED, "+", RANDOM[[spec]]), response = y), data = games, REML = TRUE)))
    if (fit_ok(fit) || spec == "intercept_only") return(list(fit = fit, spec = spec))
  }
}
ROBUST <- list(
  random_slopes = lapply(setNames(OUTCOMES, OUTCOMES), fit_reduced),
  session_order = lapply(setNames(OUTCOMES, OUTCOMES), function(y) list(
    fit = lmer(reformulate(paste(FIXED, "+ session_order + (1 | participant)"), response = y), data = games, REML = TRUE), spec = "covariate"))
)
robustness <- bind_rows(lapply(names(ROBUST), function(f) {
  emm_f <- lapply(ROBUST[[f]], function(r) emmeans(r$fit, ~ condition * play * round, lmer.df = "satterthwaite"))
  spec_of <- sapply(ROBUST[[f]], function(r) r$spec)
  contrast_rows(emm_f, CONFIRMATORY) %>% mutate(fit = f, spec = spec_of[outcome], .before = 1) %>% select(-role)
}))
write_csv(robustness, file.path(out_dir, "robustness.csv"))
print(as.data.frame(robustness[, c("fit", "spec", "hypothesis", "outcome", "estimate", "lower.CL", "upper.CL", "p.value")]), digits = 3)
