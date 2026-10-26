# Descriptive stage: the 12-cell mixed model per outcome on the per-play-game table.
# Input:  $OUTPUT_DATA_DIR/analysis/<experiment>/<version>/games/play_games.csv (src/analysis/games.py)
# Output: .../descriptive/cell_means.csv   emmeans grid, one row per outcome x condition x play x round
#         .../descriptive/contrasts.csv    H1-H5 (confirmatory: Holm within hypothesis, verdict) and the
#                                          registered exploratory contrasts, as weight vectors over the 12 cells
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
experiment_name <- Sys.getenv("EXPERIMENT_NAME", unset = "transfer_bandits_choice_textbox")
version         <- Sys.getenv("VERSION")
if (version == "") stop("VERSION is not set (e.g. VERSION=pilot_E Rscript src/r/descriptive.R)")
analysis_dir <- file.path(Sys.getenv("OUTPUT_DATA_DIR"), "analysis", experiment_name, version)
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
  H4 = list(w = (m(demo == "novice" & round == 2) - m(demo == "novice" & round == 1)) - (m(demo == "none" & round == 2) - m(demo == "none" & round == 1)),
            outcomes = PR, sign = c(1, 1), role = "confirmatory"),
  H5 = list(w = m(demo == "expert") - m(demo == "none" & play == "second"), outcomes = PR, sign = c(1, 1), role = "confirmatory"),
  # Exploratory contrasts from the registration's closed list: intervals reported, no verdict.
  expert_round_change = list(w = m(demo == "expert" & round == 2) - m(demo == "expert" & round == 1), outcomes = OUTCOMES, sign = c(1, 1, -1), role = "exploratory"),
  solo_round_change   = list(w = m(demo == "none" & round == 2) - m(demo == "none" & round == 1), outcomes = OUTCOMES, sign = c(NA, NA, NA), role = "exploratory"),
  H4_entropy          = list(w = (m(demo == "novice" & round == 2) - m(demo == "novice" & round == 1)) - (m(demo == "none" & round == 2) - m(demo == "none" & round == 1)),
                             outcomes = "entropy", sign = -1, role = "exploratory"),
  H5_round2           = list(w = m(demo == "expert" & round == 2) - m(demo == "none" & play == "second" & round == 2), outcomes = PR, sign = c(1, 1), role = "exploratory"),
  social_vs_solo4     = list(w = m(demo != "none") - m(demo == "none" & play == "second"), outcomes = PR, sign = c(1, 1), role = "exploratory"),
  novice_vs_solo4     = list(w = m(demo == "novice") - m(demo == "none" & play == "second"), outcomes = PR, sign = c(NA, NA), role = "exploratory")
))
stopifnot(all(sapply(CONTRASTS, function(k) abs(sum(k$w)) < 1e-12)))

contrasts <- bind_rows(lapply(names(CONTRASTS), function(h) {
  k <- CONTRASTS[[h]]
  bind_rows(lapply(seq_along(k$outcomes), function(i) {
    est <- summary(contrast(emms[[k$outcomes[i]]], method = list(w = k$w)), infer = TRUE)
    data.frame(hypothesis = h, outcome = k$outcomes[i], role = k$role, expected_sign = k$sign[i], as.data.frame(est)[, -1])
  }))
})) %>%
  group_by(hypothesis) %>%
  mutate(p_holm = if (first(role) == "confirmatory") p.adjust(p.value, method = "holm") else NA_real_) %>%
  ungroup() %>%
  mutate(verdict = case_when(role != "confirmatory" ~ NA_character_,
                             p_holm < 0.05 & sign(estimate) == expected_sign ~ "supported",
                             p_holm < 0.05 ~ "contradicted",
                             TRUE ~ "not supported"))
write_csv(contrasts, file.path(out_dir, "contrasts.csv"))
print(as.data.frame(contrasts[contrasts$role == "confirmatory", c("hypothesis", "outcome", "estimate", "lower.CL", "upper.CL", "p.value", "p_holm", "verdict")]), digits = 3)
