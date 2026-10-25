# Descriptive stage: the 12-cell mixed model per outcome on the per-play-game table.
# Input:  $OUTPUT_DATA_DIR/analysis/<experiment>/<version>/games/play_games.csv (src/analysis/games.py)
# Output: .../descriptive/cell_means.csv   emmeans grid, one row per outcome x condition x play x round
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
cell_means <- bind_rows(lapply(OUTCOMES, function(y) {
  emmeans(fits[[y]], ~ condition * play * round, lmer.df = "satterthwaite") %>%
    as.data.frame() %>%
    mutate(outcome = y, .before = 1)
}))

write_csv(cell_means, file.path(out_dir, "cell_means.csv"))
cat("Wrote", nrow(cell_means), "cell means for", length(OUTCOMES), "outcomes to", file.path(out_dir, "cell_means.csv"), "\n")
