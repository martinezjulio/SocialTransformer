# Descriptive stage, second script: the two levels behind each registered contrast, for the contrast_H* figures.
# Refits the model of src/r/descriptive.R (outcome ~ condition * play * round + (1 | participant), sum-to-zero coding,
# REML, Satterthwaite df) on games/play_games.csv, stops unless it reproduces descriptive/cell_means.csv and
# descriptive/contrasts.csv, and writes descriptive/contrast_levels.csv: for H1, H2, H3, H5 and each outcome, the mean of
# each side's cells with its 95% CI (a between-participant interval), and the difference (the registered contrast).
# No number of the report comes from this file. Run after descriptive.R:  VERSION=pilot_E Rscript src/r/contrast_levels.R
suppressPackageStartupMessages({ library(here); library(dplyr); library(readr); library(lme4); library(lmerTest); library(emmeans) })
if (Sys.getenv("OUTPUT_DATA_DIR") == "") dotenv::load_dot_env(here::here(".env"))
version <- Sys.getenv("VERSION"); if (version == "") stop("VERSION is not set")
analysis_dir <- file.path(Sys.getenv("OUTPUT_DATA_DIR"), "analysis", version)
out <- file.path(analysis_dir, "descriptive", "contrast_levels.csv")

OUTCOMES <- c("payoff", "reward", "entropy")
games <- read_csv(file.path(analysis_dir, "games", "play_games.csv"), show_col_types = FALSE) %>% filter(!exploratory) %>%
  mutate(participant = factor(participant), condition = factor(condition, levels = c("expert-first", "novice-first", "solo")),
         play = factor(play, levels = c("first", "second")), round = factor(round, levels = c(1, 2)))
stopifnot(nrow(games) == 12 * nlevels(games$participant), all(table(games$participant) == 12))
options(contrasts = c("contr.sum", "contr.poly"))
emm_options(lmerTest.limit = 1e5)
fits <- lapply(setNames(OUTCOMES, OUTCOMES), function(y) lmer(reformulate("condition * play * round + (1 | participant)", response = y), data = games, REML = TRUE))
emms <- lapply(fits, function(fit) emmeans(fit, ~ condition * play * round, lmer.df = "satterthwaite"))

# check 1: the cell means of the registered run
cells <- bind_rows(lapply(OUTCOMES, function(y) mutate(as.data.frame(emms[[y]]), outcome = y, .before = 1)))
ref_cells <- read_csv(file.path(analysis_dir, "descriptive", "cell_means.csv"), show_col_types = FALSE) %>% mutate(round = factor(round, levels = c(1, 2)))
j <- inner_join(cells, ref_cells, by = c("outcome", "condition", "play", "round"), suffix = c("", ".ref"))
stopifnot(nrow(j) == 36)
cat(sprintf("cell means: max |refit - registered| = %.2e (estimate), %.2e (CI bounds)\n", max(abs(j$emmean - j$emmean.ref)),
            max(abs(c(j$lower.CL - j$lower.CL.ref, j$upper.CL - j$upper.CL.ref)))))
stopifnot(max(abs(j$emmean - j$emmean.ref)) < 1e-6, max(abs(c(j$lower.CL - j$lower.CL.ref, j$upper.CL - j$upper.CL.ref))) < 1e-6)

grid <- as.data.frame(emms$payoff)[, c("condition", "play", "round")]
grid$demo <- with(grid, ifelse(condition == "solo", "none", ifelse((condition == "expert-first") == (play == "first"), "expert", "novice")))
m <- function(sel) as.numeric(sel) / sum(sel)
SIDES <- with(grid, list(   # left side first (the level subtracted), then the right side
  H1 = list(left = m(demo == "novice"), right = m(demo == "expert"), outcomes = c("payoff", "entropy")),
  H2 = list(left = m(demo == "novice" & round == 1), right = m(demo == "novice" & round == 2), outcomes = c("payoff", "entropy")),
  H3 = list(left = m(demo != "none" & round == 1), right = m(demo != "none" & round == 2), outcomes = "entropy"),
  H5 = list(left = m(demo == "none" & play == "second"), right = m(demo == "expert"), outcomes = "payoff")))
rows <- bind_rows(lapply(names(SIDES), function(h) bind_rows(lapply(SIDES[[h]]$outcomes, function(y) {
  s <- SIDES[[h]]
  est <- as.data.frame(confint(contrast(emms[[y]], method = list(left = s$left, right = s$right, diff = s$right - s$left)), level = 0.95))
  data.frame(hypothesis = h, outcome = y, side = est$contrast, estimate = est$estimate, SE = est$SE, df = est$df, lower = est$lower.CL, upper = est$upper.CL)
}))))

# check 2: the differences against the registered contrasts
ref <- read_csv(file.path(analysis_dir, "descriptive", "contrasts.csv"), show_col_types = FALSE)
d <- rows %>% filter(side == "diff") %>% inner_join(ref, by = c("hypothesis", "outcome"), suffix = c("", ".ref"))
stopifnot(nrow(d) == sum(rows$side == "diff"))
cat(sprintf("contrasts:  max |refit - registered| = %.2e (estimate), %.2e (CI bounds)\n", max(abs(d$estimate - d$estimate.ref)),
            max(abs(c(d$lower - d$lower.CL, d$upper - d$upper.CL)))))
stopifnot(max(abs(d$estimate - d$estimate.ref)) < 1e-6, max(abs(c(d$lower - d$lower.CL, d$upper - d$upper.CL))) < 1e-6)
write_csv(rows, out)
print(rows %>% mutate(across(c(estimate, lower, upper), ~ round(.x, 3))) %>% select(hypothesis, outcome, side, estimate, lower, upper), row.names = FALSE)
cat("Wrote", out, "\n")
