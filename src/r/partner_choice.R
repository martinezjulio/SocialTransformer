# Partner choice (S4): every social session ends with one pick, expert or novice.
# Input:  $EXPERIMENT_DATA_DIR/<experiment>/<version>/preprocessed_participant_data.csv (the choice game is
#         not tokenised, so it appears in no inference CSV)
# Output: $OUTPUT_DATA_DIR/analysis/<experiment>/<version>/partner_choice/rates.csv   expert-choice rate with
#         Wilson 95% intervals: all sessions, expert-first (novice seen last), novice-first (expert seen last)
#         .../partner_choice/model.csv  S4: P(choose expert) > 0.5 from a mixed logistic model with a participant
#         intercept, two-sided alpha 0.05 with sign (confirmatory); the recency check (exploratory): the
#         condition term of the same model with sum-to-zero coding, expected ~0 under expertise-following
# Run:    VERSION=pilot_E Rscript src/r/partner_choice.R

suppressPackageStartupMessages({ library(here); library(dplyr); library(readr); library(lme4) })
if (Sys.getenv("OUTPUT_DATA_DIR") == "") dotenv::load_dot_env(here::here(".env"))
experiment_name <- Sys.getenv("EXPERIMENT_NAME", unset = "transfer_bandits_choice_textbox")
version         <- Sys.getenv("VERSION")
if (version == "") stop("VERSION is not set (e.g. VERSION=pilot_E Rscript src/r/partner_choice.R)")
out_dir <- file.path(Sys.getenv("OUTPUT_DATA_DIR"), "analysis", experiment_name, version, "partner_choice")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1) One row per social session: the shot of the choice game names the demonstrator picked.
raw <- read_csv(file.path(Sys.getenv("EXPERIMENT_DATA_DIR"), experiment_name, version, "preprocessed_participant_data.csv"),
                col_types = cols(.default = col_character()))
choices <- raw %>%
  filter(game == "choice", shot == "TRUE") %>%
  transmute(participant = factor(anonymous_id), session = session_id,
            condition = factor(ifelse(game_sequence == "expert-play-novice-play-choice", "expert-first", "novice-first"),
                               levels = c("expert-first", "novice-first")),
            chose_expert = as.integer(choice_demonstrator_type == "expert"))
stopifnot(!anyNA(choices$chose_expert), n_distinct(choices$session) == nrow(choices),
          all(table(choices$participant, choices$condition) == 1))   # one pick per session, both conditions per participant

# 2) Raw rates with Wilson intervals (prop.test without continuity correction is the Wilson score interval).
wilson <- function(k, n) { ci <- prop.test(k, n, correct = FALSE)$conf.int; tibble(k = k, n = n, p = k / n, lo = ci[1], hi = ci[2]) }
rates <- bind_rows(
  mutate(wilson(sum(choices$chose_expert), nrow(choices)), group = "all", last_seen = "mixed", .before = 1),
  choices %>% group_by(group = as.character(condition)) %>% group_modify(~ wilson(sum(.x$chose_expert), nrow(.x))) %>% ungroup() %>%
    mutate(last_seen = ifelse(group == "expert-first", "novice", "expert"), .after = group))
write_csv(rates, file.path(out_dir, "rates.csv"))

# 3) S4 (intercept) and the recency check (condition term), one mixed logistic model each.
options(contrasts = c("contr.sum", "contr.poly"))
fit_row <- function(formula, term, role, expected_sign) {
  fit <- glmer(formula, data = choices, family = binomial)
  co <- coef(summary(fit))[term, ]
  est <- unname(co["Estimate"]); se <- unname(co["Std. Error"]); p <- unname(co["Pr(>|z|)"])
  tibble(role = role, term = term, estimate = est, se = se, z = unname(co["z value"]), p.value = p,
         lower = est - qnorm(0.975) * se, upper = est + qnorm(0.975) * se, expected_sign = expected_sign,
         participant_sd = sqrt(as.numeric(VarCorr(fit)$participant)), singular = isSingular(fit),
         verdict = ifelse(role != "confirmatory", NA, ifelse(p >= 0.05, "not supported", ifelse(sign(est) == expected_sign, "supported", "contradicted"))))
}
model <- bind_rows(
  fit_row(chose_expert ~ 1 + (1 | participant), "(Intercept)", "confirmatory", 1),             # S4: logit P(expert) > 0
  fit_row(chose_expert ~ condition + (1 | participant), "condition1", "exploratory", NA))     # recency: expert-first minus grand mean
model <- mutate(model, across(c(estimate, lower, upper), ~ ifelse(term == "(Intercept)", plogis(.x), NA), .names = "prob_{.col}"), .after = upper)
write_csv(model, file.path(out_dir, "model.csv"))
print(as.data.frame(rates), digits = 3); print(as.data.frame(model[, c("role", "term", "estimate", "lower", "upper", "p.value", "prob_estimate", "singular", "verdict")]), digits = 3)
