# Partner choice (H4, the registration's fourth primary; was H6 until 28 Oct): every social session ends with one pick, expert or novice.
# Input:  $EXPERIMENT_DATA_DIR/<version>/preprocessed_participant_data.csv (the choice game is
#         not tokenised, so it appears in no inference CSV)
# Output: $OUTPUT_DATA_DIR/analysis/<version>/partner_choice/choices.csv  one row per session: participant, condition, chose_expert
#         .../partner_choice/rates.csv   expert-choice rate with
#         Wilson 95% intervals: all sessions, expert-first (novice seen last), novice-first (expert seen last)
#         .../partner_choice/model.csv  one mixed logistic model, chose_expert ~ condition + (1 | participant), sum-to-zero
#         coding. H4: its intercept, P(choose expert) > 0.5 averaged over the two conditions, two-sided alpha 0.05 with
#         sign (confirmatory); the recency check (exploratory): its condition term, expected ~0 under expertise-following
#         .../partner_choice/numbers.json  H4 rate, interval, p, verdict, count and n; the recency term: keys as the report
#         stage's manifest lists them (src/analysis/numbers.py)
# Run:    VERSION=pilot_E Rscript src/r/partner_choice.R

suppressPackageStartupMessages({ library(here); library(dplyr); library(readr); library(lme4) })
if (Sys.getenv("OUTPUT_DATA_DIR") == "") dotenv::load_dot_env(here::here(".env"))
version         <- Sys.getenv("VERSION")
if (version == "") stop("VERSION is not set (e.g. VERSION=pilot_E Rscript src/r/partner_choice.R)")
out_dir <- file.path(Sys.getenv("OUTPUT_DATA_DIR"), "analysis", version, "partner_choice")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1) One row per social session: the shot of the choice game names the demonstrator picked.
raw <- read_csv(file.path(Sys.getenv("EXPERIMENT_DATA_DIR"), version, "preprocessed_participant_data.csv"),
                col_types = cols(.default = col_character()))
choices <- raw %>%
  filter(game == "choice", shot == "TRUE") %>%
  transmute(participant = factor(anonymous_id), session = session_id,
            condition = factor(ifelse(game_sequence == "expert-play-novice-play-choice", "expert-first", "novice-first"),
                               levels = c("expert-first", "novice-first")),
            chose_expert = as.integer(choice_demonstrator_type == "expert"))
stopifnot(!anyNA(choices$chose_expert), n_distinct(choices$session) == nrow(choices),
          all(table(choices$participant, choices$condition) == 1))   # one pick per session, both conditions per participant

write_csv(choices, file.path(out_dir, "choices.csv"))   # one row per session; read by power_analysis.R

# 2) Raw rates with Wilson intervals (prop.test without continuity correction is the Wilson score interval).
wilson <- function(k, n) { ci <- prop.test(k, n, correct = FALSE)$conf.int; tibble(k = k, n = n, p = k / n, lo = ci[1], hi = ci[2]) }
rates <- bind_rows(
  mutate(wilson(sum(choices$chose_expert), nrow(choices)), group = "all", last_seen = "mixed", .before = 1),
  choices %>% group_by(group = as.character(condition)) %>% group_modify(~ wilson(sum(.x$chose_expert), nrow(.x))) %>% ungroup() %>%
    mutate(last_seen = ifelse(group == "expert-first", "novice", "expert"), .after = group))
write_csv(rates, file.path(out_dir, "rates.csv"))

# 3) One mixed logistic model: H4 is its intercept (the grand-mean log-odds under sum-to-zero coding, since every participant
#    has one row per condition) and the recency check its condition term (expert-first minus the grand mean, half the
#    expert-first minus novice-first difference; advisor, 3 Nov).
options(contrasts = c("contr.sum", "contr.poly"))
fit <- glmer(chose_expert ~ condition + (1 | participant), data = choices, family = binomial)
term_row <- function(term, role, expected_sign) {
  co <- coef(summary(fit))[term, ]
  est <- unname(co["Estimate"]); se <- unname(co["Std. Error"]); p <- unname(co["Pr(>|z|)"])
  tibble(role = role, term = term, estimate = est, se = se, z = unname(co["z value"]), p.value = p,
         lower = est - qnorm(0.975) * se, upper = est + qnorm(0.975) * se, expected_sign = expected_sign,
         participant_sd = sqrt(as.numeric(VarCorr(fit)$participant)), singular = isSingular(fit),
         verdict = ifelse(role != "confirmatory", NA, ifelse(p >= 0.05, "not supported", ifelse(sign(est) == expected_sign, "supported", "contradicted"))))
}
model <- bind_rows(term_row("(Intercept)", "confirmatory", 1),    # H4: logit P(expert) > 0
                   term_row("condition1", "exploratory", NA))     # recency: expert-first minus grand mean
model <- mutate(model, across(c(estimate, lower, upper), ~ ifelse(term == "(Intercept)", plogis(.x), NA), .names = "prob_{.col}"), .after = upper)
write_csv(model, file.path(out_dir, "model.csv"))
h4 <- model[model$role == "confirmatory", ]; rc <- model[model$role == "exploratory", ]; all <- rates[rates$group == "all", ]
numbers <- list(H4_choice_estimate = h4$prob_estimate, H4_choice_lo = h4$prob_lower, H4_choice_hi = h4$prob_upper, H4_choice_p = h4$p.value,
                H4_choice_verdict = h4$verdict, H4_choice_count = all$k, H4_choice_n = all$n,
                recency_estimate = rc$estimate, recency_lo = rc$lower, recency_hi = rc$upper, recency_p = rc$p.value)
jsonlite::write_json(numbers, file.path(out_dir, "numbers.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
print(as.data.frame(rates), digits = 3); print(as.data.frame(model[, c("role", "term", "estimate", "lower", "upper", "p.value", "prob_estimate", "singular", "verdict")]), digits = 3)
