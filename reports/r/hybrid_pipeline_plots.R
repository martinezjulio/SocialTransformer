library(here)
library(tidyverse)
library(stringr)
library(patchwork)
library(ggtext)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HYBRID_COLORS <- list(
  env    = list(dark = "#1D9E75", light = "#9FE1CB"),
  expert = list(dark = "#BA7517", light = "#FAC775"),
  novice = list(dark = "#D4537E", light = "#F4C0D1"),
  demo   = list(dark = "#5F5E5A", light = "#D3D1C7")
)

PRIOR_ORDER <- c("env", "expert", "novice")

BETA_METHOD_TO_PRIOR <- c(
  theta_betas  = "env",
  expert_betas = "expert",
  novice_betas = "novice"
)

STAGE_ORDER <- c(
  "watch_expert",
  "play_expert_post",
  "watch_novice",
  "play_novice_post"
)

GAME_SEQUENCE_EXPERT_FIRST <- "expert-play-novice-play-choice"
GAME_SEQUENCE_NOVICE_FIRST <- "novice-play-expert-play-choice"

VALID_GAME_SEQUENCES <- c(
  GAME_SEQUENCE_EXPERT_FIRST,
  GAME_SEQUENCE_NOVICE_FIRST
)

# Display labels for stages — underscore-free; play rows drop the post-demo
# suffix because the surrounding watch rows already make clear which demo's
# belief drove the play.
STAGE_DISPLAY <- c(
  watch_expert     = "watch expert",
  watch_novice     = "watch novice",
  play_expert_post = "play",
  play_novice_post = "play"
)

stage_label <- function(stage_name) unname(STAGE_DISPLAY[stage_name])

# Package-local carrier for the canonical rank→payoff map so `rank_axis_scale`
# can format tick labels as "R1 (0.80)" without plumbing an argument through
# every row builder. Composers call set_rank_payoffs() before rendering.
.hybrid_env <- new.env(parent = emptyenv())
.hybrid_env$rank_payoffs <- NULL
set_rank_payoffs <- function(p) { .hybrid_env$rank_payoffs <- p }
get_rank_payoffs <- function()  { .hybrid_env$rank_payoffs }

# Derive canonical payoff-per-rank from any single zone_bandit in the data —
# all bandits in the study share the same 12 Bernoulli payoffs (only the
# arm→payoff assignment is permuted).
canonical_rank_payoffs <- function(trial_df) {
  sample_str <- trial_df$zone_bandit[which(!is.na(trial_df$zone_bandit) &
                                           nzchar(trial_df$zone_bandit))[1]]
  if (is.na(sample_str)) return(NULL)
  p <- parse_zone_bandit(sample_str)
  if (length(p) != 12 || anyNA(p)) return(NULL)
  sort(p, decreasing = TRUE)  # index i -> payoff for rank R{i}
}

# Build success / failure fills for play rows: prior color saturated for
# success, desaturated for failure. Returns named character vector keyed
# by "prior_outcome".
hybrid_play_fill_values <- function() {
  vals <- c()
  for (prior in PRIOR_ORDER) {
    vals[paste0(prior, "_success")] <- HYBRID_COLORS[[prior]]$dark
    vals[paste0(prior, "_failure")] <- HYBRID_COLORS[[prior]]$light
  }
  vals
}

theme_hybrid <- function(base_size = 12) {
  theme_classic(base_size = base_size) +
    theme(
      plot.background   = element_rect(fill = "white", colour = NA),
      panel.background  = element_rect(fill = "white", colour = NA),
      panel.grid.major.y = element_line(colour = "#EEEEEC", linewidth = 0.25),
      panel.grid.major.x = element_blank(),
      panel.grid.minor  = element_blank(),
      plot.margin       = margin(6, 10, 6, 10),
      axis.title.x      = element_blank(),  # "rank" label is redundant across 20+ rows
      axis.title.y      = element_text(size = base_size, face = "bold"),
      axis.text.y       = element_text(size = base_size - 1),
      axis.text.x       = element_text(size = base_size - 1,
                                       angle = 45, hjust = 1, vjust = 1),
      legend.position   = "bottom",
      legend.key.size   = unit(12, "pt"),
      legend.text       = element_text(size = base_size - 1),
      strip.background  = element_blank(),
      strip.text        = element_text(face = "bold"),
      plot.title        = element_text(size = base_size + 2, face = "bold")
    )
}

rank_axis_scale <- function() {
  # Explicit limits so every row renders all 12 tick positions regardless of
  # which ranks have non-zero data (e.g. an expert watch stage that only
  # pulls 3 distinct ranks still shows R1..R12 on the axis).
  # When canonical payoffs are registered via set_rank_payoffs(), annotate
  # each tick with the actual payoff, e.g. "R1 (0.80)".
  rp <- get_rank_payoffs()
  labels <- if (is.null(rp)) paste0("R", 1:12) else
    sprintf("R%d (%.2f)", 1:12, rp)
  scale_x_continuous(
    breaks = 1:12,
    labels = labels,
    limits = c(0.5, 12.5),
    expand = expansion(add = 0)
  )
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Parse a zone_bandit string like "0.6,0.1,0.8,..." into a numeric vector.
# Copied from reports/r/individual_behavior.R so this module is self-contained.
parse_zone_bandit <- function(s) {
  if (is.numeric(s) && length(s) > 1) return(as.numeric(s))
  s <- as.character(s[1])
  if (is.na(s) || s == "") return(rep(NA_real_, 12))
  nums <- regmatches(
    s,
    gregexpr("-?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?", s, perl = TRUE)
  )[[1]]
  as.numeric(nums)
}

# Parse a beta-list string like "[(2.05, 2.04), (1.91, 1.90), ...]" into a
# 12-row tibble(arm_id, alpha, beta). Regex matches those used in
# reports/r/experience_plots.R.
parse_beta_list <- function(beta_str) {
  if (is.na(beta_str) || beta_str == "" || length(beta_str) == 0) {
    return(tibble(arm_id = 0:11, alpha = NA_real_, beta = NA_real_))
  }
  matches <- str_match_all(beta_str, "\\(([0-9.]+),\\s*([0-9.]+)\\)")[[1]]
  if (nrow(matches) == 0) {
    return(tibble(arm_id = 0:11, alpha = NA_real_, beta = NA_real_))
  }
  tibble(
    arm_id = seq_len(nrow(matches)) - 1L,
    alpha  = as.numeric(matches[, 2]),
    beta   = as.numeric(matches[, 3])
  )
}

# Compute per-bandit rank where R1 = best (highest true payoff) and
# R12 = worst. NOTE: reports/r/individual_behavior.R uses the opposite
# convention (R1 = worst) via rank(payoffs). We invert here to match the
# issue-#11 spec. Returns a tibble(arm_id, rank, payoff).
compute_ranks <- function(zone_bandit_str) {
  payoffs <- parse_zone_bandit(zone_bandit_str)
  if (length(payoffs) != 12 || anyNA(payoffs)) {
    return(tibble(arm_id = 0:11, rank = NA_real_, payoff = NA_real_))
  }
  tibble(
    arm_id  = 0:11,
    payoff  = payoffs,
    rank    = 13 - rank(payoffs, ties.method = "average")
  )
}

session_label_from_id <- function(sid) {
  s <- as.character(sid)[1]
  s <- sub("^sid[_-]?", "", s)
  s <- sub(".*_sid[_-]?", "", s)
  gsub("[^A-Za-z0-9_-]", "_", s)
}

# Map (game_sequence, within_round_index ∈ 0..3) → stage name.
stage_from_sequence_and_index <- function(game_sequence, within_round_idx) {
  is_expert_first <- game_sequence == GAME_SEQUENCE_EXPERT_FIRST
  slot <- within_round_idx %% 4L
  dplyr::case_when(
    slot == 0L & is_expert_first  ~ "watch_expert",
    slot == 0L & !is_expert_first ~ "watch_novice",
    slot == 1L & is_expert_first  ~ "play_expert_post",
    slot == 1L & !is_expert_first ~ "play_novice_post",
    slot == 2L & is_expert_first  ~ "watch_novice",
    slot == 2L & !is_expert_first ~ "watch_expert",
    slot == 3L & is_expert_first  ~ "play_novice_post",
    slot == 3L & !is_expert_first ~ "play_expert_post",
    TRUE ~ NA_character_
  )
}

# Cutoff in the per-cutoff CSV that corresponds to "after watch expert" or
# "after watch novice" for a given round (watch-first schedule: each round
# contributes [bandit_sep, D1, L1, D2, L2, query] = 6 cutoffs).
cutoff_for <- function(round_index, stage_after, game_sequence) {
  base <- 6L * as.integer(round_index)
  is_expert_first <- game_sequence == GAME_SEQUENCE_EXPERT_FIRST
  if (stage_after == "watch_expert") {
    if (is_expert_first) base + 2L else base + 4L
  } else if (stage_after == "watch_novice") {
    if (is_expert_first) base + 4L else base + 2L
  } else {
    NA_integer_
  }
}

# ---------------------------------------------------------------------------
# Data-shape builders
# ---------------------------------------------------------------------------

# Per-round tibble(round_index, game_sequence, arm_id, rank, payoff,
# zone_global_index). For synthetic data there may be only one
# learner_segment_input value.
build_session_ranks_by_round <- function(trial_df, session_id) {
  sess <- trial_df %>% filter(session_id == !!session_id)
  sess %>%
    distinct(round_index, game_sequence, zone_global_index, zone_bandit) %>%
    rowwise() %>%
    mutate(ranks = list(compute_ranks(zone_bandit))) %>%
    ungroup() %>%
    select(round_index, game_sequence, zone_global_index, ranks) %>%
    unnest(ranks)
}

# Pulls / successes per (round_index, stage, rank) from the watch rows.
# Uses a single canonical slice so each trial contributes once.
# Fills missing (rank) cells with zero so every row has 12 bars, even for
# ranks the demonstrator never pulled.
build_stage_df <- function(trial_df, session_id) {
  sess <- trial_df %>% filter(session_id == !!session_id)
  canonical_beta <- if ("theta_betas" %in% sess$beta_method) "theta_betas" else
    sess$beta_method[1]
  canonical_input <- if ("model" %in% sess$learner_segment_input) "model" else
    sess$learner_segment_input[1]
  watch_canonical <- sess %>%
    filter(game == "watch",
           rollout_id == 0L,
           beta_method == canonical_beta,
           learner_segment_input == canonical_input)
  ranks <- build_session_ranks_by_round(trial_df, session_id)
  agg <- watch_canonical %>%
    mutate(within_round_idx = game_index - 4L * round_index,
           stage = stage_from_sequence_and_index(game_sequence, within_round_idx)) %>%
    left_join(ranks, by = c("round_index", "game_sequence", "zone_global_index",
                            "position" = "arm_id")) %>%
    group_by(round_index, stage, rank) %>%
    summarise(
      successes = sum(as.logical(reward_generated), na.rm = TRUE),
      failures  = sum(!as.logical(reward_generated), na.rm = TRUE),
      pulls     = dplyr::n(),
      .groups   = "drop"
    )
  # Fill in zero rows for any rank the demonstrator skipped so every row
  # shows all 12 bars.
  agg %>%
    filter(!is.na(stage), !is.na(rank)) %>%
    tidyr::complete(
      tidyr::nesting(round_index, stage),
      rank = 1:12,
      fill = list(successes = 0, failures = 0, pulls = 0)
    )
}

# Per-session play-prior pulls/successes per (round_index, stage, prior, rank).
# Fills missing (rank, prior) cells with zero so grouped bars are
# consistent across all 12 ranks.
#
# `all_rollouts = FALSE` keeps the single-rollout slice used by per-session
# figures (readable: one trajectory per session). `all_rollouts = TRUE`
# averages pulls/successes across all rollouts per (round_index, stage,
# prior, rank) so cross-session figures reflect the full N rollouts per
# session rather than just rollout 0. (Watch data is rollout-invariant, so
# build_stage_df doesn't need this param — see that function's comment.)
build_play_prior_df <- function(trial_df, session_id, all_rollouts = FALSE) {
  sess <- trial_df %>% filter(session_id == !!session_id)
  plays <- sess %>%
    filter(game == "play",
           learner_segment_input == "model",
           beta_method %in% names(BETA_METHOD_TO_PRIOR))
  if (!all_rollouts) {
    plays <- plays %>% filter(rollout_id == 0L)
  }
  plays <- plays %>% mutate(prior = unname(BETA_METHOD_TO_PRIOR[beta_method]))
  ranks <- build_session_ranks_by_round(trial_df, session_id)
  per_rollout <- plays %>%
    mutate(within_round_idx = game_index - 4L * round_index,
           stage = stage_from_sequence_and_index(game_sequence, within_round_idx)) %>%
    left_join(ranks, by = c("round_index", "game_sequence", "zone_global_index",
                            "position" = "arm_id")) %>%
    group_by(rollout_id, round_index, stage, prior, rank) %>%
    summarise(
      successes = sum(as.logical(reward_generated), na.rm = TRUE),
      failures  = sum(!as.logical(reward_generated), na.rm = TRUE),
      pulls     = dplyr::n(),
      .groups   = "drop"
    )
  agg <- per_rollout %>%
    group_by(round_index, stage, prior, rank) %>%
    summarise(
      successes = mean(successes, na.rm = TRUE),
      failures  = mean(failures, na.rm = TRUE),
      pulls     = mean(pulls, na.rm = TRUE),
      .groups   = "drop"
    )
  agg %>%
    filter(!is.na(stage), !is.na(rank), !is.na(prior)) %>%
    tidyr::complete(
      tidyr::nesting(round_index, stage),
      prior = PRIOR_ORDER,
      rank = 1:12,
      fill = list(successes = 0, failures = 0, pulls = 0)
    )
}

# Build per-session predictions for the 3 targets (env/expert/novice) at the
# "after_watch_expert" and "after_watch_novice" cutoffs in each round.
build_prediction_df <- function(cutoff_df, session_id, context_beta_method = "theta_betas",
                                ranks_by_round = NULL) {
  sess_cut <- cutoff_df %>%
    filter(session_id == !!session_id,
           beta_method == context_beta_method,
           segment_type == "demo")
  if (nrow(sess_cut) == 0) return(tibble())

  if (is.null(ranks_by_round)) {
    stop("ranks_by_round must be provided to build_prediction_df")
  }

  out <- list()
  rounds <- sort(unique(sess_cut$round_index))
  for (r in rounds) {
    game_seq <- ranks_by_round %>%
      filter(round_index == r) %>%
      dplyr::pull(game_sequence) %>%
      dplyr::first()
    ranks_r <- ranks_by_round %>% filter(round_index == r)
    for (stage_after in c("watch_expert", "watch_novice")) {
      target_cutoff <- cutoff_for(r, stage_after, game_seq)
      row <- sess_cut %>%
        filter(cutoff == target_cutoff, round_index == r) %>%
        summarise(
          theta_betas  = dplyr::first(theta_betas),
          expert_betas = dplyr::first(expert_betas),
          novice_betas = dplyr::first(novice_betas),
          true_payoffs = dplyr::first(true_payoffs)
        )
      if (nrow(row) == 0) next
      target_map <- list(env = row$theta_betas,
                         expert = row$expert_betas,
                         novice = row$novice_betas)
      tp <- parse_zone_bandit(row$true_payoffs)
      for (prior_name in PRIOR_ORDER) {
        ab <- parse_beta_list(target_map[[prior_name]])
        if (all(is.na(ab$alpha))) next
        entry <- ab %>%
          left_join(ranks_r, by = "arm_id") %>%
          mutate(
            round_index = r,
            stage_after = stage_after,
            cutoff = target_cutoff,
            prior = prior_name,
            mean = alpha / (alpha + beta),
            lo   = qbeta(0.025, alpha, beta),
            hi   = qbeta(0.975, alpha, beta),
            true_payoff = tp[arm_id + 1L]
          ) %>%
          select(round_index, stage_after, cutoff, prior, rank, arm_id,
                 alpha, beta, mean, lo, hi, true_payoff)
        out[[length(out) + 1]] <- entry
      }
    }
  }
  if (length(out) == 0) return(tibble())
  dplyr::bind_rows(out)
}

# Aggregate across sessions for a given game_sequence value.
build_cross_session_frames <- function(trial_df, cutoff_df, game_sequence_value,
                                       context_beta_method = "theta_betas") {
  trial_cond <- trial_df %>% filter(game_sequence == !!game_sequence_value)
  cutoff_cond <- cutoff_df %>% filter(game_sequence == !!game_sequence_value,
                                      beta_method == context_beta_method)
  session_ids <- unique(trial_cond$session_id)

  stage_frames <- list()
  play_frames  <- list()
  predict_frames <- list()

  for (sid in session_ids) {
    ranks_sid <- build_session_ranks_by_round(trial_cond, sid)
    stage_sid <- build_stage_df(trial_cond, sid) %>% mutate(session_id = sid)
    play_sid  <- build_play_prior_df(trial_cond, sid, all_rollouts = TRUE) %>%
      mutate(session_id = sid)
    pred_sid  <- build_prediction_df(cutoff_cond, sid, context_beta_method, ranks_sid) %>%
      mutate(session_id = sid)
    stage_frames[[length(stage_frames) + 1]] <- stage_sid
    play_frames[[length(play_frames) + 1]]   <- play_sid
    predict_frames[[length(predict_frames) + 1]] <- pred_sid
  }

  stage_all <- dplyr::bind_rows(stage_frames)
  play_all  <- dplyr::bind_rows(play_frames)
  pred_all  <- dplyr::bind_rows(predict_frames)

  # per-session stage/play frames already include zero rows for skipped ranks
  # (see build_stage_df / build_play_prior_df), so cross-session means treat
  # "rank never pulled" as 0 pulls rather than a missing observation.
  stage_summary <- stage_all %>%
    mutate(rate = ifelse(pulls > 0, successes / pulls, NA_real_)) %>%
    group_by(round_index, stage, rank) %>%
    summarise(
      n_sessions = dplyr::n(),
      mean_pulls = mean(pulls, na.rm = TRUE),
      se_pulls   = sd(pulls, na.rm = TRUE) / sqrt(dplyr::n()),
      mean_rate  = mean(rate, na.rm = TRUE),
      se_rate    = sd(rate, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups = "drop"
    )
  play_summary <- play_all %>%
    mutate(rate = ifelse(pulls > 0, successes / pulls, NA_real_)) %>%
    group_by(round_index, stage, prior, rank) %>%
    summarise(
      n_sessions = dplyr::n(),
      mean_pulls = mean(pulls, na.rm = TRUE),
      se_pulls   = sd(pulls, na.rm = TRUE) / sqrt(dplyr::n()),
      mean_rate  = mean(rate, na.rm = TRUE),
      se_rate    = sd(rate, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups = "drop"
    )
  predict_summary <- pred_all %>%
    filter(!is.na(mean)) %>%
    group_by(round_index, stage_after, cutoff, prior, rank) %>%
    summarise(
      n_sessions     = dplyr::n(),
      mean_of_mean   = mean(mean, na.rm = TRUE),
      se_of_mean     = sd(mean, na.rm = TRUE) / sqrt(dplyr::n()),
      mean_lo        = mean(lo, na.rm = TRUE),
      mean_hi        = mean(hi, na.rm = TRUE),
      mean_true_payoff = mean(true_payoff, na.rm = TRUE),
      .groups = "drop"
    )

  list(stage = stage_summary, play_prior = play_summary, predict = predict_summary)
}

# ---------------------------------------------------------------------------
# Row builders (per-session)
# ---------------------------------------------------------------------------

row_permutation_strip <- function(ranks_df) {
  if (nrow(ranks_df) == 0 || all(is.na(ranks_df$rank))) {
    return(ggplot() + theme_void())
  }
  ggplot(ranks_df, aes(x = rank, y = 1L, fill = payoff)) +
    geom_tile(colour = "white", width = 0.95, height = 0.9) +
    geom_text(aes(label = arm_id), colour = "white", size = 3, fontface = "bold") +
    scale_fill_gradient(low = "#D3D1C7", high = "#1D9E75", limits = c(0, 1),
                        guide = "none") +
    rank_axis_scale() +
    coord_cartesian(ylim = c(0.5, 1.5)) +
    theme_hybrid() +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.title = element_blank(),
          axis.text.x = element_blank())
}

row_watch_session <- function(stage_df, stage_name, round_idx) {
  d <- stage_df %>%
    filter(round_index == round_idx, stage == stage_name) %>%
    pivot_longer(cols = c(successes, failures),
                 names_to = "outcome", values_to = "count") %>%
    mutate(outcome = factor(outcome, levels = c("failures", "successes")))
  demo_cols <- c(successes = HYBRID_COLORS$demo$dark,
                 failures  = HYBRID_COLORS$demo$light)
  ggplot(d, aes(x = rank, y = count, fill = outcome)) +
    geom_col(width = 0.75) +
    scale_fill_manual(values = demo_cols,
                      labels = c(successes = "success", failures = "failure"),
                      name = NULL) +
    rank_axis_scale() +
    labs(y = stage_label(stage_name)) +
    theme_hybrid()
}

# Per-session predictions row: 3 dodged CIs per rank (env/expert/novice)
# plus the true-payoff tick. Optional ghost_means draws transparent dots
# at a prior row's posterior means (used for row 5 to compare to row 2).
row_predict_session <- function(pred_df, round_idx, stage_after, ghost_means = NULL) {
  d <- pred_df %>% filter(round_index == round_idx, stage_after == !!stage_after)
  if (nrow(d) == 0) {
    return(ggplot() + theme_void() +
             labs(title = sprintf("No predictions for round %d", round_idx + 1L)))
  }
  d$prior <- factor(d$prior, levels = PRIOR_ORDER)
  prior_colors <- sapply(PRIOR_ORDER, function(p) HYBRID_COLORS[[p]]$dark)
  dodge <- position_dodge(width = 0.6)

  p <- ggplot(d, aes(x = rank, colour = prior, group = prior)) +
    geom_linerange(aes(ymin = lo, ymax = hi), position = dodge, linewidth = 0.8) +
    geom_point(aes(y = mean), position = dodge, size = 1.5) +
    geom_segment(aes(x = rank - 0.35, xend = rank + 0.35,
                     y = true_payoff, yend = true_payoff),
                 colour = "#2C2C2A", linewidth = 0.6, inherit.aes = FALSE) +
    scale_colour_manual(values = prior_colors, name = NULL) +
    rank_axis_scale() +
    coord_cartesian(ylim = c(0, 1)) +
    labs(y = "posterior\nmean / 95% CI") +
    theme_hybrid()

  if (!is.null(ghost_means) && nrow(ghost_means) > 0) {
    ghosts <- ghost_means %>% filter(round_index == round_idx)
    if (nrow(ghosts) > 0) {
      ghosts$prior <- factor(ghosts$prior, levels = PRIOR_ORDER)
      p <- p +
        geom_point(data = ghosts,
                   aes(x = rank, y = mean, colour = prior, group = prior),
                   position = dodge, alpha = 0.3, size = 1.3,
                   shape = 21, fill = "white",
                   inherit.aes = FALSE)
    }
  }
  p
}

row_play_session <- function(play_prior_df, stage_name, round_idx) {
  d <- play_prior_df %>%
    filter(round_index == round_idx, stage == stage_name) %>%
    mutate(prior = factor(prior, levels = PRIOR_ORDER)) %>%
    pivot_longer(cols = c(successes, failures),
                 names_to = "outcome", values_to = "count") %>%
    mutate(outcome = recode(outcome, successes = "success", failures = "failure"),
           outcome = factor(outcome, levels = c("failure", "success")),
           fill_key = paste(prior, outcome, sep = "_"))
  if (nrow(d) == 0) {
    return(ggplot() + theme_void())
  }
  fill_vals <- hybrid_play_fill_values()
  dodge <- position_dodge2(width = 0.8, preserve = "single", padding = 0.1)
  ggplot(d, aes(x = rank, y = count, group = prior, fill = fill_key)) +
    geom_col(width = 0.75, position = dodge) +
    scale_fill_manual(values = fill_vals, guide = "none") +
    rank_axis_scale() +
    labs(y = stage_label(stage_name)) +
    theme_hybrid()
}

# ---------------------------------------------------------------------------
# Row builders (cross-session)
# ---------------------------------------------------------------------------

row_watch_cross_pulls <- function(stage_cross_df, stage_name, round_idx) {
  d <- stage_cross_df %>% filter(round_index == round_idx, stage == stage_name)
  if (nrow(d) == 0) return(ggplot() + theme_void())
  ggplot(d, aes(x = rank, y = mean_pulls)) +
    geom_col(width = 0.75, fill = HYBRID_COLORS$demo$dark) +
    geom_errorbar(aes(ymin = pmax(mean_pulls - se_pulls, 0),
                      ymax = mean_pulls + se_pulls),
                  width = 0.25, colour = "#2C2C2A", linewidth = 0.4) +
    rank_axis_scale() +
    labs(y = stage_label(stage_name)) +
    theme_hybrid()
}

row_watch_cross_rate <- function(stage_cross_df, stage_name, round_idx) {
  d <- stage_cross_df %>% filter(round_index == round_idx, stage == stage_name)
  if (nrow(d) == 0) return(ggplot() + theme_void())
  ggplot(d, aes(x = rank, y = mean_rate)) +
    geom_col(width = 0.75, fill = HYBRID_COLORS$demo$dark) +
    geom_errorbar(aes(ymin = pmax(mean_rate - se_rate, 0),
                      ymax = pmin(mean_rate + se_rate, 1)),
                  width = 0.25, colour = "#2C2C2A", linewidth = 0.4) +
    rank_axis_scale() +
    coord_cartesian(ylim = c(0, 1)) +
    labs(y = "success rate") +
    theme_hybrid()
}

row_predict_cross <- function(predict_cross_df, round_idx, stage_after) {
  d <- predict_cross_df %>% filter(round_index == round_idx, stage_after == !!stage_after)
  if (nrow(d) == 0) return(ggplot() + theme_void())
  d$prior <- factor(d$prior, levels = PRIOR_ORDER)
  prior_colors <- sapply(PRIOR_ORDER, function(p) HYBRID_COLORS[[p]]$dark)
  dodge <- position_dodge(width = 0.6)

  ggplot(d, aes(x = rank, colour = prior, group = prior)) +
    geom_linerange(aes(ymin = mean_lo, ymax = mean_hi),
                   position = dodge, linewidth = 1.3, alpha = 0.6) +
    geom_linerange(aes(ymin = pmax(mean_of_mean - se_of_mean, 0),
                       ymax = pmin(mean_of_mean + se_of_mean, 1)),
                   position = dodge, linewidth = 0.6) +
    geom_point(aes(y = mean_of_mean), position = dodge, size = 1.3) +
    geom_segment(aes(x = rank - 0.35, xend = rank + 0.35,
                     y = mean_true_payoff, yend = mean_true_payoff),
                 colour = "#2C2C2A", linewidth = 0.5, inherit.aes = FALSE) +
    scale_colour_manual(values = prior_colors, name = NULL) +
    rank_axis_scale() +
    coord_cartesian(ylim = c(0, 1)) +
    labs(y = "posterior\nmean 95% CI") +
    theme_hybrid()
}

row_play_cross_pulls <- function(play_cross_df, stage_name, round_idx) {
  d <- play_cross_df %>% filter(round_index == round_idx, stage == stage_name)
  if (nrow(d) == 0) return(ggplot() + theme_void())
  d$prior <- factor(d$prior, levels = PRIOR_ORDER)
  prior_colors <- sapply(PRIOR_ORDER, function(p) HYBRID_COLORS[[p]]$dark)
  dodge <- position_dodge(width = 0.6)
  ggplot(d, aes(x = rank, y = mean_pulls, fill = prior, group = prior)) +
    geom_col(width = 0.5, position = dodge) +
    geom_errorbar(aes(ymin = pmax(mean_pulls - se_pulls, 0),
                      ymax = mean_pulls + se_pulls),
                  position = dodge, width = 0.2, colour = "#2C2C2A",
                  linewidth = 0.3) +
    scale_fill_manual(values = prior_colors, name = NULL) +
    rank_axis_scale() +
    labs(y = stage_label(stage_name)) +
    theme_hybrid()
}

row_play_cross_rate <- function(play_cross_df, stage_name, round_idx) {
  d <- play_cross_df %>% filter(round_index == round_idx, stage == stage_name)
  if (nrow(d) == 0) return(ggplot() + theme_void())
  d$prior <- factor(d$prior, levels = PRIOR_ORDER)
  prior_colors <- sapply(PRIOR_ORDER, function(p) HYBRID_COLORS[[p]]$dark)
  dodge <- position_dodge(width = 0.6)
  ggplot(d, aes(x = rank, y = mean_rate, fill = prior, group = prior)) +
    geom_col(width = 0.5, position = dodge) +
    geom_errorbar(aes(ymin = pmax(mean_rate - se_rate, 0),
                      ymax = pmin(mean_rate + se_rate, 1)),
                  position = dodge, width = 0.2, colour = "#2C2C2A",
                  linewidth = 0.3) +
    scale_fill_manual(values = prior_colors, name = NULL) +
    rank_axis_scale() +
    coord_cartesian(ylim = c(0, 1)) +
    labs(y = "success rate") +
    theme_hybrid()
}

# ---------------------------------------------------------------------------
# Composers
# ---------------------------------------------------------------------------

plot_session <- function(trial_df, cutoff_df, session_id,
                         context_beta_method = "theta_betas") {
  stage_df <- build_stage_df(trial_df, session_id)
  play_prior_df <- build_play_prior_df(trial_df, session_id)
  ranks_df <- build_session_ranks_by_round(trial_df, session_id)
  pred_df <- build_prediction_df(cutoff_df, session_id, context_beta_method, ranks_df)

  # Register canonical rank payoffs so row builders can annotate R1..R12 ticks.
  set_rank_payoffs(canonical_rank_payoffs(trial_df))
  on.exit(set_rank_payoffs(NULL), add = TRUE)

  rounds <- sort(unique(ranks_df$round_index))
  if (length(rounds) == 0) return(ggplot() + theme_void())

  row_heights_per_round <- c(0.4, 1, 1.3, 1.3, 1, 1.3, 1.3)
  plots <- list()
  heights <- numeric(0)
  for (r in rounds) {
    ranks_r <- ranks_df %>% filter(round_index == r)
    game_seq_r <- dplyr::first(ranks_r$game_sequence)
    round_title <- sprintf("Round %d  ·  %s", r + 1L,
                           if (game_seq_r == GAME_SEQUENCE_EXPERT_FIRST)
                             "Expert First" else "Novice First")

    strip <- row_permutation_strip(ranks_r) + labs(title = round_title)
    w_exp <- row_watch_session(stage_df, "watch_expert", r)
    pred_after_exp <- row_predict_session(pred_df, r, "watch_expert")
    play_exp <- row_play_session(play_prior_df, "play_expert_post", r)
    w_nov <- row_watch_session(stage_df, "watch_novice", r)
    ghost_means <- pred_df %>%
      filter(stage_after == "watch_expert") %>%
      select(round_index, rank, prior, mean)
    pred_after_nov <- row_predict_session(pred_df, r, "watch_novice", ghost_means)
    play_nov <- row_play_session(play_prior_df, "play_novice_post", r)

    plots <- c(plots, list(strip, w_exp, pred_after_exp, play_exp,
                           w_nov, pred_after_nov, play_nov))
    heights <- c(heights, row_heights_per_round)
  }

  combined <- patchwork::wrap_plots(plots, ncol = 1) +
    plot_layout(heights = heights, guides = "collect")
  sid_label <- session_label_from_id(session_id)
  game_seq <- dplyr::first(ranks_df$game_sequence)
  combined +
    plot_annotation(
      title    = sprintf("Hybrid Pipeline — Session %s", sid_label),
      subtitle = sprintf("Game sequence: %s  ·  context: beta_method=%s",
                         game_seq, context_beta_method)
    )
}

plot_cross_session <- function(trial_df, cutoff_df, game_sequence_value,
                               context_beta_method = "theta_betas") {
  set_rank_payoffs(canonical_rank_payoffs(trial_df))
  on.exit(set_rank_payoffs(NULL), add = TRUE)

  frames <- build_cross_session_frames(trial_df, cutoff_df, game_sequence_value,
                                       context_beta_method)
  rounds <- sort(unique(c(frames$stage$round_index,
                          frames$play_prior$round_index,
                          frames$predict$round_index)))
  if (length(rounds) == 0) return(ggplot() + theme_void())

  row_heights_per_round <- c(0.25, 0.9, 0.9, 1.3, 0.9, 0.9,
                             0.9, 0.9, 1.3, 0.9, 0.9)
  plots <- list()
  heights <- numeric(0)
  for (r in rounds) {
    round_title <- sprintf("Round %d  ·  %s", r + 1L,
                           if (game_sequence_value == GAME_SEQUENCE_EXPERT_FIRST)
                             "Expert First (cross-session)" else
                             "Novice First (cross-session)")

    strip <- ggplot() + theme_void() +
      annotate("text", x = 0.5, y = 0.5,
               label = "R1 (best) … R12 (worst)  ·  rank axis is invariant across sessions",
               hjust = 0.5, vjust = 0.5, size = 3.5, fontface = "italic") +
      labs(title = round_title)

    w_exp_pulls <- row_watch_cross_pulls(frames$stage, "watch_expert", r)
    w_exp_rate  <- row_watch_cross_rate (frames$stage, "watch_expert", r)
    pred_after_exp <- row_predict_cross(frames$predict, r, "watch_expert")
    play_exp_pulls <- row_play_cross_pulls(frames$play_prior, "play_expert_post", r)
    play_exp_rate  <- row_play_cross_rate (frames$play_prior, "play_expert_post", r)
    w_nov_pulls <- row_watch_cross_pulls(frames$stage, "watch_novice", r)
    w_nov_rate  <- row_watch_cross_rate (frames$stage, "watch_novice", r)
    pred_after_nov <- row_predict_cross(frames$predict, r, "watch_novice")
    play_nov_pulls <- row_play_cross_pulls(frames$play_prior, "play_novice_post", r)
    play_nov_rate  <- row_play_cross_rate (frames$play_prior, "play_novice_post", r)

    plots <- c(plots, list(strip,
                           w_exp_pulls, w_exp_rate,
                           pred_after_exp,
                           play_exp_pulls, play_exp_rate,
                           w_nov_pulls, w_nov_rate,
                           pred_after_nov,
                           play_nov_pulls, play_nov_rate))
    heights <- c(heights, row_heights_per_round)
  }
  combined <- patchwork::wrap_plots(plots, ncol = 1) +
    plot_layout(heights = heights, guides = "collect")
  combined +
    plot_annotation(
      title    = sprintf("Hybrid Pipeline — %s (cross-session)", game_sequence_value),
      subtitle = sprintf("Aggregated across %d sessions · context: beta_method=%s",
                         length(unique(trial_df$session_id[trial_df$game_sequence ==
                                                             game_sequence_value])),
                         context_beta_method)
    )
}

# ---------------------------------------------------------------------------
# Saver
# ---------------------------------------------------------------------------

save_per_session_figures <- function(trial_df, cutoff_df, fig_dir,
                                     context_beta_method = "theta_betas",
                                     width = 18, height = 48, dpi = 150) {
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  valid_sess <- trial_df %>%
    filter(game_sequence %in% VALID_GAME_SEQUENCES) %>%
    distinct(session_id, game_sequence) %>%
    arrange(session_id)
  if (nrow(valid_sess) == 0) {
    message("No watch-first sessions to render.")
    return(invisible(character(0)))
  }
  paths <- character(nrow(valid_sess))
  for (i in seq_len(nrow(valid_sess))) {
    sid <- valid_sess$session_id[i]
    seq_short <- if (valid_sess$game_sequence[i] == GAME_SEQUENCE_EXPERT_FIRST)
      "expert_first" else "novice_first"
    fname <- sprintf("session_%s_%s.png", session_label_from_id(sid), seq_short)
    out <- file.path(fig_dir, fname)
    message(sprintf("  [%d/%d] rendering session %s -> %s",
                    i, nrow(valid_sess), sid, fname))
    p <- tryCatch(
      plot_session(trial_df, cutoff_df, sid, context_beta_method),
      error = function(e) {
        warning(sprintf("session %s failed: %s", sid, conditionMessage(e)))
        NULL
      }
    )
    if (is.null(p)) next
    ggsave(out, p, width = width, height = height, dpi = dpi, bg = "white",
           limitsize = FALSE)
    paths[i] <- out
  }
  invisible(paths[paths != ""])
}
