suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(ggplot2)
  library(patchwork)
  library(stringr)
  library(purrr)
})

# Correlation-with-demonstrator figures for issue #13.
#
# Compares how five play sources — human baseline plus four hybrid model
# variants (env / expert / novice / recent-watched priors) — correlate
# with expert and novice demonstrator pulls from the SAME round. Each
# correlation is a Pearson r over the 12-dim arm-pull-count vector of one
# play game against the 12-dim arm-pull-count vector of one demonstrator's
# same-round watch. Same-round-only: at play 1 of a round the second
# demonstrator has not yet been watched, so that correlation is NA and
# shown as a dashed 3-point line.

N_ARMS_HYBRID <- 12L

PLAY_SOURCE_LEVELS <- c("human", "env_prior", "expert_prior",
                        "novice_prior", "recent_prior")

PLAY_SOURCE_LABELS <- c(
  human        = "Human",
  env_prior    = "Hybrid: env prior",
  expert_prior = "Hybrid: expert prior",
  novice_prior = "Hybrid: novice prior",
  recent_prior = "Hybrid: recent prior"
)

BETA_METHOD_TO_PLAY_SOURCE <- c(
  theta_betas  = "env_prior",
  expert_betas = "expert_prior",
  novice_betas = "novice_prior",
  recent_betas = "recent_prior"
)

CONDITION_LEVELS <- c("expert_first", "novice_first")
CONDITION_LABELS <- c(expert_first = "Expert First",
                      novice_first = "Novice First")

# Source colors — Okabe-Ito palette, kept in lockstep with the `series` /
# `beta_method` / `method_palette` mappings elsewhere in eval_report.qmd so
# one source has one color throughout the report.
SOURCE_COLORS <- c(
  human        = "#000000",
  env_prior    = "#E69F00",
  expert_prior = "#009E73",
  novice_prior = "#CC79A7",
  recent_prior = "#56B4E9"
)

# Demonstrator colors for the per-cell correlation grid — matches the
# blue/red standard used by the N_total and mean_reward plots, plus
# reports/r/correlations.R in the analysis pipeline.
DEMO_COLORS <- c(expert = "#1f77b4", novice = "#d62728")

HUMAN_TINT <- "#FAF7F0"
NA_BAND    <- "#F1EFE8"
GRID_GREY  <- "#D3D1C7"
ZERO_INK   <- "#2C2C2A"

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

theme_hybrid_corr <- function(panel_fill = "white", base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.background = element_rect(fill = panel_fill, color = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "#F1EFE8", linewidth = 0.3),
      plot.title = element_text(face = "bold", size = base_size + 1,
                                hjust = 0.5),
      plot.subtitle = element_text(color = "#5F5E5A", size = base_size - 1),
      axis.text.x = element_text(size = base_size - 2),
      axis.title.y = element_text(face = "bold", color = "#2C2C2A"),
      legend.position = "top"
    )
}

# ---------------------------------------------------------------------------
# Data: correlation computation
# ---------------------------------------------------------------------------

compute_arm_frequencies <- function(positions) {
  as.numeric(table(factor(positions, levels = 0:(N_ARMS_HYBRID - 1))))
}

# Per (session, condition, round, play_within_round, play_source,
# demonstrator) correlation of play pulls vs same-round watch pulls,
# averaged across rollouts (human source already has one effective rollout
# per session since the CSV duplicates human rows across rollout_ids).
# Applies the within-round rule: at play 1 only the first-watched
# demonstrator is eligible.
compute_hybrid_correlations <- function(trial_df) {
  annotated <- trial_df %>%
    filter(!is.na(position), position >= 0, position < N_ARMS_HYBRID) %>%
    mutate(
      round = as.integer(floor(game_index / 4L)),
      position_in_round = as.integer(game_index %% 4L),
      condition = if_else(startsWith(as.character(game_sequence), "expert"),
                          "expert_first", "novice_first")
    )

  # Watch arm-frequency vectors per (session, round, demonstrator). Watch is
  # rollout-invariant and also identical across beta_method for human rows,
  # so we pick one canonical slice to dedupe.
  watch_vecs <- annotated %>%
    filter(game == "watch",
           learner_segment_input == "human",
           beta_method == "theta_betas",
           rollout_id == 0L,
           position_in_round %in% c(0L, 2L)) %>%
    mutate(demonstrator = case_when(
      condition == "expert_first" & position_in_round == 0L ~ "expert",
      condition == "expert_first" & position_in_round == 2L ~ "novice",
      condition == "novice_first" & position_in_round == 0L ~ "novice",
      condition == "novice_first" & position_in_round == 2L ~ "expert"
    )) %>%
    group_by(session_id, condition, round, demonstrator) %>%
    summarise(watch_freq = list(compute_arm_frequencies(position)),
              .groups = "drop")

  # Play arm-frequency vectors per (session, rollout, round, play_within_round,
  # play_source). Human rows are duplicated across rollouts/beta_methods in
  # the CSV, so we pin to a single canonical slice for the human source.
  play_human <- annotated %>%
    filter(game == "play",
           learner_segment_input == "human",
           beta_method == "theta_betas",
           rollout_id == 0L,
           position_in_round %in% c(1L, 3L)) %>%
    mutate(play_source = "human",
           play_within_round = if_else(position_in_round == 1L, 1L, 2L))

  play_model <- annotated %>%
    filter(game == "play",
           learner_segment_input == "model",
           beta_method %in% names(BETA_METHOD_TO_PLAY_SOURCE),
           position_in_round %in% c(1L, 3L)) %>%
    mutate(play_source = unname(BETA_METHOD_TO_PLAY_SOURCE[beta_method]),
           play_within_round = if_else(position_in_round == 1L, 1L, 2L))

  play_vecs <- bind_rows(play_human, play_model) %>%
    group_by(session_id, condition, round, play_within_round, play_source,
             rollout_id) %>%
    summarise(play_freq = list(compute_arm_frequencies(position)),
              .groups = "drop")

  # Within-round rule: at play_within_round == 1, only the first-watched
  # demonstrator is defined. At play_within_round == 2, both are defined.
  eligible_demos <- tribble(
    ~condition,     ~play_within_round, ~demonstrator,
    "expert_first", 1L,                 "expert",
    "expert_first", 2L,                 "expert",
    "expert_first", 2L,                 "novice",
    "novice_first", 1L,                 "novice",
    "novice_first", 2L,                 "expert",
    "novice_first", 2L,                 "novice"
  )

  per_rollout <- play_vecs %>%
    inner_join(eligible_demos,
               by = c("condition", "play_within_round"),
               relationship = "many-to-many") %>%
    inner_join(watch_vecs,
               by = c("session_id", "condition", "round", "demonstrator")) %>%
    rowwise() %>%
    mutate(correlation = suppressWarnings(
      stats::cor(unlist(play_freq), unlist(watch_freq), method = "pearson")
    )) %>%
    ungroup() %>%
    select(session_id, condition, round, play_within_round, play_source,
           rollout_id, demonstrator, correlation)

  # Collapse rollouts — one r per (session, ..., demonstrator).
  per_rollout %>%
    group_by(session_id, condition, round, play_within_round, play_source,
             demonstrator) %>%
    summarise(correlation = mean(correlation, na.rm = TRUE),
              .groups = "drop")
}

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

aggregate_for_grid <- function(corr_data) {
  corr_data %>%
    filter(!is.na(correlation)) %>%
    group_by(condition, round, play_within_round, play_source, demonstrator) %>%
    summarise(
      mean_corr = mean(correlation, na.rm = TRUE),
      se_corr   = sd(correlation, na.rm = TRUE) / sqrt(dplyr::n()),
      n_sessions = dplyr::n(),
      .groups = "drop"
    ) %>%
    mutate(play_idx = (round) * 2L + play_within_round)  # rounds are 0..2
}

aggregate_for_difference <- function(corr_data) {
  corr_data %>%
    filter(!is.na(correlation)) %>%
    pivot_wider(names_from = demonstrator, values_from = correlation) %>%
    filter(!is.na(expert) & !is.na(novice)) %>%
    mutate(diff = expert - novice,
           play_idx = (round) * 2L + play_within_round) %>%
    group_by(condition, round, play_within_round, play_idx, play_source) %>%
    summarise(
      mean_diff = mean(diff, na.rm = TRUE),
      se_diff   = sd(diff, na.rm = TRUE) / sqrt(dplyr::n()),
      n_sessions = dplyr::n(),
      .groups = "drop"
    )
}

plot_correlation_cell <- function(agg, condition_name, play_source_name,
                                  tinted = FALSE,
                                  show_y_axis = TRUE,
                                  show_x_axis = TRUE,
                                  show_col_title = FALSE,
                                  row_label = NULL) {
  d <- agg %>%
    filter(condition == condition_name,
           play_source == play_source_name)

  always_demo <- if (condition_name == "expert_first") "expert" else "novice"
  d_always    <- d %>% filter(demonstrator == always_demo)
  d_sometimes <- d %>% filter(demonstrator != always_demo)

  panel_fill <- if (tinted) HUMAN_TINT else "white"

  p <- ggplot() +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = GRID_GREY) +
    geom_vline(xintercept = c(2.5, 4.5), color = GRID_GREY, linewidth = 0.3)

  if (nrow(d_always) > 0) {
    p <- p +
      geom_line(data = d_always,
                aes(x = play_idx, y = mean_corr, color = demonstrator),
                linewidth = 0.7) +
      geom_errorbar(data = d_always,
                    aes(x = play_idx, ymin = mean_corr - se_corr,
                        ymax = mean_corr + se_corr, color = demonstrator),
                    width = 0.15, linewidth = 0.4) +
      geom_point(data = d_always,
                 aes(x = play_idx, y = mean_corr, color = demonstrator),
                 size = 2)
  }
  if (nrow(d_sometimes) > 0) {
    p <- p +
      geom_line(data = d_sometimes,
                aes(x = play_idx, y = mean_corr, color = demonstrator),
                linetype = "dashed", linewidth = 0.7, alpha = 0.7) +
      geom_errorbar(data = d_sometimes,
                    aes(x = play_idx, ymin = mean_corr - se_corr,
                        ymax = mean_corr + se_corr, color = demonstrator),
                    width = 0.15, linewidth = 0.4) +
      geom_point(data = d_sometimes,
                 aes(x = play_idx, y = mean_corr, color = demonstrator),
                 size = 2)
  }

  y_title <- if (!is.null(row_label)) row_label else (if (show_y_axis) "Pearson r" else NULL)

  p <- p +
    scale_color_manual(values = DEMO_COLORS, guide = "none") +
    scale_x_continuous(breaks = 1:6,
                       labels = c("P1", "P2", "P1", "P2", "P1", "P2")) +
    coord_cartesian(ylim = c(0, 1), xlim = c(0.7, 6.3)) +
    labs(x = NULL, y = y_title,
         title = if (show_col_title) CONDITION_LABELS[[condition_name]] else NULL) +
    theme_hybrid_corr(panel_fill = panel_fill)

  if (!show_y_axis) {
    p <- p + theme(axis.text.y = element_blank(),
                   axis.ticks.y = element_blank())
  }
  if (!show_x_axis) {
    p <- p + theme(axis.text.x = element_blank(),
                   axis.ticks.x = element_blank())
  }
  p
}

plot_correlation_grid <- function(corr_data, fig_dir = NULL) {
  agg <- aggregate_for_grid(corr_data)
  sources <- PLAY_SOURCE_LEVELS
  cells <- list()
  for (i in seq_along(sources)) {
    src <- sources[i]
    is_top    <- (i == 1)
    is_bottom <- (i == length(sources))
    for (j in seq_along(CONDITION_LEVELS)) {
      cond <- CONDITION_LEVELS[j]
      is_left <- (j == 1)
      cells[[paste(cond, src, sep = "__")]] <- plot_correlation_cell(
        agg            = agg,
        condition_name = cond,
        play_source_name = src,
        tinted         = (src == "human"),
        show_y_axis    = is_left,
        show_x_axis    = is_bottom,
        show_col_title = is_top,
        row_label      = if (is_left) PLAY_SOURCE_LABELS[[src]] else NULL
      )
    }
  }

  rows <- lapply(sources, function(src) {
    cells[[paste("expert_first", src, sep = "__")]] |
      cells[[paste("novice_first", src, sep = "__")]]
  })
  grid_fig <- Reduce(`/`, rows) +
    plot_annotation(
      title    = "Play-Game Correlation with Same-Round Demonstrator Watches",
      subtitle = paste0("Pearson r over 12 per-arm pull counts. ",
                        "Blue = correlation vs expert, red = vs novice. ",
                        "Dashed line: demonstrator not yet watched at play-1."),
      theme = theme(plot.title    = element_text(face = "bold", size = 14, hjust = 0.5),
                    plot.subtitle = element_text(size = 10, hjust = 0.5,
                                                 color = "#5F5E5A"))
    )

  if (!is.null(fig_dir)) {
    if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
    out_png <- file.path(fig_dir, "play_game_correlation_demonstrator_watches.png")
    ggsave(out_png, grid_fig, width = 10, height = 14, dpi = 300, bg = "white")
  }
  grid_fig
}

plot_difference_panel <- function(corr_data, fig_dir = NULL) {
  diff_data <- aggregate_for_difference(corr_data)
  diff_data <- diff_data %>%
    mutate(
      play_source = factor(play_source, levels = PLAY_SOURCE_LEVELS),
      linetype    = if_else(condition == "expert_first", "solid", "dashed"),
      # Human is the reference baseline so it gets a slightly heavier
      # stroke, but we keep all lines on the lighter side so the 10-line
      # panel stays legible.
      linewidth   = if_else(play_source == "human", 1.6, 0.6),
      pointsize   = if_else(play_source == "human", 2.8, 1.6)
    )

  p <- ggplot(diff_data,
              aes(x = play_idx, y = mean_diff,
                  color = play_source,
                  group = interaction(play_source, condition))) +
    annotate("rect", xmin = c(0.7, 2.7, 4.7), xmax = c(1.3, 3.3, 5.3),
             ymin = -Inf, ymax = Inf, fill = NA_BAND, alpha = 0.6) +
    annotate("text", x = c(1, 3, 5), y = 0, label = "n/a",
             color = "#888780", size = 3) +
    geom_hline(yintercept = 0, color = ZERO_INK, linewidth = 0.5) +
    geom_hline(yintercept = c(-0.5, 0.5), linetype = "dotted", color = GRID_GREY) +
    geom_vline(xintercept = c(2.5, 4.5), color = GRID_GREY, linewidth = 0.3) +
    geom_line(aes(linewidth = I(linewidth), linetype = I(linetype))) +
    geom_errorbar(aes(ymin = mean_diff - se_diff, ymax = mean_diff + se_diff),
                  width = 0.15, linewidth = 0.4) +
    geom_point(aes(size = I(pointsize))) +
    scale_color_manual(values = SOURCE_COLORS,
                       labels = PLAY_SOURCE_LABELS,
                       name = NULL) +
    scale_x_continuous(breaks = 1:6,
                       labels = c("P1", "P2", "P1", "P2", "P1", "P2")) +
    coord_cartesian(ylim = c(-1, 1), xlim = c(0.7, 6.3)) +
    labs(
      title    = "Difference: corr(expert) − corr(novice) by play source",
      subtitle = paste0("Solid = expert-first, dashed = novice-first. ",
                        "Human drawn slightly thicker as the reference baseline. ",
                        "P1 positions are structurally undefined (n/a bands)."),
      x = NULL, y = NULL
    ) +
    theme_hybrid_corr() +
    theme(legend.position = "top",
          plot.title    = element_text(face = "bold", size = 13, hjust = 0.5),
          plot.subtitle = element_text(size = 10, hjust = 0.5, color = "#5F5E5A"))

  if (!is.null(fig_dir)) {
    if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
    out_png <- file.path(fig_dir, "play_game_correlation_difference.png")
    ggsave(out_png, p, width = 13, height = 5, dpi = 300, bg = "white")
  }
  p
}

plot_correlation_figures <- function(trial_df, fig_dir = NULL) {
  corr_data <- compute_hybrid_correlations(trial_df)
  list(
    grid       = plot_correlation_grid(corr_data, fig_dir),
    difference = plot_difference_panel(corr_data, fig_dir),
    data       = corr_data
  )
}
