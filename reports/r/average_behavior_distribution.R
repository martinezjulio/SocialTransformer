# Dependencies
library(here)
library(tidyverse)
library(janitor)
library(ggthemes)
library(stringr)
library(ggtext)
library(ggplot2)

#' Create Average Behavior Distribution Plots
#'
#' - Axis tick labels are "payoff, rank" in plain text format.
#' - Rotation is handled by guide_axis(angle = 90).
#'
#' @param preprocessed_df Preprocessed data frame
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plot to disk
#' @param use_geom_richtext_ticks Fallback: if TRUE, hide the axis labels and draw
#'   rich tick labels with geom_richtext at the bottom of each panel. Defaults to FALSE.
#' @return A ggplot object showing the average distribution grid
average_behavior_distribution <- function(preprocessed_df, experiment_name, version, save = FALSE,
                                          use_geom_richtext_ticks = FALSE, fig_dir = NULL) {

  # Create the figure directory if saving
  if (save) {
    if (is.null(fig_dir)) fig_dir <- here("src", "analysis", "figures", experiment_name, version)
    if (!dir.exists(fig_dir)) {
      dir.create(fig_dir, recursive = TRUE)
    }
  }
  
  n_arms <- 12L
  
  # Helper function to parse zone_bandit string
  parse_zone_bandit <- function(s) {
    if (is.numeric(s) && length(s) > 1) return(as.numeric(s))
    s <- as.character(s[1])
    nums <- regmatches(
      s,
      gregexpr("-?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?", s, perl = TRUE)
    )[[1]]
    as.numeric(nums)
  }
  
  # Map position (0-11) to payoff rank (1=worst, 12=best) based on zone_bandit
  position_to_rank <- function(position, zone_bandit_str) {
    if (is.na(position) || is.na(zone_bandit_str) || zone_bandit_str == "") return(NA_real_)
    
    payoffs <- parse_zone_bandit(zone_bandit_str)
    if (length(payoffs) != 12) return(NA_real_)
    if (position < 0 || position > 11) return(NA_real_)
    
    ranks <- rank(payoffs, ties.method = "average")
    ranks[position + 1]
  }
  
  # Helper function to compute rank frequency vector
  compute_rank_frequencies <- function(ranks) {
    freq <- table(factor(ranks, levels = 1:n_arms))
    as.numeric(freq)
  }
  
  # Map positions to ranks and compute rank frequencies per game for each participant
  game_rank_freqs <- preprocessed_df %>%
    dplyr::filter(game != "choice", action == "shoot",
           !is.na(position), position >= 0, position < n_arms) %>%
    dplyr::mutate(
      game_sequence_label = dplyr::case_when(
        str_starts(game_sequence, "expert") ~ "Expert First Condition",
        str_starts(game_sequence, "novice") ~ "Novice First Condition",
        TRUE ~ game_sequence
      ),
      round_num = floor(game_index / 4) + 1,
      position_in_round = game_index %% 4,
      game_type_label = dplyr::case_when(
        game_sequence_label == "Expert First Condition" & position_in_round == 0 ~ "Expert",
        game_sequence_label == "Expert First Condition" & position_in_round == 1 ~ "Participants",
        game_sequence_label == "Expert First Condition" & position_in_round == 2 ~ "Novice",
        game_sequence_label == "Expert First Condition" & position_in_round == 3 ~ "Participants",
        game_sequence_label == "Novice First Condition" & position_in_round == 0 ~ "Novice",
        game_sequence_label == "Novice First Condition" & position_in_round == 1 ~ "Participants",
        game_sequence_label == "Novice First Condition" & position_in_round == 2 ~ "Expert",
        game_sequence_label == "Novice First Condition" & position_in_round == 3 ~ "Participants",
        TRUE ~ game
      ),
      arm_rank = purrr::map2_dbl(position, zone_bandit, position_to_rank)
    ) %>%
    dplyr::filter(!is.na(arm_rank)) %>%
    dplyr::group_by(across(any_of(c(
      "anonymous_id", "game_sequence_label", "game_index", "game",
      "game_type_label", "round_num", "rollout_id"
    )))) %>%
    dplyr::summarise(
      rank_frequencies = list(compute_rank_frequencies(arm_rank)),
      .groups = "drop"
    )
  
  # Convert counts to proportions for each participant, then calculate mean and SE
  average_rank_proportions <- game_rank_freqs %>%
    dplyr::group_by(game_sequence_label, game_index, game, game_type_label, round_num) %>%
    dplyr::summarise(
      stats = list({
        all_freqs <- do.call(rbind, rank_frequencies)
        proportions <- all_freqs / rowSums(all_freqs)
        n <- nrow(proportions)
        means <- colMeans(proportions, na.rm = TRUE)
        sds <- apply(proportions, 2, sd, na.rm = TRUE)
        ses <- sds / sqrt(n)
        list(mean = means, se = ses, sd = sds)
      }),
      n_participants = dplyr::n(),
      .groups = "drop"
    )
  
  # Unnest to long format for plotting
  plot_data <- average_rank_proportions %>%
    dplyr::mutate(
      rank = purrr::map(stats, ~1:n_arms),
      mean = purrr::map(stats, ~.x$mean),
      se = purrr::map(stats, ~.x$se),
      sd = purrr::map(stats, ~.x$sd)
    ) %>%
    dplyr::select(-stats) %>%
    tidyr::unnest(c(rank, mean, se, sd)) %>%
    dplyr::mutate(
      percentage = mean * 100,
      se_percentage = se * 100,
      ymin = pmax(0, percentage - se_percentage),
      ymax = pmin(100, percentage + se_percentage),
      game_sequence_label = factor(game_sequence_label, 
                                   levels = c("Expert First Condition", "Novice First Condition"))
    )
  
  # Factors for ordering
  plot_data <- plot_data %>%
    dplyr::mutate(
      rank_factor = factor(rank, levels = 1:n_arms),
      game_index_factor = factor(game_index, levels = 0:11)
    )
  
  # ---- Simple labels for the x-axis ----
  payoffs <- c("0.00","0.10","0.20","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.75","0.80")
  ranks_chr <- sprintf("%02d", 1:n_arms)  # Two-digit format: 01, 02, ..., 12
  # Simple plain text: payoff, rank (with extra spacing)
  rank_labels_vec <- paste0(payoffs, ",   ", ranks_chr)
  rank_labels_named <- stats::setNames(rank_labels_vec, as.character(1:n_arms))
  
  # Column labels for facets
  game_type_map <- plot_data %>%
    dplyr::distinct(game_index, game_type_label, game_sequence_label, round_num) %>%
    dplyr::group_by(game_index) %>%
    dplyr::summarise(
      round_num = dplyr::first(round_num),
      expert_first_type = dplyr::first(game_type_label[game_sequence_label == "Expert First Condition"]),
      novice_first_type = dplyr::first(game_type_label[game_sequence_label == "Novice First Condition"]),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      col_label = paste0("Game ", game_index, ", Round ", round_num, "\nTop: ", expert_first_type, "\nBottom: ", novice_first_type)
    )
  game_labels <- stats::setNames(game_type_map$col_label, game_type_map$game_index)
  
  # ---- Base plot ----
  p_dist <- plot_data %>%
    ggplot(aes(x = rank_factor, y = percentage)) +
    geom_col(aes(fill = game_type_label), width = 0.8, alpha = 0.8) +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), width = 0.4, alpha = 0.7, linewidth = 0.3) +
    scale_fill_manual(
      values = c("Expert" = "#1f77b4", "Novice" = "#d62728", "Participants" = "#2ca02c"),
      name = "Game Type"
    ) +
    facet_grid(
      game_sequence_label ~ game_index_factor,
      scales = "free_y",
      labeller = labeller(game_index_factor = game_labels, game_sequence_label = label_value)
    ) +
    # Use named vector + guide_axis for rotation; element_markdown renders the HTML.
    scale_x_discrete(labels = rank_labels_named,
                     breaks = levels(plot_data$rank_factor),
                     drop = FALSE,
                     guide = ggplot2::guide_axis(angle = 90, check.overlap = FALSE)) +
    labs(
      title = "Average Arm-Pull Distribution by Rank Across Games and Rounds",
      subtitle = "Expert First Condition: Expert-Play-Novice-Play-Choice; Novice First Condition: Novice-Play-Expert-Play-Choice. \u00B11 SE for Error Bars.",
      x = "Payoff / Rank (1=lowest, 12=highest)",
      y = "Average Percentage of Pulls (%)"
    ) +
    theme_minimal(base_size = 9) +
    theme(
      strip.text.x = element_text(size = 6, face = "bold", lineheight = 0.9),
      strip.text.y = element_text(size = 9, face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
      plot.subtitle = element_text(hjust = 0.5, size = 8, color = "gray40"),
      legend.position = "bottom",
      panel.spacing = unit(0.3, "lines"),
      axis.text.x = element_text(size = 7, hjust = 1, vjust = 0.5),
      axis.text.y = element_text(size = 6),
      axis.title = element_text(size = 8),
      axis.title.x = element_text(size = 8, margin = margin(t = 10, r = 0, b = 0, l = 0)),
      panel.background = element_rect(fill = "white", colour = "grey90"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.grid.major = element_line(color = "grey95", linewidth = 0.3),
      panel.grid.minor = element_blank()
    )
  
  # ---- Optional fallback: draw tick labels with geom_richtext ----
  # Note: This fallback is no longer needed with plain text labels,
  # but kept for compatibility if use_geom_richtext_ticks is still passed.
  if (isTRUE(use_geom_richtext_ticks)) {
    tick_df <- plot_data %>%
      dplyr::distinct(game_sequence_label, game_index_factor, rank_factor) %>%
      dplyr::mutate(label = rank_labels_named[as.character(rank_factor)])
    
    p_dist <- p_dist +
      theme(axis.text.x = element_blank()) +
      ggtext::geom_richtext(
        data = tick_df,
        aes(x = rank_factor, y = 0, label = label),
        inherit.aes = FALSE,
        angle = 90,
        vjust = 1.05,      # just below the axis
        size = 2,
        label.color = NA,  # no border
        fill = NA
      ) +
      coord_cartesian(clip = "off")
  }
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_average_behavior_distribution.png")),
      plot = p_dist, width = 20, height = 7, dpi = 300, bg = "white"
    )
    message("Saved average behavior distribution plot to: ",
            file.path(fig_dir, paste0(experiment_name, "_", version, "_average_behavior_distribution.png")))
  }
  
  return(p_dist)
}
