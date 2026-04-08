library(here)
library(tidyverse)
library(janitor)
library(ggthemes)
library(stringr)
library(ggtext)

#' Create Behavior Distribution Plots
#'
#' Creates a grid of histograms showing aggregate arm-pulling distributions across
#' all games and rounds, organized by condition.
#' - Axis tick labels are "payoff, rank" in plain text format.
#' - Rotation is handled by guide_axis(angle = 90).
#'
#' @param preprocessed_df Preprocessed data frame with columns: anonymous_id, 
#'   session_id, game_sequence, game_index, game, position, action
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plot to disk
#' @return A ggplot object showing the distribution grid
aggregate_behavior_distribution <- function(preprocessed_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

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
  # This ensures each game's positions are mapped to ranks based on that game's specific bandit configuration
  position_to_rank <- function(position, zone_bandit_str) {
    if (is.na(position) || is.na(zone_bandit_str) || zone_bandit_str == "") return(NA_real_)
    
    payoffs <- parse_zone_bandit(zone_bandit_str)
    if (length(payoffs) != 12) return(NA_real_)
    if (position < 0 || position > 11) return(NA_real_)
    
    # Rank arms: 1 = lowest payoff (worst), 12 = highest payoff (best)
    ranks <- rank(payoffs, ties.method = "average")
    ranks[position + 1]
  }
  
  # Helper function to compute rank frequency vector (instead of position frequencies)
  compute_rank_frequencies <- function(ranks) {
    freq <- table(factor(ranks, levels = 1:n_arms))
    as.numeric(freq)
  }
  
  # Map positions to ranks and compute rank frequencies per game for each participant
  game_rank_freqs <- preprocessed_df %>%
    filter(game != "choice", action == "shoot",
           !is.na(position), position >= 0, position < n_arms) %>%
    mutate(
      # Create cleaner game sequence labels
      game_sequence_label = case_when(
        str_starts(game_sequence, "expert") ~ "Expert First Condition",
        str_starts(game_sequence, "novice") ~ "Novice First Condition",
        TRUE ~ game_sequence
      ),
      # Calculate round number (games 0-3 = round 1, games 4-7 = round 2, games 8-11 = round 3)
      round_num = floor(game_index / 4) + 1,
      # Determine position within round (0-3)
      position_in_round = game_index %% 4,
      # Create descriptive game type label based on condition and position in round
      game_type_label = case_when(
        # Expert first condition
        game_sequence_label == "Expert First Condition" & position_in_round == 0 ~ "Expert",
        game_sequence_label == "Expert First Condition" & position_in_round == 1 ~ "Participants",
        game_sequence_label == "Expert First Condition" & position_in_round == 2 ~ "Novice",
        game_sequence_label == "Expert First Condition" & position_in_round == 3 ~ "Participants",
        # Novice first condition
        game_sequence_label == "Novice First Condition" & position_in_round == 0 ~ "Novice",
        game_sequence_label == "Novice First Condition" & position_in_round == 1 ~ "Participants",
        game_sequence_label == "Novice First Condition" & position_in_round == 2 ~ "Expert",
        game_sequence_label == "Novice First Condition" & position_in_round == 3 ~ "Participants",
        TRUE ~ game
      ),
      # Map each position to its rank within this game's bandit configuration
      arm_rank = purrr::map2_dbl(position, zone_bandit, position_to_rank)
    ) %>%
    filter(!is.na(arm_rank)) %>%
    group_by(anonymous_id, game_sequence_label, game_index, game, game_type_label, round_num) %>%
    summarise(
      rank_frequencies = list(compute_rank_frequencies(arm_rank)),
      .groups = "drop"
    )
  
  # Sum total rank pulls across all participants (raw counts)
  total_rank_counts <- game_rank_freqs %>%
    group_by(game_sequence_label, game_index, game, game_type_label, round_num) %>%
    summarise(
      # Sum the frequency vectors across all participants/demonstrators
      total_counts = list({
        all_freqs <- do.call(rbind, rank_frequencies)
        colSums(all_freqs, na.rm = TRUE)
      }),
      .groups = "drop"
    )
  
  # Unnest to long format for plotting
  plot_data <- total_rank_counts %>%
    mutate(rank = map(total_counts, ~1:n_arms)) %>%
    unnest(c(rank, total_counts)) %>%
    rename(count = total_counts) %>%
    mutate(
      game_sequence_label = factor(game_sequence_label, 
                                   levels = c("Expert First Condition", "Novice First Condition"))
    )
  
  # Create factor for proper ordering (1=worst to 12=best)
  plot_data <- plot_data %>%
    mutate(
      rank_factor = factor(rank, levels = 1:n_arms),
      game_index_factor = factor(game_index, levels = 0:11)
    )
  
  # Create named vector for rank labels (plain text format)
  payoffs <- c("0.00","0.10","0.20","0.30","0.35","0.40","0.45","0.50","0.55","0.60","0.75","0.80")
  ranks_chr <- sprintf("%02d", 1:n_arms)  # Two-digit format: 01, 02, ..., 12
  rank_labels_vec <- paste0(payoffs, ",   ", ranks_chr)
  rank_labels_vec <- stats::setNames(rank_labels_vec, as.character(1:n_arms))
  
  # Create game type mapping for column labels (show what each game is for each condition)
  game_type_map <- plot_data %>%
    distinct(game_index, game_type_label, game_sequence_label, round_num) %>%
    group_by(game_index) %>%
    summarise(
      round_num = first(round_num),
      # Get game types for each condition
      expert_first_type = first(game_type_label[game_sequence_label == "Expert First Condition"]),
      novice_first_type = first(game_type_label[game_sequence_label == "Novice First Condition"]),
      .groups = "drop"
    ) %>%
    mutate(
      # Create label showing game types for both conditions
      col_label = paste0("Game ", game_index, ", Round ", round_num, "\nTop: ", expert_first_type, "\nBottom: ", novice_first_type)
    )
  
  # Create named vector for labeller
  game_labels <- setNames(game_type_map$col_label, game_type_map$game_index)
  
  # Create the grid plot
  p_dist <- plot_data %>%
    ggplot(aes(x = rank_factor, y = count)) +
    geom_col(aes(fill = game_type_label), width = 0.8, alpha = 0.8) +
    scale_fill_manual(
      values = c(
        "Expert" = "#1f77b4",       # Blue for expert demonstrations
        "Novice" = "#d62728",       # Red for novice demonstrations
        "Participants" = "#2ca02c"  # Green for participant play
      ),
      name = "Game Type"
    ) +
    facet_grid(
      game_sequence_label ~ game_index_factor,
      scales = "free_y",
      labeller = labeller(
        game_index_factor = game_labels,
        game_sequence_label = label_value
      )
    ) +
    scale_x_discrete(labels = rank_labels_vec,
                     breaks = levels(plot_data$rank_factor),
                     drop = FALSE,
                     guide = ggplot2::guide_axis(angle = 90, check.overlap = FALSE)) +
    labs(
      title = "Total Arm-Pull Counts by Rank Across Games and Rounds",
      subtitle = "Expert First Condition: Expert-Play-Novice-Play-Choice; Novice First Condition: Novice-Play-Expert-Play-Choice.",
      x = "Payoff / Rank (1=lowest, 12=highest)",
      y = "Total Pull Count"
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
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_aggregate_behavior_distribution.png")),
      plot = p_dist, width = 20, height = 7, dpi = 300, bg = "white"
    )
    message("Saved behavior distribution plot to: ",
            file.path(fig_dir, paste0(experiment_name, "_", version, "_aggregate_behavior_distribution.png")))
  }
  
  # Return the plot object
  return(p_dist)
}

