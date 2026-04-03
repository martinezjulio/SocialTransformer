library(here)
library(tidyverse)
library(janitor)
library(ggthemes)
library(stringr)
library(ggtext)

#' Create Average Behavior Distribution Plots
#'
#' Creates a grid showing average proportion of arm-pulls by rank across
#' all games and rounds, organized by condition. Unlike behavior_distribution,
#' this normalizes each participant's behavior to proportions first, then averages.
#'
#' @param preprocessed_df Preprocessed data frame
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plot to disk
#' @return A ggplot object showing the average distribution grid
behavior_distribution_average <- function(preprocessed_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

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
    
    # Rank arms: 1 = lowest payoff (worst), 12 = highest payoff (best)
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
    filter(game != "choice", action == "shoot",
           !is.na(position), position >= 0, position < n_arms) %>%
    mutate(
      game_sequence_label = case_when(
        str_starts(game_sequence, "expert") ~ "Expert First Condition",
        str_starts(game_sequence, "novice") ~ "Novice First Condition",
        TRUE ~ game_sequence
      ),
      round_num = floor(game_index / 4) + 1,
      position_in_round = game_index %% 4,
      game_type_label = case_when(
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
    filter(!is.na(arm_rank)) %>%
    group_by(anonymous_id, game_sequence_label, game_index, game, game_type_label, round_num) %>%
    summarise(
      rank_frequencies = list(compute_rank_frequencies(arm_rank)),
      .groups = "drop"
    )
  
  # Convert counts to proportions for each participant, then calculate mean and SE
  average_rank_proportions <- game_rank_freqs %>%
    group_by(game_sequence_label, game_index, game, game_type_label, round_num) %>%
    summarise(
      # Calculate mean and standard error of proportions across participants
      stats = list({
        all_freqs <- do.call(rbind, rank_frequencies)
        # Convert each row (participant) to proportions
        proportions <- all_freqs / rowSums(all_freqs)
        n <- nrow(proportions)
        
        # Calculate mean and standard error for each rank
        means <- colMeans(proportions, na.rm = TRUE)
        sds <- apply(proportions, 2, sd, na.rm = TRUE)
        ses <- sds / sqrt(n)
        
        list(
          mean = means,
          se = ses,
          sd = sds
        )
      }),
      n_participants = n(),
      .groups = "drop"
    )
  
  # Unnest to long format for plotting
  plot_data <- average_rank_proportions %>%
    mutate(
      rank = map(stats, ~1:n_arms),
      mean = map(stats, ~.x$mean),
      se = map(stats, ~.x$se),
      sd = map(stats, ~.x$sd)
    ) %>%
    select(-stats) %>%
    unnest(c(rank, mean, se, sd)) %>%
    mutate(
      percentage = mean * 100,  # Convert to percentage for display
      se_percentage = se * 100,  # Convert SE to percentage
      ymin = pmax(0, percentage - se_percentage),  # Lower bound (can't go below 0)
      ymax = pmin(100, percentage + se_percentage),  # Upper bound (can't exceed 100)
      game_sequence_label = factor(game_sequence_label, 
                                   levels = c("Expert First Condition", "Novice First Condition"))
    )
  
  # Create factor for proper ordering with simple numeric labels
  plot_data <- plot_data %>%
    mutate(
      rank_factor = factor(rank, levels = 1:n_arms),
      game_index_factor = factor(game_index, levels = 0:11)
    )
  
  # Create a vector of labels with HTML formatting (for scale_x_discrete)
  rank_labels_vec <- c(
    "0.0, <span style='font-weight:bold;'>1</span>",
    "0.10, <span style='font-weight:bold;'>2</span>",
    "0.20, <span style='font-weight:bold;'>3</span>",
    "0.30, <span style='font-weight:bold;'>4</span>",
    "0.35, <span style='font-weight:bold;'>5</span>",
    "0.40, <span style='font-weight:bold;'>6</span>",
    "0.45, <span style='font-weight:bold;'>7</span>",
    "0.50, <span style='font-weight:bold;'>8</span>",
    "0.55, <span style='font-weight:bold;'>9</span>",
    "0.60, <span style='font-weight:bold;'>10</span>",
    "0.75, <span style='font-weight:bold;'>11</span>",
    "<span style='color:#00CC00;font-weight:bold;'>0.80, 12</span>"
  )
  
  # Create game type mapping for column labels
  game_type_map <- plot_data %>%
    distinct(game_index, game_type_label, game_sequence_label, round_num) %>%
    group_by(game_index) %>%
    summarise(
      round_num = first(round_num),
      expert_first_type = first(game_type_label[game_sequence_label == "Expert First Condition"]),
      novice_first_type = first(game_type_label[game_sequence_label == "Novice First Condition"]),
      .groups = "drop"
    ) %>%
    mutate(
      col_label = paste0("Game ", game_index, ", Round ", round_num, "\nTop: ", expert_first_type, "\nBottom: ", novice_first_type)
    )
  
  game_labels <- setNames(game_type_map$col_label, game_type_map$game_index)
  
  # Create the grid plot
  p_dist <- plot_data %>%
    ggplot(aes(x = rank_factor, y = percentage)) +
    geom_col(aes(fill = game_type_label), width = 0.8, alpha = 0.8) +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), width = 0.4, alpha = 0.7, linewidth = 0.3) +
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
    scale_x_discrete(labels = rank_labels_vec) +
    labs(
      title = "Average Arm-Pull Distribution by Rank Across Games and Rounds",
      subtitle = "Expert condition first: Expert-Novice-Play-Choice; Novice condition first: Novice-Expert-Play-Choice. ±1 SE for Error Bars.",
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
      axis.text.x = ggtext::element_markdown(size = 7, angle = 90, hjust = 1, vjust = 0.5),
      axis.text.y = element_text(size = 6),
      axis.title = element_text(size = 8),
      panel.background = element_rect(fill = "white", colour = "grey90"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.grid.major = element_line(color = "grey95", linewidth = 0.3),
      panel.grid.minor = element_blank()
    )
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_behavior_distribution_average.png")),
      plot = p_dist, width = 20, height = 7, dpi = 300, bg = "white"
    )
    message("Saved average behavior distribution plot to: ",
            file.path(fig_dir, paste0(experiment_name, "_", version, "_behavior_distribution_average.png")))
  }
  
  return(p_dist)
}

