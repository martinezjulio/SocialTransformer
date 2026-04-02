library(here)
library(tidyverse)
library(janitor)
library(lme4)
library(ggthemes)
library(stringr)
library(ggtext)

#' Create Correlation Plots Between Demonstrators and Play Games
#'
#' @param preprocessed_df Preprocessed data frame with columns: anonymous_id, 
#'   session_id, game_sequence, game_index, game, position, action, demonstrator_type
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plot to disk
#' @return A ggplot object
correlations <- function(preprocessed_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

  # Create the figure directory if saving
  if (save) {
    if (is.null(fig_dir)) fig_dir <- here("figures", experiment_name, version)
    if (!dir.exists(fig_dir)) {
      dir.create(fig_dir, recursive = TRUE)
    }
  }
  
  n_arms <- 12L
  
  # Helper function to compute arm frequency vector for a game
  compute_arm_frequencies <- function(positions) {
    freq <- table(factor(positions, levels = 0:(n_arms - 1)))
    as.numeric(freq)
  }
  
  # Filter to relevant games and compute arm frequencies per game
  game_arm_freqs <- preprocessed_df %>%
    filter(game != "choice", action == "shoot",
           !is.na(position), position >= 0, position < n_arms) %>%
    # Determine which demonstrator this is (for watch games)
    mutate(
      round = floor(game_index / 4),  # 0, 1, 2
      position_in_round = game_index %% 4,  # 0, 1, 2, 3
      demo_or_play = case_when(
        position_in_round == 0 ~ "demo1",
        position_in_round == 1 ~ "play1", 
        position_in_round == 2 ~ "demo2",
        position_in_round == 3 ~ "play2",
        TRUE ~ NA_character_
      )
    ) %>%
    group_by(across(any_of(c(
      "anonymous_id", "game_sequence", "round", "demo_or_play",
      "game_index", "game", "rollout_id"
    )))) %>%
    summarise(
      arm_frequencies = list(compute_arm_frequencies(position)),
      .groups = "drop"
    )
  
  # Separate demonstrators and plays
  demos <- game_arm_freqs %>%
    filter(game %in% c("watch", "expert", "novice")) %>%
    select(anonymous_id, game_sequence, round, demo_or_play, arm_frequencies) %>%
    rename(demo_type = demo_or_play, demo_freqs = arm_frequencies)
  
  plays <- game_arm_freqs %>%
    filter(game == "play") %>%
    select(anonymous_id, game_sequence, round, demo_or_play, arm_frequencies) %>%
    rename(play_type = demo_or_play, play_freqs = arm_frequencies)
  
  # Compute all correlations within each round
  # We want: demo1 vs play1, demo1 vs play2, demo2 vs play1, demo2 vs play2
  correlations_df <- plays %>%
    # Cross join with demos from same participant, sequence, and round
    inner_join(demos, by = c("anonymous_id", "game_sequence", "round")) %>%
    rowwise() %>%
    mutate(
      correlation = cor(unlist(demo_freqs), unlist(play_freqs), method = "pearson")
    ) %>%
    ungroup() %>%
    select(anonymous_id, game_sequence, round, demo_type, play_type, correlation)
  
  # Average correlations across participants
  avg_correlations <- correlations_df %>%
    group_by(game_sequence, round, demo_type, play_type) %>%
    summarise(
      mean_corr = mean(correlation, na.rm = TRUE),
      se_corr = sd(correlation, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      round_label = paste("Round", round + 1),  # Convert 0,1,2 to Round 1, 2, 3
      comparison = paste0(demo_type, " → ", play_type),
      # Create cleaner game sequence labels
      game_sequence_label = case_when(
        str_starts(game_sequence, "expert") ~ "Expert first",
        str_starts(game_sequence, "novice") ~ "Novice first",
        TRUE ~ game_sequence
      )
    )
  
  # Create labels that are more intuitive
  avg_correlations <- avg_correlations %>%
    mutate(
      demo_label = case_when(
        demo_type == "demo1" & str_starts(game_sequence, "expert") ~ "Expert (1st)",
        demo_type == "demo1" & str_starts(game_sequence, "novice") ~ "Novice (1st)",
        demo_type == "demo2" & str_starts(game_sequence, "expert") ~ "Novice (2nd)",
        demo_type == "demo2" & str_starts(game_sequence, "novice") ~ "Expert (2nd)",
        TRUE ~ demo_type
      ),
      play_label = case_when(
        play_type == "play1" ~ "Play 1",
        play_type == "play2" ~ "Play 2",
        TRUE ~ play_type
      ),
      demo_expertise = case_when(
        str_detect(demo_label, "Expert") ~ "Expert",
        str_detect(demo_label, "Novice") ~ "Novice",
        TRUE ~ "Other"
      ),
      # Create combined x-axis position for connecting across rounds
      x_position = round * 2 + ifelse(play_type == "play1", 1, 2),
      x_label = paste(play_label, "\n", round_label)
    )
  
  # Create the plot with lines connected across rounds
  p_corr <- avg_correlations %>%
    ggplot(aes(x = x_position, y = mean_corr, color = demo_expertise, 
               group = demo_type)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(
      aes(ymin = mean_corr - se_corr, ymax = mean_corr + se_corr),
      width = 0.25, linewidth = 0.8
    ) +
    scale_color_manual(
      values = c(Expert = "#1f77b4", Novice = "#d62728"),
      name = "Demonstrator"
    ) +
    scale_x_continuous(
      breaks = c(1, 2, 3, 4, 5, 6),
      labels = c("Play 1\nRound 1", "Play 2\nRound 1", 
                 "Play 1\nRound 2", "Play 2\nRound 2",
                 "Play 1\nRound 3", "Play 2\nRound 3")
    ) +
    facet_wrap(~ game_sequence_label, ncol = 1) +
    labs(
      title = "Arm-Pulling Frequency Correlations:\nDemonstrators vs. Participant Play Games",
      x = "Play Game",
      y = "Pearson Correlation",
      subtitle = "Lines show correlation between demonstrator and play game arm frequencies across rounds"
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(face = "bold", size = 16),
      plot.title = element_text(face = "bold", hjust = 0.5, size = 20),
      plot.subtitle = element_text(hjust = 0.5, size = 16, color = "gray40"),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 13),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 13),
      legend.position = "bottom",
      panel.grid.major.x = element_blank(),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA)
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", alpha = 0.5)
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_arm_correlations.png")),
      plot = p_corr, width = 12, height = 8, bg = "white"
    )
  }
  
  # Return the plot object
  return(p_corr)
}

