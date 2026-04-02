library(here)
library(tidyverse)
library(janitor)
library(lme4)
library(ggthemes)
library(stringr)
library(ggtext)

#' Create Reward Overlay Plot
#'
#' @param preprocessed_df Preprocessed data frame with columns: anonymous_id, 
#'   session_id, session_index, game_sequence, game_index, game, round_index, 
#'   reward_generated
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plot to disk
#' @return A ggplot object
reward_plots <- function(preprocessed_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

  # Create the figure directory if saving
  if (save) {
    if (is.null(fig_dir)) fig_dir <- here("figures", experiment_name, version)
    if (!dir.exists(fig_dir)) {
      dir.create(fig_dir, recursive = TRUE)
    }
  }
  
  # Total reward per game
  game_totals <- preprocessed_df %>%
    group_by(across(any_of(c(
      "anonymous_id", "session_id", "session_index",
      "game_sequence", "game_index", "game", "round_index", "rollout_id"
    )))) %>%
    summarise(
      game_total_reward = sum(reward_generated, na.rm = TRUE),
      .groups = "drop"
    )
  
  # --- Tag play games by which demonstrator they follow -------------------------
  game_totals_tagged <- game_totals %>%
    mutate(
      cycle_pos = game_index %% 4,  # position within 4-game chunk
      play_after = case_when(
        game == "play" & game_sequence == "expert-play-novice-play-choice"  & cycle_pos == 1 ~ "after_expert",
        game == "play" & game_sequence == "expert-play-novice-play-choice"  & cycle_pos == 3 ~ "after_novice",
        game == "play" & game_sequence == "novice-play-expert-play-choice"  & cycle_pos == 1 ~ "after_novice",
        game == "play" & game_sequence == "novice-play-expert-play-choice"  & cycle_pos == 3 ~ "after_expert",
        TRUE ~ NA_character_
      )
    )
  
  # Recompute averages/SE so 'play_after' is carried through
  avg_rewards <- game_totals_tagged %>%
    group_by(game_sequence, game_index, game, round_index, play_after) %>%
    summarise(
      mean_reward = mean(game_total_reward, na.rm = TRUE),
      se_reward   = sd(game_total_reward,   na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  
  # --- Base plot (same look) ----------------------------------------------------
  p1 <- avg_rewards %>%
    filter(game != "choice") %>%
    ggplot(aes(x = game_index, y = mean_reward, group = 1)) +
    geom_line(linewidth = 0.5, colour = "grey70") +
    geom_errorbar(aes(
      ymin = mean_reward - se_reward,
      ymax = mean_reward + se_reward
    ), width = 0.2) +
    scale_x_continuous(breaks = 0:11) +
    labs(
      title = "Average Total Reward by Game Index",
      x = "Game Sequence Index",
      y = "Mean Total Reward"
    ) +
    facet_wrap(~ game_sequence, ncol = 2) +
    theme_minimal() +
    theme(
      strip.text       = element_text(face = "bold", size = 16),
      plot.title       = element_text(face = "bold", hjust = 0.5, size = 20),
      axis.title       = element_text(size = 15),
      axis.text        = element_text(size = 13),
      legend.position  = "none",
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA)
    )
  
  # --- Overlay: connect play-after-expert (blue) & play-after-novice (red) ------
  p1 <- p1 +
    # connect the three play indices that follow EXPERT (1,5,9 or 3,7,11 depending on sequence)
    geom_line(
      data = avg_rewards %>% filter(game == "play", play_after == "after_expert"),
      aes(group = play_after),
      linewidth = 1, colour = "#1f77b4"
    ) +
    geom_point(
      data = avg_rewards %>% filter(game == "play", play_after == "after_expert"),
      size = 1, colour = "#1f77b4"
    ) +
    # connect the three play indices that follow NOVICE
    geom_line(
      data = avg_rewards %>% filter(game == "play", play_after == "after_novice"),
      aes(group = play_after),
      linewidth = 1, colour = "#d62728"
    ) +
    geom_point(
      data = avg_rewards %>% filter(game == "play", play_after == "after_novice"),
      size = 1, colour = "#d62728"
    )
  
  # --- Raw data: jittered points for individual games (participants & demos) ----
  p1 <- p1 +
    geom_jitter(
      data = game_totals_tagged %>% filter(game != "choice"),
      aes(x = game_index, y = game_total_reward, shape = game),
      width = 0.3, height = .1, alpha = 0.1, size = 1.4,
      inherit.aes = FALSE
    ) +
    scale_shape_manual(values = c(expert = 15, novice = 17, play = 16))
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_reward.png")),
      plot = p1, width = 10, height = 8, bg = "white"
    )
  }
  
  # Return the plot object
  return(p1)
}