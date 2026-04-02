library(here)
library(tidyverse)
library(janitor)
library(lme4)
library(ggthemes)
library(stringr)
library(ggtext)

#' Create Entropy Plots
#'
#' @param preprocessed_df Preprocessed data frame with columns: anonymous_id, 
#'   session_id, session_index, game_sequence, game_index, game, position, action
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plot to disk
#' @return A ggplot object
entropy_plots <- function(preprocessed_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

  # Create the figure directory if saving
  if (save) {
    if (is.null(fig_dir)) fig_dir <- here("figures", experiment_name, version)
    if (!dir.exists(fig_dir)) {
      dir.create(fig_dir, recursive = TRUE)
    }
  }
  
  # ---- Parameters & helper -----------------------------------------------------
  n_arms <- 12L
  logK   <- log2(n_arms)
  
  # Shannon entropy in bits for a vector of arm indices (0..11)
  H_bits <- function(pos, K = n_arms) {
    tab <- table(factor(pos, levels = 0:(K - 1)))
    p   <- as.numeric(tab) / sum(tab)
    p   <- p[p > 0]
    -sum(p * log(p, base = 2))
  }
  
  # ---- Per-person, per-game entropy -------------------------------------------
  game_entropy <- preprocessed_df %>%
    filter(game != "choice", action == "shoot",
           !is.na(position), position >= 0, position < n_arms) %>%
    group_by(across(any_of(c(
      "anonymous_id", "session_id", "session_index",
      "game_sequence", "game_index", "game", "rollout_id"
    )))) %>%
    summarise(
      n_shoots      = n(),
      entropy_bits  = H_bits(position, n_arms),
      .groups = "drop"
    ) %>%
    mutate(entropy_norm = entropy_bits / logK)   # optional 0..1 scale
  
  # ---- Tag play games by which demo they follow (same logic as before) --------
  game_entropy <- game_entropy %>%
    mutate(
      cycle_pos = game_index %% 4,
      play_after = case_when(
        game == "play" & game_sequence == "expert-play-novice-play-choice" & cycle_pos == 1 ~ "after_expert",
        game == "play" & game_sequence == "expert-play-novice-play-choice" & cycle_pos == 3 ~ "after_novice",
        game == "play" & game_sequence == "novice-play-expert-play-choice" & cycle_pos == 1 ~ "after_novice",
        game == "play" & game_sequence == "novice-play-expert-play-choice" & cycle_pos == 3 ~ "after_expert",
        TRUE ~ NA_character_
      )
    )
  
  # ---- Average & SE across people (per game index) ----------------------------
  avg_entropy <- game_entropy %>%
    group_by(game_sequence, game_index, game, play_after) %>%
    summarise(
      mean_entropy = mean(entropy_bits, na.rm = TRUE),
      se_entropy   = sd(entropy_bits,   na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  # ---- Plot (thin light-grey base; blue/red overlays; jitter raw points) ------
  pH <- avg_entropy %>%
    filter(game != "choice") %>%
    ggplot(aes(x = game_index, y = mean_entropy)) +
    # base mean line & SE bars (lighter/thinner)
    geom_line(linewidth = 0.5, colour = "grey70") +
    geom_errorbar(aes(
      ymin = mean_entropy - se_entropy,
      ymax = mean_entropy + se_entropy
    ), width = 0.15, linewidth = 0.4, colour = "grey75") +
    # overlays: play-after-expert (blue) and play-after-novice (red)
    geom_line(
      data = avg_entropy %>% filter(game == "play", play_after == "after_expert"),
      aes(group = play_after),
      linewidth = 1, colour = "#1f77b4"
    ) +
    geom_point(
      data = avg_entropy %>% filter(game == "play", play_after == "after_expert"),
      size = 2, colour = "#1f77b4"
    ) +
    geom_line(
      data = avg_entropy %>% filter(game == "play", play_after == "after_novice"),
      aes(group = play_after),
      linewidth = 1, colour = "#d62728"
    ) +
    geom_point(
      data = avg_entropy %>% filter(game == "play", play_after == "after_novice"),
      size = 2, colour = "#d62728"
    ) +
    # raw per-person per-game entropies
    geom_jitter(
      data = game_entropy %>% filter(game != "choice"),
      aes(x = game_index, y = entropy_bits, shape = game),
      width = 0.15, height = 0, alpha = 0.1, size = 1.4, inherit.aes = FALSE
    ) +
    scale_shape_manual(values = c(expert = 15, novice = 17, play = 16)) +
    scale_x_continuous(breaks = 0:11) +
    labs(
      title = "Average Choice Entropy by Game Index",
      x     = "Game Sequence Index",
      y     = "Choice Entropy"
    ) +
    facet_wrap(~ game_sequence, ncol = 2) +
    theme_minimal() +
    theme(
      strip.text      = element_text(face = "bold", size = 16),
      plot.title      = element_text(face = "bold", hjust = 0.5, size = 20),
      axis.title      = element_text(size = 15),
      axis.text       = element_text(size = 13),
      legend.position = "none"
    )
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_entropy.png")),
      plot = pH, width = 10, height = 8, bg = "white"
    )
  }
  
  # Return the plot object
  return(pH)
}