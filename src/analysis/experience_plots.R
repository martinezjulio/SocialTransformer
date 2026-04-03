library(here)
library(tidyverse)
library(janitor)
library(lme4)
library(ggthemes)
library(stringr)
library(ggtext)

#' Create Model Experience Predictions Plot
#'
#' @param model_df Model predictions data frame with columns: session_id, 
#'   game_sequence, game_index, expert_betas, novice_betas
#' @param experiment_name Name of the experiment
#' @param version Version identifier (e.g., "hybrid_model")
#' @param save Logical, whether to save the plot to disk
#' @return A ggplot object
experience_plots <- function(model_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

  # Create the figure directory if saving
  if (save) {
    if (is.null(fig_dir)) fig_dir <- here("src", "analysis", "figures", experiment_name, version)
    if (!dir.exists(fig_dir)) {
      dir.create(fig_dir, recursive = TRUE)
    }
  }
  
  # Function to parse beta string and calculate N_total (sum of all alpha+beta)
  parse_n_total <- function(beta_string) {
    if (is.na(beta_string) || beta_string == "") {
      return(NA_real_)
    }
    
    # Extract all numeric pairs from string like "[(2.05, 2.04), (1.91, 1.90), ...]"
    # Match pattern: (number, number)
    matches <- str_match_all(beta_string, "\\(([0-9.]+),\\s*([0-9.]+)\\)")[[1]]
    
    if (nrow(matches) == 0) {
      return(NA_real_)
    }
    
    # Convert to numeric and sum all alpha+beta pairs
    alphas <- as.numeric(matches[, 2])
    betas <- as.numeric(matches[, 3])
    n_total <- sum(alphas + betas, na.rm = TRUE)
    
    return(n_total)
  }
  
  # Calculate N_total for each row
  model_df_processed <- model_df %>%
    rowwise() %>%
    mutate(
      n_total_expert = parse_n_total(expert_betas),
      n_total_novice = parse_n_total(novice_betas)
    ) %>%
    ungroup()
  
  # Get unique game instances with their N_totals
  # For expert betas (keep rows where expert_betas is not NA)
  expert_data <- model_df_processed %>%
    filter(!is.na(n_total_expert)) %>%
    distinct(session_id, session_index, anonymous_id, game_sequence, game_index, 
             .keep_all = TRUE) %>%
    select(session_id, session_index, anonymous_id, game_sequence, game_index, 
           n_total = n_total_expert) %>%
    mutate(demonstrator_type = "expert")
  
  # For novice betas (keep rows where novice_betas is not NA)
  novice_data <- model_df_processed %>%
    filter(!is.na(n_total_novice)) %>%
    distinct(session_id, session_index, anonymous_id, game_sequence, game_index, 
             .keep_all = TRUE) %>%
    select(session_id, session_index, anonymous_id, game_sequence, game_index, 
           n_total = n_total_novice) %>%
    mutate(demonstrator_type = "novice")
  
  # Combine expert and novice data
  combined_data <- bind_rows(expert_data, novice_data)
  
  # Calculate averages and SE for each game_sequence, game_index, demonstrator_type
  avg_n_total <- combined_data %>%
    group_by(game_sequence, game_index, demonstrator_type) %>%
    summarise(
      mean_n_total = mean(n_total, na.rm = TRUE),
      se_n_total = sd(n_total, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  # Create the plot
  p1 <- avg_n_total %>%
    ggplot(aes(x = game_index, y = mean_n_total, color = demonstrator_type, group = demonstrator_type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_errorbar(aes(
      ymin = mean_n_total - se_n_total,
      ymax = mean_n_total + se_n_total
    ), width = 0.2) +
    # Add jittered individual points
    geom_jitter(
      data = combined_data,
      aes(x = game_index, y = n_total, color = demonstrator_type),
      width = 0.3, height = 0, alpha = 0.3, size = 1.2,
      inherit.aes = FALSE
    ) +
    scale_color_manual(
      values = c(expert = "#1f77b4", novice = "#d62728"),
      labels = c(expert = "Expert", novice = "Novice")
    ) +
    scale_x_continuous(breaks = 0:11) +
    labs(
      title = "Model Experience Predictions (N_total) by Game Index",
      x = "Game Sequence Index",
      y = "N_total (Sum of Alpha + Beta across 12 Arms)",
      color = "Demonstrator Type"
    ) +
    facet_wrap(~ game_sequence, ncol = 2) +
    theme_minimal() +
    theme(
      strip.text = element_text(face = "bold", size = 16),
      plot.title = element_text(face = "bold", hjust = 0.5, size = 20),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 13),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 13),
      legend.position = "bottom",
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA)
    )
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_experience.png")),
      plot = p1, width = 10, height = 8, bg = "white"
    )
  }
  
  # Return the plot object
  return(p1)
}

