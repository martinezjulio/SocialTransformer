library(here)
library(tidyverse)
library(janitor)
library(lme4)
library(ggthemes)
library(stringr)
library(ggtext)

#' Create Choice Frequency Plots
#'
#' @param preprocessed_df Preprocessed data frame with columns: game, action, 
#'   position, game_sequence
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plot to disk
#' @return A ggplot object
choice_frequencies <- function(preprocessed_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

  # Create the figure directory if saving
  if (save) {
    if (is.null(fig_dir)) fig_dir <- here("src", "analysis", "figures", experiment_name, version)
    if (!dir.exists(fig_dir)) {
      dir.create(fig_dir, recursive = TRUE)
    }
  }
  
  choice_shoots <- preprocessed_df %>%
    filter(game == "choice", action == "shoot")
  
  
  # Calculate proportions and confidence intervals
  choice_proportions <- choice_shoots %>%
    group_by(game_sequence) %>%
    summarise(
      total_choices = n(),
      demo1_choices = sum(position == 0),
      demo2_choices = sum(position == 1),
      demo1_prop = demo1_choices / total_choices,
      demo2_prop = demo2_choices / total_choices,
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(
      # Calculate 95% confidence intervals using Wilson score interval
      demo1_ci_lower = if(demo1_choices == 0) 0 else binom.test(demo1_choices, total_choices)$conf.int[1],
      demo1_ci_upper = if(demo1_choices == 0) 0 else binom.test(demo1_choices, total_choices)$conf.int[2],
      demo2_ci_lower = if(demo2_choices == 0) 0 else binom.test(demo2_choices, total_choices)$conf.int[1],
      demo2_ci_upper = if(demo2_choices == 0) 0 else binom.test(demo2_choices, total_choices)$conf.int[2]
    ) %>%
    ungroup() %>%
    # Reshape for plotting
    pivot_longer(
      cols = c(demo1_prop, demo2_prop),
      names_to = "demonstrator",
      values_to = "proportion"
    ) %>%
    mutate(
      demonstrator = factor(demonstrator,
                           levels = c("demo1_prop", "demo2_prop"),
                           labels = c("Demonstrator 1", "Demonstrator 2")),
      ci_lower = ifelse(demonstrator == "Demonstrator 1", demo1_ci_lower, demo2_ci_lower),
      ci_upper = ifelse(demonstrator == "Demonstrator 1", demo1_ci_upper, demo2_ci_upper)
    )
  
  
  # Map each bar to the expertise of the demonstrator shown in that sequence
  choice_proportions_col <- choice_proportions %>%
    mutate(
      first_is_expert = str_starts(game_sequence, regex("^expert", ignore_case = TRUE)),
      expertise = case_when(
        demonstrator == "Demonstrator 1" &  first_is_expert ~ "Expert",
        demonstrator == "Demonstrator 1" & !first_is_expert ~ "Novice",
        demonstrator == "Demonstrator 2" &  first_is_expert ~ "Novice",
        demonstrator == "Demonstrator 2" & !first_is_expert ~ "Expert",
        TRUE ~ NA_character_
      ),
      expertise = factor(expertise, levels = c("Expert", "Novice"))
    )
  
  pd <- position_dodge(width = 0.9)
  
  p2 <- choice_proportions_col %>%
    ggplot(aes(x = game_sequence,
               y = proportion,
               fill = expertise,
               group = demonstrator)) +
    geom_col(position = pd, width = 0.75, alpha = 0.9) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                  position = pd, width = 0.25) +
    scale_fill_manual(
      values = c(Expert = "#1f77b4", Novice = "#d62728"),
      name   = "Chosen demonstrator"
    ) +
    scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Choice Demonstrator Proportions by Game Sequence",
      x     = "Game Sequence",
      y     = "Proportion of Choices"
    ) +
    theme_minimal() +
    theme(
      axis.text.x     = element_text(angle = 30, hjust = 1),
      legend.position = "bottom",
      legend.title    = element_text(face = "bold"),
      plot.title      = element_text(face = "bold", hjust = 0.5)
    )
  
  # Save plot if requested
  if (save) {
    ggsave(
      filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_choice_frequency_by_expertise.png")),
      plot = p2, width = 8, height = 5, bg = "white"
    )
  }
  
  # Return the plot object
  return(p2)
}