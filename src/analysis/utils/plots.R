library(here)
library(tidyverse)
library(janitor)
library(lme4)
library(ggthemes)
library(stringr)
library(ggtext)

# First, create the figure directory
dotenv::load_dot_env(here::here("../../.env"))
modeling_data_dir <- Sys.getenv("MODELING_DATA_DIR")
fig_dir <- file.path(modeling_data_dir, "analysis", "figures", experiment_name, version)
if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# Set Experiment Name and Version
experiment_name <- "transfer_bandits_choice_textbox"
version         <- "pilot_C"

experiment_data_path <- Sys.getenv("EXPERIMENT_DATA_PATH")
df <- read_csv(experiment_data_path)



# Compute the Total Reward per game
game_totals <- df %>%
  group_by(
    anonymous_id, session_id, session_index,
    game_sequence, game_index, game, round_index
  ) %>%
  summarise(
    game_total_reward = sum(reward_generated, na.rm = TRUE),
    .groups = "drop"
  )

# Tag play games by which demonstrator they follow 
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
avg_rewards2 <- game_totals_tagged %>%
  group_by(game_sequence, game_index, game, round_index, play_after) %>%
  summarise(
    mean_reward = mean(game_total_reward, na.rm = TRUE),
    se_reward   = sd(game_total_reward,   na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Avg & SE by sequence & index
avg_rewards <- game_totals %>%
  group_by(game_sequence, game_index, game, round_index) %>%
  summarise(
    mean_reward = mean(game_total_reward, na.rm = TRUE),
    se_reward   = sd(game_total_reward,   na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Plot
p1 <- avg_rewards %>%
  filter(game != "choice") %>%
  ggplot(aes(x = game_index, y = mean_reward)) +
  geom_line(size = 1) +
  geom_errorbar(aes(
    ymin = mean_reward - se_reward,
    ymax = mean_reward + se_reward
  ), width = 0.2) +
  scale_x_continuous(breaks = 0:11) +
  labs(
    title = "Average Total Reward by Game Index",
    x     = "Game Sequence Index",
    y     = "Mean Total Reward"
  ) +
  facet_wrap(~ game_sequence, ncol = 2) +
  theme_minimal() +
  theme(
    strip.text       = element_text(face = "bold"),
    plot.title       = element_text(face = "bold", hjust = 0.5),
    legend.position  = "none",
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA)
  )

ggsave(
  filename = file.path(fig_dir, paste0(experiment_name, "_", version, "_reward.png")),
  plot = p1, width = 10, height = 8,
  bg = "white"
)