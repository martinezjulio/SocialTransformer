# Load Libraries
library(here)
library(dotenv)
library(tidyverse)

# Load environment variables from project root .env
dotenv::load_dot_env(here::here(".env"))
experiment_data_dir <- Sys.getenv("EXPERIMENT_DATA_DIR")

# 1) Set Experiment Name and Version (override via env vars or edit defaults).
experiment_name <- Sys.getenv("EXPERIMENT_NAME", unset = "transfer_bandits_choice_textbox")
version         <- Sys.getenv("VERSION",         unset = "pilot_D")

demographics_path  <- file.path(experiment_data_dir, experiment_name, version,
                                paste0(experiment_name, "_", version, "_demographics.csv"))
mongologs_path     <- file.path(experiment_data_dir, experiment_name, version,
                                paste0(experiment_name, "_", version, "_mongologs.csv"))
preprocessed_path  <- file.path(experiment_data_dir, experiment_name, version,
                                "preprocessed_data.csv")

cat(sprintf("Experiment: %s / %s\n", experiment_name, version))
cat(sprintf("  demographics: %s\n", demographics_path))
cat(sprintf("  mongologs:    %s\n", mongologs_path))
cat(sprintf("  output:       %s\n", preprocessed_path))

# 2) Read Demographics File from Prolific
demographics <- read_csv(demographics_path)

# 3) Filter Out Invalid Participants
demographics <- demographics |>
  dplyr::filter(
    !(Status == "RETURNED" | Status == "TIMED-OUT" | grepl("^Yes", Colourblindness))
  )

# 3) Read Mongo Logged Data
mongologs <- read_csv(mongologs_path)

# 4) Filter Mongologs Data
mongologs <- mongologs %>%
  filter(prolific_id %in% demographics$`Participant id`) |>
  filter(!str_detect(game, "practice")) 

# 5) Add Within-Participant Index
mongologs <- mongologs %>%
  group_by(prolific_id) %>%
  mutate(.within = row_number()) %>%   # preserves current per-user order
  ungroup() %>%
  arrange(prolific_id, .within) %>%
  select(-.within)

# 5b) Quality filter — drop sessions that can't be used downstream.
# Mirrors src/run_inference.py so preprocessed_data.csv is self-consistent
# with what inference and analysis expect.
shot_rows <- mongologs %>%
  filter(shot == TRUE, game %in% c("watch", "play"))

sessions_no_gameplay <- setdiff(unique(mongologs$session_id),
                                unique(shot_rows$session_id))

expected_rounds <- c(0, 1, 2)
expected_games_per_round <- 4
session_structure <- shot_rows %>%
  group_by(session_id) %>%
  summarise(
    rounds_ok = setequal(unique(round_index), expected_rounds),
    n_games   = n_distinct(game_index),
    .groups   = "drop"
  )
sessions_incomplete <- session_structure %>%
  filter(!rounds_ok |
         n_games != expected_games_per_round * length(expected_rounds)) %>%
  pull(session_id)

expected_shots_per_play <- 6
sessions_short_play <- shot_rows %>%
  filter(game == "play") %>%
  count(session_id, game_index) %>%
  filter(n != expected_shots_per_play) %>%
  pull(session_id) %>%
  unique()

bad_sessions <- unique(c(sessions_no_gameplay,
                         sessions_incomplete,
                         sessions_short_play))
if (length(bad_sessions) > 0) {
  cat(sprintf("Dropping %d session(s) for quality issues:\n", length(bad_sessions)))
  for (sid in bad_sessions) {
    reasons <- c(
      if (sid %in% sessions_no_gameplay) "no-gameplay",
      if (sid %in% sessions_incomplete)  "incomplete-rounds",
      if (sid %in% sessions_short_play)  "short-play"
    )
    cat(sprintf("  %s  [%s]\n", sid, paste(reasons, collapse = ",")))
  }
  mongologs <- mongologs %>% filter(!session_id %in% bad_sessions)
} else {
  cat("No sessions dropped by quality filter.\n")
}

# 6) IRB Compliance: Anonymize Data
mongologs <- mongologs %>%
  # Create anonymous participant IDs (consistent mapping from prolific_id)
  group_by(prolific_id) %>%
  mutate(anonymous_id = cur_group_id()) %>%
  ungroup() %>%
  # Remove identifiable information (only if columns exist)
  select(-prolific_id, -any_of(c("_id", "timestamp", "created_at", "uid", "experiment_id", "__v")))

# 7) Write Preprocessed Data
write_csv(mongologs, preprocessed_path)
cat(sprintf("Wrote %d rows, %d sessions to %s\n",
            nrow(mongologs),
            length(unique(mongologs$session_id)),
            preprocessed_path))
