# Load Libraries
library(here)
library(dotenv)
library(tidyverse)

# Load environment variables from project root .env
dotenv::load_dot_env(here::here(".env"))
experiment_data_dir <- Sys.getenv("EXPERIMENT_DATA_DIR")

# 1) Set Experiment Name and Version
experiment_name <- "transfer_bandits_choice_textbox"
version         <- "pilot_D"
#experiment_name <- "choice"
#version         <- "full_experiment"

# 2) Read Demographics File from Prolific
demographics <- read_csv(
  file.path(
    experiment_data_dir,
    experiment_name,
    version,
    paste0(experiment_name, "_", version, "_demographics.csv")
  )
)

# 3) Filter Out Invalid Participants
demographics <- demographics |>
  dplyr::filter(
    !(Status == "RETURNED" | Status == "TIMED-OUT" | grepl("^Yes", Colourblindness))
  )

# 3) Read Mongo Logged Data
mongologs <- read_csv(
  file.path(
    experiment_data_dir,
    experiment_name,
    version,
    paste0(experiment_name, "_", version, "_mongologs.csv")
  )
)

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

# 6) IRB Compliance: Anonymize Data
mongologs <- mongologs %>%
  # Create anonymous participant IDs (consistent mapping from prolific_id)
  group_by(prolific_id) %>%
  mutate(anonymous_id = cur_group_id()) %>%
  ungroup() %>%
  # Remove identifiable information (only if columns exist)
  select(-prolific_id, -any_of(c("_id", "timestamp", "created_at", "uid", "experiment_id", "__v")))

# 7) Write Preprocessed Data
write_csv(
  mongologs,
  file.path(
    experiment_data_dir,
    experiment_name,
    version,
    "preprocessed_data.csv"
  )
)
