# Clean one experiment version's anonymised data into preprocessed_participant_data.csv.
# Input: the *_anonymized.csv pair written by src/anonymize_participant_data.py (uncommitted);
# this script never sees Prolific IDs. Every exclusion rule lives here.

# Load Libraries
library(here)
library(tidyverse)   # dotenv is not attached: attaching it loads .env and overrides the caller's variables

# Load environment variables from project root .env unless the caller already set them
if (Sys.getenv("EXPERIMENT_DATA_DIR") == "") dotenv::load_dot_env(here::here(".env"))
experiment_data_dir <- Sys.getenv("EXPERIMENT_DATA_DIR")

# 1) Experiment name and version. VERSION has no default: a silent fallback could
#    overwrite another pilot's preprocessed file, so the caller must set it, e.g.
#    VERSION=pilot_E Rscript src/r/preprocess_participant_data.R
version         <- Sys.getenv("VERSION")
if (version == "") stop("VERSION is not set (e.g. VERSION=pilot_E Rscript src/r/preprocess_participant_data.R)")
file_prefix     <- Sys.getenv("FILE_PREFIX",     unset = "socialbandits_transfer")

demographics_path  <- file.path(experiment_data_dir, version,
                                paste0(file_prefix, "_", version, "_demographics_anonymized.csv"))
mongologs_path     <- file.path(experiment_data_dir, version,
                                paste0(file_prefix, "_", version, "_trials_anonymized.csv"))
preprocessed_path  <- file.path(experiment_data_dir, version,
                                "preprocessed_participant_data.csv")

cat(sprintf("Version: %s\n", version))
cat(sprintf("  demographics: %s\n", demographics_path))
cat(sprintf("  mongologs:    %s\n", mongologs_path))
cat(sprintf("  output:       %s\n", preprocessed_path))

# 2) Read the anonymised demographics (Prolific export minus identifiers)
demographics <- read_csv(demographics_path)

# 3) Filter Out Invalid Participants. `excl` is the ledger of counts per rule, written as exclusions.csv.
excl <- tibble(level = character(), rule = character(), n = integer())
note <- function(level, rule, ids) excl <<- add_row(excl, level = level, rule = rule, n = length(unique(ids)))
# X1: returned on Prolific only. A Prolific time-out is not a rule: a timed-out finisher is kept if the
# logged rules below pass (the span ceiling and the three-session rule catch the pilot E case).
note("participant", "returned", demographics$anonymous_id[demographics$Status == "RETURNED"])
note("participant", "colour_vision", demographics$anonymous_id[grepl("^Yes", demographics$Colourblindness)])
demographics <- demographics |>
  dplyr::filter(!(Status == "RETURNED" | grepl("^Yes", Colourblindness)))

# 3) Read the anonymised trial logs
mongologs <- read_csv(mongologs_path,
                      col_types = cols(timestamp = col_character(), .default = col_guess()))  # keep ms

# 3b) Speed floor (participant-level, whole study). The experimenter's fastest possible run through
# the whole study, instructions and practice included, took 4 min 30 s; a participant under 5 min
# was not attending to the task. Measured on the logged span (first to last event, practice
# included), which is shorter than the stopwatch time. Pilot E minimum: 416 s (nobody excluded).
# Mirror ceiling: a logged span over 60 min means the study was left and resumed (slowest ordinary
# pilot E completer: 14 min of logged play).
SPEED_FLOOR_S <- 300
SPAN_CEILING_S <- 3600
study_span <- mongologs %>%
  group_by(anonymous_id) %>%
  summarise(span_s = as.numeric(difftime(max(ymd_hms(timestamp)), min(ymd_hms(timestamp)), units = "secs")),
            .groups = "drop")
too_fast <- study_span %>% filter(span_s < SPEED_FLOOR_S) %>% pull(anonymous_id)
cat(sprintf("Speed floor %d s: dropping %d participant(s); fastest logged span %.0f s\n",
            SPEED_FLOOR_S, length(too_fast), min(study_span$span_s)))
too_slow <- study_span %>% filter(span_s > SPAN_CEILING_S) %>% pull(anonymous_id)
cat(sprintf("Span ceiling %d s: dropping %d participant(s); longest logged span %.0f s\n",
            SPAN_CEILING_S, length(too_slow), max(study_span$span_s)))
note("participant", "speed_floor", too_fast); note("participant", "span_ceiling", too_slow)
mongologs <- mongologs %>% filter(!anonymous_id %in% c(too_fast, too_slow))

# 4) Filter Mongologs Data
mongologs <- mongologs %>%
  filter(anonymous_id %in% demographics$anonymous_id) |>
  filter(!str_detect(game, "practice")) 

# 5) Add Within-Participant Index
mongologs <- mongologs %>%
  group_by(anonymous_id) %>%
  mutate(.within = row_number()) %>%   # preserves current per-user order
  ungroup() %>%
  arrange(anonymous_id, .within) %>%
  select(-.within)

# 5b) Quality filter — drop sessions that can't be used downstream.
# Mirrors src/run_inference.py so preprocessed_participant_data.csv is self-consistent
# with what inference and analysis expect.
shot_rows <- mongologs %>%
  filter(shot == TRUE, game %in% c("watch", "play"))

sessions_no_gameplay <- setdiff(unique(mongologs$session_id),
                                unique(shot_rows$session_id))

# pilot_E ran 2 rounds per session; every other version ran 3.
expected_rounds <- if (version == "pilot_E") c(0, 1) else c(0, 1, 2)
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

note("session", "no_gameplay", sessions_no_gameplay); note("session", "incomplete_rounds", sessions_incomplete)
note("session", "short_play", sessions_short_play)
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

# 5d) Reload rule (session-level). refresh_count is stamped on every row at log time and, when the
# participant leaves the end screen, bumped retroactively on every row of the final expedition, so a
# constant 1 on one session is normal completion. A value above 1, or a value that changes within a
# session, means the page was reloaded while that session was being logged.
sessions_reloaded <- mongologs %>%
  group_by(session_id) %>%
  summarise(n_values = n_distinct(refresh_count), max_refresh = max(refresh_count), .groups = "drop") %>%
  filter(n_values > 1 | max_refresh > 1) %>%
  pull(session_id)
cat(sprintf("Reload rule: dropping %d session(s)\n", length(sessions_reloaded)))
note("session", "reload", sessions_reloaded)
mongologs <- mongologs %>% filter(!session_id %in% sessions_reloaded)

# 5e) Participant-level rules, after the session-level ones.
#  - restart: more than one page load (page_load_index > 0); a reload restarts the study.
#  - inattention: a session in which every play shot stayed on the game's starting position.
#  - completeness: exactly three main sessions survive (covers dropouts, missing sessions and
#    repeated attempts); the design is within-participant and the stopping rule counts whole participants.
EXPECTED_SESSIONS <- 3
restarted <- mongologs %>% filter(page_load_index > 0) %>% distinct(anonymous_id) %>% pull()
never_moved <- mongologs %>%
  filter(game == "play") %>%
  arrange(timestamp, step) %>%
  group_by(session_id, game_index) %>%
  summarise(anonymous_id = first(anonymous_id),
            all_on_start = all(position[shot == TRUE] == first(position)), .groups = "drop") %>%
  group_by(session_id) %>%
  summarise(anonymous_id = first(anonymous_id), session_on_start = all(all_on_start), .groups = "drop") %>%
  filter(session_on_start) %>% distinct(anonymous_id) %>% pull()
incomplete <- mongologs %>% distinct(anonymous_id, session_id) %>% count(anonymous_id) %>%
  filter(n != EXPECTED_SESSIONS) %>% pull(anonymous_id)
cat(sprintf("Participant rules: restarted %d, never moved %d, not exactly %d sessions %d\n",
            length(restarted), length(never_moved), EXPECTED_SESSIONS, length(incomplete)))
note("participant", "restarted", restarted); note("participant", "never_moved", never_moved)
note("participant", "not_three_sessions", incomplete)
mongologs <- mongologs %>% filter(!anonymous_id %in% c(restarted, never_moved, incomplete))

# 5c) pilot_E labels its demonstrators demo-1 / demo-2 and leaves demonstrator_type and
# choice_demonstrator_type empty. The identity is still recorded: demonstrator_index (0 = demo-1,
# 1 = demo-2) on every watch row, and demonstrator_experience (128 shots = expert, 0 = novice)
# on one row per watch game; choice_demonstrator_index uses the same 0/1. Derive the pilot_D
# labels so every downstream reader works unchanged. Solo-play sessions keep their own
# game_sequence ("play-play-play-play") and are skipped by run_inference.py.
if (version == "pilot_E") {
  idx_to_type <- c(`0` = "expert", `1` = "novice")
  exp_check <- mongologs |>
    filter(game == "watch", !is.na(demonstrator_experience)) |>
    distinct(demonstrator_index, demonstrator_experience)
  stopifnot(nrow(exp_check) == 2,
            exp_check$demonstrator_experience[exp_check$demonstrator_index == 0] == 128,
            exp_check$demonstrator_experience[exp_check$demonstrator_index == 1] == 0)
  mongologs <- mongologs |>
    mutate(
      demonstrator_type = if_else(game == "watch" & !is.na(demonstrator_index),
                                  unname(idx_to_type[as.character(demonstrator_index)]),
                                  as.character(demonstrator_type)),
      choice_demonstrator_type = if_else(game == "choice" & !is.na(choice_demonstrator_index),
                                         unname(idx_to_type[as.character(choice_demonstrator_index)]),
                                         as.character(choice_demonstrator_type)),
      game_sequence = recode(game_sequence,
        "watch-demo-1-play-watch-demo-2-play-choice" = "expert-play-novice-play-choice",
        "watch-demo-2-play-watch-demo-1-play-choice" = "novice-play-expert-play-choice"))
  stopifnot(all(!is.na(mongologs$demonstrator_type[mongologs$game == "watch"])))
  # pilot_E's game_index skips one slot per round (0-3, 5-8, choice 10-11). Downstream code
  # takes the position within a round from game_index %% 4 and expects pilot_D's numbering
  # (4 * round + position, choice games after the last round), so renumber densely per session.
  mongologs <- mongologs |>
    group_by(session_id) |>
    mutate(game_index = dense_rank(game_index) - 1L) |>
    ungroup()
  stopifnot(all(mongologs$game_index[mongologs$game != "choice"] ==
                4L * mongologs$round_index[mongologs$game != "choice"] +
                  (mongologs$game_index[mongologs$game != "choice"] %% 4L)))
  cat("pilot_E: derived demonstrator_type / choice_demonstrator_type, renamed game_sequence, renumbered game_index\n")
  print(mongologs |> distinct(session_id, game_sequence) |> count(game_sequence))
}

# 6) Identifiers were removed upstream by src/anonymize_participant_data.py; timestamp is kept.
# Value checks: the experiment's action enum and the three main-session game types.
stopifnot("timestamp" %in% names(mongologs),
          all(na.omit(mongologs$action) %in% c("left", "right", "up", "down", "shoot", "feedback", "trajectory_complete")),
          all(mongologs$game %in% c("watch", "play", "choice")))

# 7) Write Preprocessed Data, in time order (the export's row order is arbitrary): participant,
# session start, event time, step (tie-break for key repeats within the same millisecond).
mongologs <- mongologs %>%
  group_by(session_id) %>% mutate(.session_start = min(timestamp)) %>% ungroup() %>%
  arrange(anonymous_id, .session_start, timestamp, step) %>%
  select(-.session_start)
write_csv(mongologs, preprocessed_path)
excl <- add_row(excl, level = "participant", rule = "kept", n = n_distinct(mongologs$anonymous_id)) |>
  add_row(level = "session", rule = "kept", n = n_distinct(mongologs$session_id))
write_csv(excl, file.path(dirname(preprocessed_path), "exclusions.csv"))
print(excl, n = Inf)
cat(sprintf("Wrote %d rows, %d sessions to %s\n",
            nrow(mongologs),
            length(unique(mongologs$session_id)),
            preprocessed_path))
