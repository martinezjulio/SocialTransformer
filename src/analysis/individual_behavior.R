library(here)
library(tidyverse)
library(ggplot2)
library(ggtext)

#' Create Individual Participant Behavior Plots
#'
#' Creates detailed trajectory plots for each participant showing their arm-pulling
#' behavior alongside demonstrators across all games.
#'
#' @param preprocessed_df Preprocessed data frame with columns: anonymous_id, 
#'   session_id, game_sequence, game_index, game, position, action, reward_generated
#' @param experiment_name Name of the experiment
#' @param version Version identifier
#' @param save Logical, whether to save the plots to disk
#' @return Invisible list of generated file paths
individual_behavior <- function(preprocessed_df, experiment_name, version, save = FALSE, fig_dir = NULL) {

  # Create the figure directory if saving
  if (save) {
    if (is.null(fig_dir)) fig_dir <- here("src", "analysis", "figures", experiment_name, version)
    participant_fig_dir <- file.path(fig_dir, "individual_participants")
    if (!dir.exists(participant_fig_dir)) {
      dir.create(participant_fig_dir, recursive = TRUE)
    }
  } else {
    message("Note: save=FALSE, no files will be generated. Set save=TRUE to create individual plots.")
    return(invisible(NULL))
  }
  
  # =========================
  # Build participant_watch_play with game_step
  # =========================
  participant_watch_play <- preprocessed_df %>%
    filter(game != "choice", action == "shoot") %>%
    arrange(anonymous_id, session_id, game_sequence, round_index, game_index) %>%
    arrange(anonymous_id, session_id, session_index, game_sequence, round_index, game_index, step) %>%
    group_by(anonymous_id, session_id, game_sequence, round_index, game_index, game) %>%
    mutate(game_step = row_number() - 1) %>%   # underlying 0..5 (we'll display 1..6)
    ungroup()
  
  # =========================
  # Helper functions
  # =========================
  
  # Build 6 column labels from demonstrator_type at even indices (0,2,4,6,8,10)
  col_labels_from_demonstrator <- function(df, show_indices = TRUE) {
    sapply(0:5, function(j) {
      eidx <- 2 * j
      # Prefer rows marked as watch for that even index, fall back to any rows at that index
      types_watch <- df %>%
        filter(game_index == eidx, tolower(game) == "watch") %>%
        pull(demonstrator_type)
      types_any <- df %>%
        filter(game_index == eidx) %>%
        pull(demonstrator_type)
      typ <- unique(na.omit(tolower(c(types_watch, types_any))))
      label_type <- if (length(typ)) typ[1] else "unknown"
      if (show_indices) {
        paste0(eidx, " | ", eidx + 1, ", ", label_type)
      } else {
        label_type
      }
    })
  }
  
  # Robust TRUE/FALSE from reward_generated
  as_reward_flag <- function(x) {
    if (is.logical(x)) return(x)
    tolower(as.character(x)) %in% c("true","t","1","yes","y")
  }
  
  # Strip "sid"/"_sid_" prefixes from session_id for display/filenames
  session_label_from_id <- function(sid) {
    s <- as.character(sid)[1]
    s <- sub("^sid[_-]?", "", s)         # leading "sid_" or "sid-"
    s <- sub(".*_sid[_-]?", "", s)       # anything up to "_sid_"
    s
  }
  
  # Parse a zone_bandit string like "0.6,0,0.8,..."
  .parse_zone_bandit_once <- function(s) {
    if (is.numeric(s) && length(s) > 1) return(as.numeric(s))
    s <- as.character(s[1])
    nums <- regmatches(
      s,
      gregexpr("-?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?", s, perl = TRUE)
    )[[1]]
    as.numeric(nums)
  }
  
  # Map position (0-11) to payoff rank (1=worst, 12=best) based on zone_bandit
  # This ensures each game's positions are mapped to ranks based on that game's specific bandit configuration
  position_to_rank <- function(position, zone_bandit_str) {
    # Handle missing values
    if (is.na(position) || is.na(zone_bandit_str) || zone_bandit_str == "") return(NA_real_)
    
    # Parse the payoff values for this specific game's bandit
    payoffs <- .parse_zone_bandit_once(zone_bandit_str)
    
    # Verify we have exactly 12 payoffs (one per arm)
    if (length(payoffs) != 12) {
      warning(sprintf("Expected 12 payoffs but got %d from: %s", 
                      length(payoffs), substr(zone_bandit_str, 1, 50)))
      return(NA_real_)
    }
    
    # Verify position is valid (0-11)
    if (position < 0 || position > 11) {
      warning(sprintf("Invalid position %d (must be 0-11)", position))
      return(NA_real_)
    }
    
    # Rank the arms by payoff: rank 1 = lowest payoff (worst), rank 12 = highest payoff (best)
    # rank() with positive values sorts ascending (lowest payoff gets rank 1)
    ranks <- rank(payoffs, ties.method = "average")
    
    # Return the rank for this position (position is 0-indexed, so add 1)
    ranks[position + 1]
  }
  
  # Build y-axis label function: ranks 1-12 (1=worst, 12=best)
  make_y_labeler <- function() {
    function(v) {
      vi <- as.integer(round(v))
      sapply(vi, function(idx) {
        if (is.na(idx) || idx == 0) return("")  # blank tick at 0
        if (idx == 1) {
          return("<span style='color:#D32F2F;'>1 (Worst)</span>")
        } else if (idx == 12) {
          return("<span style='color:#2E7D32; font-weight:700;'>12 (Best)</span>")
        }
        as.character(idx)
      })
    }
  }
  
  # =========================
  # Diagnostics
  # =========================
  
  # Check how many unique bandits we have
  unique_bandits <- participant_watch_play %>%
    distinct(game_index, zone_bandit) %>%
    nrow()
  message(sprintf("Found %d unique game_index/zone_bandit combinations", unique_bandits))
  
  # Verify the position-to-rank mapping with a sample
  if (nrow(participant_watch_play) > 0) {
    sample_row <- participant_watch_play[1, ]
    if (!is.na(sample_row$zone_bandit) && !is.na(sample_row$position)) {
      payoffs <- .parse_zone_bandit_once(sample_row$zone_bandit)
      if (length(payoffs) == 12) {
        ranks <- rank(payoffs, ties.method = "average")
        message(sprintf("\nSample verification (first row):"))
        message(sprintf("  Payoffs: %s", paste(round(payoffs, 2), collapse=", ")))
        message(sprintf("  Ranks:   %s (1=worst, 12=best)", paste(ranks, collapse=", ")))
        message(sprintf("  Position %d maps to rank %.1f", sample_row$position, ranks[sample_row$position + 1]))
      }
    }
  }
  
  # =========================
  # Plotting function for one (anonymous_id, session_id)
  # =========================
  make_panel <- function(df, participant_index) {
    sid         <- dplyr::first(df$session_id)
    session_lab <- session_label_from_id(sid)
    gseq        <- dplyr::first(df$game_sequence)
    
    # USE demonstrator_type per even index for the 6 column labels:
    col_labels <- col_labels_from_demonstrator(df, show_indices = TRUE)
    y_lab      <- make_y_labeler()
    
    df <- df %>%
      filter(game_index >= 0, game_index <= 11) %>%
      mutate(
        facet_row = if_else(game_index %% 2L == 0L, "Demonstrator", "Participant"),
        facet_row = factor(facet_row, levels = c("Demonstrator","Participant")),
        facet_col_pair = floor(game_index / 2),  # 0..5
        facet_col = factor(facet_col_pair, levels = 0:5, labels = col_labels),
        reward_flag = as_reward_flag(reward_generated),
        
        # DISPLAY variables:
        step_disp = game_step + 1,   # show 1..6 on x (with a blank 0 tick)
        # Map position to payoff rank for y-axis
        pos_rank = purrr::map2_dbl(position, zone_bandit, position_to_rank),
        pos_disp = pos_rank  # display rank: 1 = best, 12 = worst
      )
    
    ggplot(df, aes(x = step_disp, y = pos_disp)) +
      # Lines: even row (black), odd row (blue)
      geom_line(
        data = dplyr::filter(df, facet_row == "Demonstrator"),
        aes(group = interaction(game_sequence, round_index, game_index)),
        linewidth = 0.25, color = "black", alpha = 0.6
      ) +
      geom_line(
        data = dplyr::filter(df, facet_row == "Participant"),
        aes(group = interaction(game_sequence, round_index, game_index)),
        linewidth = 0.25, color = "blue", alpha = 0.6
      ) +
      # Points: green if reward TRUE, else red
      geom_point(aes(color = reward_flag), size = 1) +
      facet_grid(facet_row ~ facet_col, drop = FALSE) +
      # X: ticks 0..6, label 0 as blank, 1..6 as steps
      scale_x_continuous(
        breaks = 0:6, limits = c(0, 6),
        labels = c("", as.character(1:6)),
        expand = expansion(mult = c(0, 0.02))
      ) +
      # Y: ticks 0..12, label 0 as blank; 1..12 as "i, payoff"
      scale_y_continuous(
        breaks = 0:12, limits = c(0, 12),
        labels = y_lab,
        expand = expansion(mult = c(0, 0.02))
      ) +
      scale_color_manual(values = c(`TRUE` = "green", `FALSE` = "red"), guide = "none") +
      labs(
        title    = paste0("Participant: ", participant_index, " • Session ", session_lab),
        subtitle = paste0("Game sequence: ", gseq, " | Y-axis: Arm rank by payoff value for EACH game (1=worst, 12=best)"),
        x = "Step",
        y = "Arm Rank (by Payoff)"
      ) +
      coord_cartesian(clip = "off") +
      theme_classic(base_size = 11) +
      theme(
        plot.background  = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        strip.background = element_rect(fill = "white", color = "grey70"),
        # --- turn ON grid lines ---
        panel.grid.major.x = element_line(color = "grey85", linewidth = 0.3),
        panel.grid.major.y = element_line(color = "grey85", linewidth = 0.3),
        panel.grid.minor   = element_blank(),
        # --------------------------
        axis.text.y = ggtext::element_markdown(size = 8),
        plot.margin = margin(t = 5.5, r = 5.5, b = 5.5, l = 10, unit = "mm")
      )
  }
  
  # =========================
  # Stable participant indices (unique anonymous_id -> 1..N)
  # =========================
  pid_index_tbl <- participant_watch_play %>%
    distinct(anonymous_id) %>%
    arrange(anonymous_id) %>%
    mutate(participant_index = row_number())
  
  # Groups with participant index attached; order by participant then session
  groups <- participant_watch_play %>%
    distinct(anonymous_id, session_id) %>%
    inner_join(pid_index_tbl, by = "anonymous_id") %>%
    arrange(participant_index, session_id)
  
  # =========================
  # Loop and save
  # =========================
  saved_files <- character(nrow(groups))
  
  for (gi in seq_len(nrow(groups))) {
    pid  <- groups$anonymous_id[gi]
    sid  <- groups$session_id[gi]
    pidx <- groups$participant_index[gi]
    
    df_i <- participant_watch_play %>%
      filter(anonymous_id == pid, session_id == sid)
    
    p <- make_panel(df_i, participant_index = pidx)
    
    sess_lab <- session_label_from_id(sid)
    filename <- file.path(
      participant_fig_dir,
      paste0("participant_", pidx, "_session_", gsub("[^A-Za-z0-9_-]", "_", sess_lab), "_watch_play.png")
    )
    
    ggsave(filename, plot = p, width = 14, height = 6, dpi = 200, bg = "white")
    saved_files[gi] <- filename
    message("Saved: ", filename)
  }
  
  message(sprintf("\nGenerated %d individual participant plots in: %s", 
                  length(saved_files), participant_fig_dir))
  
  # Return file paths invisibly
  invisible(saved_files)
}
