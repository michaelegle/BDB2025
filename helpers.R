library(tidyverse)
library(tidymodels)

read_tracking_data <- function(weeks = seq(1, 9)) {
  
  
  #blank dataframe to store tracking data
  df_tracking <- data.frame()
  
  #iterating through all weeks
  for (w in weeks) {
    #temporary dataframe used for reading week for given iteration
    df_tracking_temp <-
      read_csv(paste0("data/tracking_week_", w, ".csv"),
               col_types = cols())
    
    #storing temporary dataframe in full season dataframe
    df_tracking <-
      bind_rows(df_tracking_temp, df_tracking)
    
  }
  
  df_tracking
}

# a helper function to standardize the tracking data for field position where offense is always going left to right
process_tracking_data <- function(df) {
  df <- df %>%
    mutate(
      x = ifelse(playDirection == "left", 120 - x, x),
      y = ifelse(playDirection == "left", 160 / 3 - y, y),
      dir = ifelse(playDirection == "left", dir + 180, dir),
      dir = ifelse(dir > 360, dir - 360, dir),
      o = ifelse(playDirection == "left", o + 180, o),
      o = ifelse(o > 360, o - 360, o)
    )
  
  df
}

# a helper function to remove penalties
process_plays_data <- function(df) {
  #create a is penalty column
  df <- df %>%
    mutate(is_penalty = ifelse(is.na(foulName1), 0, 1))
  
  df <- df %>%
    filter(is_penalty == 0)
  
  df
}

add_play_details <- function(tracking_df){
  ftn_data <- nflreadr::load_ftn_charting(2022)
  
  pbp_data <- nflfastR::load_pbp(2022) %>% 
    left_join(ftn_data,
              by =c("game_id" = "nflverse_game_id",
                    "play_id" = "nflverse_play_id",
                    "week" = "week")) %>% 
    filter(week <= 9) %>% 
    select(gameId = old_game_id, playId = play_id, week, play_type, pass, rush, qb_dropback,
           qb_scramble, yards_gained, no_huddle, air_yards, qb_kneel, pass_attempt,
           pass_location, pass_length, run_location, run_gap, yards_after_catch, out_of_bounds,
           # FTN-specific data to include:
           n_offense_backfield, is_screen_pass, is_trick_play, is_rpo, is_play_action) %>% 
    mutate(gameId = as.numeric(gameId))
  
  fin_df <- tracking_df %>% 
    left_join(pbp_data)
  
  fin_df
  
}