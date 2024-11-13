library(tidyverse)

setwd("C:/Users/Michael Egle/BDB2025")

# test each week's tracking data to look for missing frames, players etc

#' Check for missingness in tracking data
#'
#' @param week_num - week number to test
#'
#' @return - a data frame of plays and frames with missing data
#'
check_for_missing <- function(week_num)
{
  week <- read_csv(paste0("data/tracking_week_", week_num, ".csv"))
  
  grouped_by_frame <- week %>% 
    group_by(gameId, playId, frameId) %>% 
    count() %>% 
    filter(n != 23) %>% 
    select(-n)
  
  grouped_by_play <- week %>% 
    group_by(gameId, playId) %>% 
    summarize(min_frame = min(frameId),
              max_frame = max(frameId),
              frames_observed = n_distinct(frameId),
              frames_expected = max_frame - min_frame + 1) %>% 
    filter(frames_observed != frames_expected) %>% 
    select(gameId, playId)
  
  missing <- grouped_by_frame %>% 
    bind_rows(grouped_by_play)
  
  # TODO - test for missing event labels - must have some tackle/OOB event
  
  return(missing)
}

missing <- pmap_dfr(.l = list(seq(1, 9)),
                    .f = check_for_missing,
                    .progress = T)

# only one play has some frames with missing data

unique(week1$event)

# Make sure each play has a frame with an event tagging a start and end point
# start points: pass_outcome_caught (NOT pass arrival)

start_events <- c("pass_outcome_caught", "run", "handoff", "lateral")

end_events <- c("tackle", "touchdown", "out_of_bounds", "fumble_defense_recovered",
                "qb_slide", "qb_sack", 
                # will have to have a sort of precedence in case some events are missing
                "fumble")

# if a play doesn't have a certain start event, we may want to exclude it.
# this is because it could very well be outside of the scope of our project.
# ex: gameId 2022091102 and playId 4102 is Trey Lance fumbling the snap,
# we don't need this in our training sets

week1 %>% 
  group_by(gameId, playId) %>% 
  summarize(start_event = ifelse(any(event %in% start_events), 1, 0),
            end_event = ifelse(any(event %in% end_events), 1, 0)) %>% 
  filter(start_event == 0 | end_event == 0)

test <- week1_standardized %>% 
  filter(gameId == 2022091102, playId == 4102)



week1

