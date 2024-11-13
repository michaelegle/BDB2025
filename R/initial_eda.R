library(tidyverse)

source("helpers.R")

# Read in and process tracking data
tracking_data <- read_tracking_data()
tracking_data <- process_tracking_data(tracking_data)

# Focus on a specific game for the sake of EDA
tracking_data <- tracking_data %>% 
  filter(gameId == "2022110700")

# Read in plays data
df_plays <- read_csv("/Users/ajaypatel/Downloads/data/plays.csv")

# Look at different possible coverage labels
unique(df_plays$pff_passCoverage)

# Counts of specific coverage types
df_plays %>% 
  group_by(pff_passCoverage) %>% 
  summarise(count = n()) %>% 
  arrange(desc(count))

# This isn't binary for whatever reason, stored as NA, "Zone", "Man", "Other"
unique(df_plays$pff_manZone)

# Not sure if we can fully remove Other coverages (818 plays)
df_plays %>% 
  group_by(pff_manZone) %>% 
  summarise(count = n()) %>% 
  arrange(desc(count))

# Read in players' play data
df_player_play <- read_csv("/Users/ajaypatel/Downloads/data/player_play.csv")

df_player_play %>% 
  group_by(inMotionAtBallSnap) %>% 
  summarise(n = n())

# Motioned players are targeted 12% of the time, ~7% if not motioned
df_player_play %>% 
  group_by(inMotionAtBallSnap) %>% 
  summarise(targeted = sum(wasTargettedReceiver) / n())

# What routes are players in motion running
df_player_play %>% 
  filter(inMotionAtBallSnap == TRUE) %>% 
  group_by(routeRan) %>% 
  summarise(n = n())
