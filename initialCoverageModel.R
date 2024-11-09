library(tidyverse)

source("~/Downloads/BDB2025/helpers.R")

# Read in and process tracking data
tracking_data <- read_tracking_data()
tracking_data <- process_tracking_data(tracking_data)

df_plays <- read_csv("/Users/ajaypatel/Downloads/data/plays.csv")

# Get positions from player_data
player_data <- read_csv("/Users/ajaypatel/Downloads/data/players.csv") %>% 
  select(nflId, position)

# Join positions to tracking data
tracking_data <- left_join(tracking_data, player_data, by = c("nflId"))

# We only care about pre snap information
tracking_data <- tracking_data %>% filter(frameType == 'BEFORE_SNAP')

# Get a glimpse of our data
tracking_data %>% 
  head(20) %>% 
  View()

### Target variables of interest
# df_plays$pff_manZone
# df_plays$pff_passCoverage

# See unique positions
unique(tracking_data$position)

defense_positions <- c("DE", "CB", "SS", "ILB", "DT", "OLB", "FS", "MLB", "NT", "LB", "DB")

# Filter our tracking data to only defenders
tracking_data <- tracking_data %>% 
  filter(position %in% defense_positions)

df_plays_small <- df_plays %>% 
  select(gameId, playId, pff_passCoverage, pff_manZone)

# Merge data
df_merged <- merge(tracking_data, df_plays_small, by = c("gameId", "playId"))

df_merged %>% 
  head(20) %>% 
  View()

# Calculate initial position and how much they moved
df_merged <- df_merged %>%
  group_by(gameId, playId, nflId) %>%
  mutate(
    initial_x = first(x),
    initial_y = first(y),
    movement_distance = sum(sqrt(diff(x)^2 + diff(y)^2), na.rm = TRUE)
  ) %>%
  # Filter dataframe down to first frame now that we have total distance moved
  filter(frameId == 1)

# Create a unique identifier for each play
df_merged <- df_merged %>%
  mutate(play_id = paste(gameId, playId, sep = "_"))

# Group by play and assign a unique number (1-11) to each defender in each play
df_merged <- df_merged %>%
  group_by(play_id) %>%
  mutate(defender_number = row_number()) %>%
  filter(defender_number <= 11) %>% 
  ungroup()

# Select columns to pivot wider
defender_columns <- c("nflId", "x", "y", "o", "initial_x", "initial_y", 
                      "movement_distance")

# Pivot wider, creating columns like defender_1_x, defender_2_y, etc.
df_wide <- df_merged %>%
  select(play_id, defender_number, all_of(defender_columns)) %>%
  pivot_wider(names_from = defender_number, values_from = all_of(defender_columns), 
              names_glue = "defender_{defender_number}_{.value}")

# View the resulting DataFrame
head(df_wide) %>% View()

# Join coverage information to wide dataframe
targets <- df_merged %>% 
  group_by(play_id) %>% 
  summarize(pff_passCoverage = unique(pff_passCoverage),
            pff_manZone = unique(pff_manZone))

df_wide <- left_join(df_wide, targets, by = c("play_id"))

# Ensure 'pff_manZone' and 'pff_passCoverage' are factors
df_wide <- df_wide %>%
  mutate(pff_manZone = as.factor(pff_manZone),
         pff_passCoverage = as.factor(pff_passCoverage)) %>% 
  filter(!is.na(pff_manZone), !is.na(pff_passCoverage))

library(caret)
library(randomForest)

# Split data into train and test sets (e.g., 80% train, 20% test)
set.seed(123)
trainIndex <- createDataPartition(df_wide$pff_manZone, p = 0.8, list = FALSE)
trainData <- df_wide[trainIndex, ]
testData <- df_wide[-trainIndex, ]

# 1. Model for Man/Zone Prediction (Binary Classification)
manZone_model <- randomForest(pff_manZone ~ ., data = trainData, ntree = 100, mtry = 3, importance = TRUE)

# Evaluate the model on test data
manZone_pred <- predict(manZone_model, newdata = testData)
confusionMatrix(manZone_pred, testData$pff_manZone) ## ~98% accuracy, confusion matrix looks really strong

# Train a model on the filtered data to predict 'pff_passCoverage'
coverage_model <- randomForest(pff_passCoverage ~ ., data = trainData, ntree = 100, mtry = 3, importance = TRUE)

coverage_pred <- predict(coverage_model, newdata = testData)
conf_matrix <- confusionMatrix(coverage_pred, testData$pff_passCoverage) ## 56% accuracy, need to do cross-validation or 
# class balancing next but initial results are promising

# Extract confusion matrix table
conf_matrix_table <- as.data.frame(as.table(conf_matrix))

# Plot confusion matrix using ggplot
ggplot(conf_matrix_table, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(x = "Predicted", y = "Actual", fill = "Frequency") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = Freq), color = "black", size = 5)  # Add text labels to cells

