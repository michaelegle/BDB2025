import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from helpers import util 


week1 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_1.csv")
plays = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/plays.csv")

device = torch.device("cuda:0")

week1_processed = util.process_tracking_data(week1, plays)
print(week1_processed)

week1_processed_x = week1_processed[['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense', 'gameId', 'playId', 'frameId', 'nflId']]

print(week1_processed_x)

features = ['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense']

week1_pivoted = week1_processed_x.pivot_table(index = ['gameId', 'playId', 'frameId'], values = features, aggfunc = list)

print(week1_pivoted)

tensor_data = np.array(week1_pivoted.values.tolist(), dtype = 'float64')

print(tensor_data)

tensor = torch.tensor(tensor_data, device = device).transpose(1, 2)

print(tensor.shape)
print(tensor)

""" 
test = torch.tensor(week1_processed_x.groupby(['gameId', 'playId', 'frameId']).apply(week1_processed_x[['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense']].values),
                     device = device)
"""

