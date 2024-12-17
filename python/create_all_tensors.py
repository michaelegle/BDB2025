import torch
import pandas as pd
import numpy as np
# import torch.nn as nn
from helpers import util 
# from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import *

import time

def create_week_tensor(week_number):
    start_time = time.time()
    print("Beginning Process to Train Model with Data from Week " + str(week_number))
    print("Importing Data")
    track = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_" + str(week_number) + ".csv")
    plays = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/plays.csv")
    players = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/players.csv")
    device = torch.device("cuda:0")

    print("Beginning Data Manipulation Process")
    track_processed = util.process_tracking_data(track, plays)

    print("Creating Relative Defensive Player to Offensive Player Features")
    track_processed = util.add_relative_features(track_processed, players)

    print("Building Data Into Tensor Format")
    x_tensor, y_tensor = util.reformat_model_data(track_processed, device)
    print("--- Model Data Loaded and Formatted in %s Seconds ---" % (time.time() - start_time))

    torch.save(x_tensor, "data/model_data/week_" + str(week_number) + "_x_tensor.pt")
    torch.save(y_tensor, "data/model_data/week_" + str(week_number) + "_y_tensor.pt")

    print("Saved Tensors for Future Use")

create_week_tensor(8)
create_week_tensor(9)