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

test_x = torch.load("data/model_data/week_2_x_tensor.pt")
test_y = torch.load("data/model_data/week_2_y_tensor.pt")

model = CoverageClassifier(num_players = 11, num_features = 15, num_classes = 9, num_heads = 8, num_layers = 6, model_dim = 64)
model.state_dict(torch.load('models/model_week_1.pth'))
model.eval()

dataset = CoverageDataset(frames = test_x, labels = test_y)
test_loader = DataLoader(dataset, batch_size = 32, shuffle = False)

device = torch.device("cuda:0")

model.to(device)

predictions = torch.empty(0, 9)
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        predictions = torch.cat([predictions, outputs.cpu()])

        _, predicted_label = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted_label.cpu().numpy())

print(predictions)

print(util.calculate_accuracy(predictions, test_y.cpu()))

print(confusion_matrix(true_labels, predicted_labels))