import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from helpers import util 
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import time
start_time = time.time()

print("running")

week1 = pd.read_csv("/Users/michaelegle/BDB2025/data/tracking_week_1.csv")
plays = pd.read_csv("/Users/michaelegle/BDB2025/data/plays.csv")

device = torch.device("cpu")

week1_processed = util.process_tracking_data(week1, plays)
print(week1_processed)

week1_processed_x = week1_processed[['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense', 'gameId', 'playId', 'frameId', 'nflId']]
week1_processed_y = week1_processed[['gameId', 'playId', 'frameId', 'pff_passCoverage']].drop_duplicates()

print(week1_processed_y['pff_passCoverage'].drop_duplicates())

print(week1_processed_x)
print(week1_processed_y)

features = ['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense']

week1_pivoted = week1_processed_x.pivot_table(index = ['gameId', 'playId', 'frameId'], values = features, aggfunc = list)

print(week1_pivoted)

tensor_data = np.array(week1_pivoted.values.tolist(), dtype = 'float64')

print(tensor_data)

tensor = torch.tensor(tensor_data, device = device).transpose(1, 2).to(torch.long)

print(tensor.shape)
print(tensor)




subset_features = tensor[:10000, :, :]
subset_labels = week1_processed_y[:10000]

print("--- %s seconds ---" % (time.time() - start_time))

""" 
test = torch.tensor(week1_processed_x.groupby(['gameId', 'playId', 'frameId']).apply(week1_processed_x[['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense']].values),
                     device = device)
"""

class CoverageClassifier(nn.Module):
    # num_features = 8
    # num_players = 22

    def __init__(self, num_players, num_features, num_classes, num_heads, num_layers, ff_size, dropout=0.1):
        super(CoverageClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_players, num_features)

        # Transformer encoder layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = num_features, nhead = num_heads, dim_feedforward = ff_size, dropout = dropout),
            num_layers = num_layers
        )
        
        # this final linear layer will map the transformer layer outputs to class predictions
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        embedded = embedded.permute(1, 0, 2)  # (seq_length, batch_size, embed_size) for transformer

        transformer_output = self.transformer(embedded)
        output = transformer_output.mean(dim=0)  # Pooling: take the mean across the sequence
        logits = self.fc(output)  # (batch_size, num_classes)
        return logits

class CoverageDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx, :, :]
        label = self.labels[idx]
        
        return frame, label

print(subset_features)
print(subset_labels)

dataset = CoverageDataset(frames = subset_features, labels = subset_labels['pff_passCoverage'].values)
dataset_size = len(dataset)

x, y = dataset.__getitem__(1)

print(x)
print(y)

train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


criterion = torch.nn.MSELoss(reduction='sum')
model = CoverageClassifier(num_players = 22, num_features = 8, num_classes = 10, num_heads = 8, num_layers = 10, ff_size = 2048)

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)


optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/coverage_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1