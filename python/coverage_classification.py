import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from helpers import util 
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import time
start_time = time.time()

print("running")

week1 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_1.csv")
#week2 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_2.csv")
#week3 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_3.csv")
#week4 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_4.csv")
#week5 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_5.csv")
#week6 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_6.csv")
#week7 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_7.csv")
#week8 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_8.csv")
#week9 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_9.csv")


#all_weeks = pd.concat([week1, week2])

plays = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/plays.csv")
players = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/players.csv")

device = torch.device("cuda:0")

week1_processed = util.process_tracking_data(week1, plays)
week1_processed = util.add_relative_features(week1_processed, players)

print(week1_processed)

print(week1_processed[(week1_processed['gameId'] == 2022091200) & (week1_processed['playId'] == 85) & (week1_processed['frameId'] == 1)])

asdfasdfjkoe

week1_processed_x = week1_processed[['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense', 'gameId', 'playId', 'frameId', 'nflId']]
week1_processed_y = week1_processed[['gameId', 'playId', 'frameId', 'pff_passCoverage']].drop_duplicates()

print(week1_processed_y['pff_passCoverage'].drop_duplicates())

print(week1_processed_x)
print(week1_processed_y)


features = ['x', 'rel_x', 'y', 'dir', 'o', 's', 'a']

week1_pivoted = week1_processed_x.pivot_table(index = ['gameId', 'playId', 'frameId'], values = features, aggfunc = list)

print(week1_pivoted)

tensor_data = np.array(week1_pivoted.values.tolist(), dtype = 'float64')

print(tensor_data)

tensor = torch.tensor(tensor_data, device = device).transpose(1, 2).to(torch.float)

print(tensor.shape)
print(tensor)


# TODO - might be best to have a more sensical ordering for these indices, but since you can just map them back together at the end, I don't think it's a huge deal
coverage_class_to_index = {
    "Cover-3": 0,
    "Quarters": 1,
    "Cover-1": 2,
    "Cover-6": 3, 
    "Cover-2": 4,
    "Cover-0": 5,
    "Red Zone/Goal Line": 6,
    "Miscellaneous": 7,
    "Bracket": 8
}

#subset_features = tensor[:10000, :, :]
#subset_labels = week1_processed_y[:10000]

subset_labels_tensor = torch.tensor([coverage_class_to_index[t] for t in week1_processed_y['pff_passCoverage']]).to(device=device)

#print(subset_labels['pff_passCoverage'].unique())

print(subset_labels_tensor)

print("--- %s seconds ---" % (time.time() - start_time))

""" 
test = torch.tensor(week1_processed_x.groupby(['gameId', 'playId', 'frameId']).apply(week1_processed_x[['x', 'rel_x', 'y', 'dir', 'o', 's', 'a', 'on_defense']].values),
                     device = device)
"""

class CoverageClassifier(nn.Module):
    # num_features = 8
    # num_players = 22

    def __init__(self, num_players, num_features, num_classes, num_heads, num_layers, model_dim, dropout=0.1):
        super(CoverageClassifier, self).__init__()

        ff_size = model_dim * 4
        # Normalize the batch
        self.normalization_layer = nn.BatchNorm1d(num_features)


        # Embedding layer
        #self.embedding = nn.Embedding(num_players, num_features)
        #self.num_players = num_players
        #self.num_features = num_features
        
        self.embedding = nn.Sequential(
            nn.Linear(num_features, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        # Transformer encoder layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = model_dim, nhead = num_heads, dim_feedforward = ff_size, dropout = dropout),
            num_layers = num_layers
        )
        

        # this final linear layer will map the transformer layer outputs to class predictions
        self.fc = nn.Linear(model_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        #print(x)
        #print(self.embedding)
        #print(x.shape)
        #print(x.squeeze().shape)
        #print(x.type())
        #embedded = self.embedding(x.squeeze()) # (batch_size, seq_length, embed_size)
        #embedded = embedded.permute(1, 0, 2)  # (seq_length, batch_size, embed_size) for transformer
        #print(x.shape)
        #print(self.embedding)
        #print(x.shape)
        x_normalized = self.normalization_layer(x.permute(0, 2, 1))
        #print(x_normalized.shape)
        x_embedded = self.embedding(x_normalized.permute(0, 2, 1))

        #print(x_embedded.shape)
        transformer_output = self.transformer(x_embedded)
        #print(transformer_output.shape)
        output = transformer_output.mean(dim=1)  # average pooling
        #print(output.shape)
        logits = self.fc(output)  # (batch_size, num_classes)
        #print(logits.shape)
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

#print(subset_features)
#print(subset_labels)

dataset = CoverageDataset(frames = tensor, labels = subset_labels_tensor)
dataset_size = len(dataset)

#x, y = dataset.__getitem__(1)

""" print(x)
print(x.dim())
print(y)
 """
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

criterion = torch.nn.MSELoss(reduction='sum')
model = CoverageClassifier(num_players = 11, num_features = 7, num_classes = 9, num_heads = 8, num_layers = 6, model_dim = 64)

model.to(device)

batch_size = 16

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)

loss_fn = torch.nn.CrossEntropyLoss()

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    running_accuracy = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero the gradients for each batch
        optimizer.zero_grad()

        #print(inputs.shape)
        #print(labels.shape)

        # Make predictions for this batch
        outputs = model(inputs)

        batch_accuracy = util.calculate_accuracy(outputs, labels)

        #print(outputs.shape)
        #print(outputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        running_accuracy += batch_accuracy
        
        # Adjust learning weights
        # scheduler.step()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            print('  batch {} accuracy: {}'.format(i + 1, running_accuracy / (i + 1)))
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

start_time = time.time()



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

    scheduler.step()
    epoch_number += 1

print("--- %s seconds ---" % (time.time() - start_time))

model.eval()

all_labels = []
all_preds = []

# Disable gradient calculation
with torch.no_grad():
    for data in testing_loader:
        inputs, labels = data
        
        # Forward pass: Compute predicted outputs
        outputs = model(inputs)
        
        # Get the predicted class (index with the maximum score)
        _, predicted = torch.max(outputs, 1)
        
        # Store the true labels and predicted labels
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Generate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels = test_dataset.classes, yticklabels = test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(all_labels, all_preds, target_names = test_dataset.classes))