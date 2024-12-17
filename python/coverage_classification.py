import torch
import pandas as pd
import numpy as np
# import torch.nn as nn
from helpers import util 
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import *


import time

def train_model(week_number):

    print("Beginning Model Training for Week " + str(week_number))
    start_time = time.time()
    #print("Beginning Process to Train Model with Data from Week " + str(week_number))
    #print("Importing Data")
    #track = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_" + str(week_number) + ".csv")
    #plays = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/plays.csv")
    #players = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/players.csv")
    device = torch.device("cuda:0")

    #print("Beginning Data Manipulation Process")
    #track_processed = util.process_tracking_data(track, plays)

    #print("Creating Relative Defensive Player to Offensive Player Features")
    #track_processed = util.add_relative_features(track_processed, players)

    #print("Building Data Into Tensor Format")
    #x_tensor, y_tensor = util.reformat_model_data(track_processed, device)

    #print(x_tensor.shape)
    #print("--- Model Data Loaded and Formatted in %s Seconds ---" % (time.time() - start_time))


    #x_tensor.to(device)
    #y_tensor.to(device)


    model = CoverageClassifier(num_players = 11, num_features = 15, num_classes = 9, num_heads = 8, num_layers = 6, model_dim = 64)
    model.to(device = device)

    # Assign batch size
    batch_size = 32
    patience = 5

    # Assign the optimizer and create a decaying learning schedule
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)


    loss_fn = torch.nn.CrossEntropyLoss()

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/coverage_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 30

    

    start_time = time.time()

    
    print("Beginning Model Training")

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

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

            #print(outputs.shape)
            #print(outputs)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            # scheduler.step()
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
    
    for wk in range(1, 2):
        #if wk == week_number:
        #    continue

        
        print("Training Model on Data from Week " + str(wk))
        
        x_tensor = torch.load("data/model_data/week_" + str(wk) + "_x_tensor.pt").to(device)
        y_tensor = torch.load("data/model_data/week_" + str(wk) + "_y_tensor.pt").to(device)

        dataset = CoverageDataset(frames = x_tensor, labels = y_tensor)
        dataset_size = len(dataset)

        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        test_size = 0

        #train_dataset = Subset(dataset = dataset, indices = range(train_size))
        #val_dataset = Subset(dataset = dataset, indices = range(train_size, dataset_size))

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        # Create the train/test/validation sets
        training_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        # testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

        #x, y = dataset.__getitem__(1)

        # Reset the best validation loss each week of training to avoid overfitting on one week
        best_vloss = 1_000_000.
        epochs_with_no_improvement = 0
        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(epoch_number, writer)

            running_vloss = 0.0
            running_vaccuracy = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(val_loader):
                    vinputs, vlabels = vdata
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    vaccuracy = util.calculate_accuracy(voutputs, vlabels)
                    running_vloss += vloss
                    running_vaccuracy += vaccuracy

            avg_vloss = running_vloss / (i + 1)
            avg_vaccuracy = running_vaccuracy / (i + 1)

            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('ACCURACY validation {}'.format(avg_vaccuracy))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                epochs_with_no_improvement = 0
                model_path = 'models/model_week_{}.pth'.format(week_number)
                torch.save(model.state_dict(), model_path)
            else:
                epochs_with_no_improvement += 1

            if epochs_with_no_improvement >= patience:
                print(f"Early stopping at Epoch {epoch_number + 1}")
                break

            #scheduler.step()
            epoch_number += 1

    print("--- %s seconds ---" % (time.time() - start_time))

    """ model.eval()

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
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Generate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(all_labels, all_preds)) """


model_training_start_time = time.time()
train_model(1)
print("--- Model Training Completed in %s seconds ---" % (time.time() - model_training_start_time))