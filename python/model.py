import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CoverageClassifier(nn.Module):
    # num_features = 8
    # num_players = 22

    def __init__(self, num_players, num_features, num_classes, num_heads, num_layers, model_dim, dropout=0.1):
        super(CoverageClassifier, self).__init__()

        ff_size = model_dim * 4
        # Normalize the batch
        # TODO - remove if this isn't necessary
        self.normalization_layer = nn.BatchNorm2d(num_features)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = num_features, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(),
            #nn.Dropout(p = 0.25),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(),
            #nn.Dropout(p = 0.25),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(),
            #nn.Dropout(p = 0.25),

            nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(),
            #nn.Dropout(p = 0.25),

            nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
            #nn.Dropout(p = 0.25)
        )

        # Embedding layer
        
        self.embedding = nn.Sequential(
            nn.Linear(num_players, model_dim),
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

        x = self.normalization_layer(x)
        # Pass through convolutional layers
        x = self.conv_layers(x).squeeze(1)
        
        # Pass through embedding layer for the transformer
        x_embedded = self.embedding(x)

        # Pass through transformer layer
        transformer_output = self.transformer(x_embedded)
        
        # Average pool
        output = transformer_output.mean(dim=1)  # average pooling
        
        # Create logit predictions
        logits = self.fc(output)
        
        # Return predicted logits
        return logits

class CoverageDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx, :, :, :]
        label = self.labels[idx]
        
        return frame, label


