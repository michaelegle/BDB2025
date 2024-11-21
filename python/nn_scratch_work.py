import torch
import pandas as pd
import numpy as np
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(d_model=10, nhead=10)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(5, 7, 10)
# batch (frame), player, feature is the order for the tensor

out = transformer_encoder(src)

test_model = nn.Transformer(d_model = 8,
                            nhead = 8)

test_model.to('cuda:0')

print(test_model)

print(src)

print(src.shape)

