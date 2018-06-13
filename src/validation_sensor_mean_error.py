import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from z_24dataset import z24Dataset
from shallow_model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100

dataset = z24Dataset(mode='validating', window_size=100, normalize=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = torch.load(f='../results/trained_autoencoder_correct.pt', map_location='cpu')
loss_criterion = torch.nn.MSELoss(reduce=False)


all_window_loss = np.zeros([len(dataset), 100,7])
i = 0
for X, Y in dataloader:
    X_tensor = X.float().to(device)
    Y_tensor = Y.float().to(device)

    batch_size, output_size = Y.shape

    prediction = model(X_tensor)

    loss_full = loss_criterion(prediction, Y_tensor)

    for j in range(batch_size):
        window_loss = loss_full[j,:].detach().numpy()
        window_reshaped = np.reshape(window_loss, (100,7))
        all_window_loss[i,:,:] = window_reshaped
        
        i += 1
        
final_val_errors = np.concatenate(all_window_loss)
final_val_means = np.mean(final_val_errors, axis=0)

np.save('../tools/validation_sensor_mean_error.npy', final_val_means)