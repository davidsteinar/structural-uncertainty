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

dataset = z24Dataset(mode='testing', window_size=100, normalize=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = torch.load(f='../results/trained_autoencoder_correct.pt', map_location='cpu')
loss_criterion = torch.nn.MSELoss(reduce=False)

all_window_loss = []
for X, Y in dataloader:
    X_tensor = X.float().to(device)
    Y_tensor = Y.float().to(device)

    batch_size, output_size = Y.shape
    N = 100
    N_predictions = torch.zeros([N, batch_size, output_size])

    for i in range(N):
        N_predictions[i,:,:] = model(X_tensor)

    prediction_mean = torch.mean(N_predictions, dim=0)
    prediction_std = torch.std(N_predictions, dim=0)

    loss_full = loss_criterion(prediction_mean, Y_tensor)

    lower_y = prediction_mean - 2*prediction_std
    upper_y = prediction_mean + 2*prediction_std
    within_lower = Y_tensor > lower_y
    within_upper = Y_tensor < upper_y
    within_range = within_lower & within_upper

    loss_full[within_range] = 0

    for j in range(batch_size):
        window_loss = torch.sum(loss_full[j,:]) / torch.numel(loss_full[j,:])
        all_window_loss.append(window_loss.item())

losses_no_outliers = np.sort(all_window_loss)[:int(0.95*len(all_window_loss))]

#TODO add mean error to each sensor

np.save('../tools/testset_error_distribution.npy', losses_no_outliers)