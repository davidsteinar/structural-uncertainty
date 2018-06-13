import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np


loss_full = torch.nn.MSELoss(reduce=False)

batchloss_train = []
for X_train, Y_train in validating_dataloader:
    X_train_tensor = X_train.float().to(device)
    Y_train_tensor = Y_train.float().to(device)
    
    batch_size, output_size = Y_train.shape
    N = 100
    N_predictions = torch.zeros([N, batch_size, output_size])
    
    for i in range(N):
        N_predictions[i,:,:] = model(X_train_tensor)
    
    prediction_mean = torch.mean(N_predictions, dim=0)
    prediction_std = torch.std(N_predictions, dim=0)

    trainloss_full = loss_full(predicted_Y, Y_train_tensor)
    
    lower_y = predicted_Y - 2*prediction_std
    upper_y = predicted_Y + 2*prediction_std
    within_lower = Y_train_tensor > lower_y
    within_upper = Y_train_tensor < upper_y
    within_range = within_lower & within_upper
    
    trainloss_full[within_range] = 0
    
    uncertainty_loss = torch.sum(trainloss_full)/ torch.numel(trainloss_full)

    batchloss_train.append(uncertainty_loss.item())