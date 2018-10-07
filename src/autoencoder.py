import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

from z24_dataset import z24Dataset
from deep_model import Model

print('###Starting script###')
print('torch version: {}'.format(torch.__version__))
print('Number CUDA Devices: {}'.format(torch.cuda.device_count()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--wsize', type=int, default=100)
args = parser.parse_args()

n_epochs = args.nepochs
batch_size = args.batchsize
window_size = args.wsize

print('batch size: {}'.format(batch_size))
print('epochs: {}'.format(n_epochs))

class Initializer:
    # to apply xavier_uniform:
    #Initializer.initialize(model=net, initialization=init.xavier_uniform_, gain=init.calculate_gain('relu'))
    # or maybe normal distribution:
    #Initializer.initialize(model=net, initialization=init.normal_, mean=0, std=0.2)
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass
        model.apply(weights_init)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Datasets
training_dataset = z24Dataset(mode='training', window_size=window_size, normalize=True)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

validating_dataset = z24Dataset(mode='validating', window_size=window_size, normalize=True)
validating_dataloader = DataLoader(validating_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

#hyperparameters
hyperspace = {
'dropout_p': 0.1,
'learning_rate': 0.0003,
'weight_init': init.orthogonal_,
'hidden_size1': 512,
'hidden_size2': 256,
'z_size': 128}

#define model
model = Model(input_size=7*window_size+53,
              output_size=7*window_size,
              hidden_size1=hyperspace['hidden_size1'],
              hidden_size2=hyperspace['hidden_size2'],
              z_size=hyperspace['z_size'],
              dropout_p=hyperspace['dropout_p']).to(device)

print(model)
print(count_parameters(model))
print(str(hyperspace))

Initializer.initialize(model=model, initialization=hyperspace['weight_init'])

optimizer = torch.optim.Adam(model.parameters(),
                            lr=hyperspace['learning_rate'],
                            betas=(0.9, 0.99),
                            eps=1e-08,
                            weight_decay=0,
                            amsgrad=True)
                            
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(n_epochs*0.7)], gamma=0.1)
loss_criterion = torch.nn.MSELoss()


#training
train_loss = []
val_loss   = []
for epoch in range(n_epochs):
    batchloss_train = []
    
    for X_train, Y_train in training_dataloader:
        X_train_tensor = X_train.float().to(device)
        Y_train_tensor = Y_train.float().to(device)

        optimizer.zero_grad()  # zero the gradient buffer
        predicted_Y = model(X_train_tensor)
        trainloss = loss_criterion(predicted_Y, Y_train_tensor)
        trainloss.backward()
        optimizer.step()
    
        batchloss_train.append(trainloss.item())
        
    scheduler.step()
    train_loss.append(np.mean(batchloss_train))
    
    batchloss_val = []
    with torch.no_grad():
        for X_val, Y_val in validating_dataloader:
            X_val_tensor = X_val.float().to(device)
            Y_val_tensor = Y_val.float().to(device)
            predicted_Y = model(X_val_tensor)
            valloss = loss_criterion(predicted_Y, Y_val_tensor)
            batchloss_val.append(valloss.item())
        
    val_loss.append(np.mean(batchloss_val))
    
    print('Epoch: {}'.format(epoch))
    print('train loss: {}'.format(np.mean(batchloss_train)))
    print('valiation loss: {}'.format(np.mean(batchloss_val)))
    
#save trained model
torch.save(model, '../results/trained_deep_relu200_20epochs.pt')