import numpy as np
import json
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hyperopt import fmin, tpe, hp, pyll, Trials

from z24_dataset import z24Dataset
from deep_model import Model


n_epochs = 50
batch_size = 1000
max_evals = 5
window_size = 200


        
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
    
training_dataset = z24Dataset(mode='training', window_size=window_size, normalize=True)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

validating_dataset = z24Dataset(mode='validating', window_size=window_size, normalize=True)
validating_dataloader = DataLoader(validating_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


hyperspace = {'learning_rate': hp.uniform('learningrate', low=0.00005, high=0.001),
         'dropout_p': hp.uniform('dropout', low=0.05, high=0.3),
         'hidden_size1': hp.quniform('hidden_size1', low=256,high=512,q=1),
         'hidden_size2': hp.quniform('hidden_size2', low=128,high=256,q=1),
         'z_size': hp.quniform('z_size', low=64,high=128,q=1),
         'weight_init': hp.choice('init', [init.xavier_normal_,
                                           init.xavier_uniform_,
                                           init.kaiming_normal_,
                                           init.kaiming_uniform_,
                                           init.orthogonal_])}


def f(hyperspace):
    
    print(str(hyperspace))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #define model
    model = Model(input_size=7*window_size+53,
              output_size=7*window_size,
              hidden_size1=int(hyperspace['hidden_size1']),
              hidden_size2=int(hyperspace['hidden_size2']),
              z_size=int(hyperspace['z_size']),
              dropout_p=hyperspace['dropout_p']).to(device)
    print(model)
    
    Initializer.initialize(model=model, initialization=hyperspace['weight_init'])
    
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=hyperspace['learning_rate'],
                                betas=(0.9, 0.99),
                                eps=1e-08,
                                weight_decay=0,
                                amsgrad=True)
                                
    loss_criterion = torch.nn.MSELoss()
    
    #training
    train_loss = []
    for epoch in range(n_epochs):
        batchloss_train = []
        
        for X_train, Y_train in training_dataloader:
            X_train_tensor = X_train.float().to(device)
            Y_train_tensor = Y_train.float().to(device)
    
            optimizer.zero_grad()  # zero the gradient buffer
            predicted_Y = model(X_train_tensor)
            batch_train_loss = loss_criterion(predicted_Y, Y_train_tensor)
            batch_train_loss.backward()
            optimizer.step()
        
            batchloss_train.append(batch_train_loss.item())
            
        train_loss.append(np.mean(batchloss_train))
        print('Train loss: {}'.format(np.mean(batchloss_train)))
        
    val_loss = []
    with torch.no_grad():
        for X_val, Y_val in validating_dataloader:
            X_val_tensor = X_val.float().to(device)
            Y_val_tensor = Y_val.float().to(device)
            predicted_Y = model(X_val_tensor)
            batch_val_loss = loss_criterion(predicted_Y, Y_val_tensor)
            val_loss.append(batch_val_loss.item())
            
    
    final_val_loss = np.mean(val_loss)
    
    print('valiation loss: {}'.format(final_val_loss))
    
    return final_val_loss

trials = Trials()

best = fmin(
    fn=f,
    space=hyperspace,
    algo=tpe.suggest,
    max_evals = max_evals)
    
print('###Best model###')
print(best)