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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--maxevals', type=int, default=50)

args = parser.parse_args()

n_epochs = args.nepochs
batch_size = args.batchsize
max_evals = args.maxevals

print('###Starting script###')
print('torch version: {}'.format(torch.__version__))
print('Number CUDA Devices: {}'.format(torch.cuda.device_count()))
print('batch size: {}'.format(batch_size))
print('epochs: {}'.format(n_epochs))


class z24Dataset(Dataset):
    def __init__(self, mode='training', window_size=100, normalize=True):
        self.window_size = window_size
        self.slices_per_file = 65536 // self.window_size
        self.normalize = normalize
        
        if mode == 'training':
            self.index_file = np.loadtxt('../tools/training_set_index.txt',dtype=str)
        elif mode == 'testing' :
            self.index_file = np.loadtxt('../tools/test_set_index.txt',dtype=str)
        elif mode == 'validating':
            self.index_file = np.loadtxt('../tools/validation_set_index.txt',dtype=str)
        
        self.name_index_dict = dict(zip(range(len(self.index_file)),list(self.index_file)))
        
        self.env_mean = np.load('../tools/env_mean.npy')
        self.env_std = np.load('../tools/env_std.npy')
        self.vibration_mean = np.load('../tools/vibration_mean.npy')
        self.vibration_std = np.load('../tools/vibration_std.npy')

    def __len__(self):
        return len(self.index_file) * self.slices_per_file

    def __getitem__(self, index):
        index_to_read = index // self.slices_per_file
        file_to_read = self.name_index_dict[index_to_read]
        index_in_dataframe = (index - index_to_read*self.slices_per_file) * self.window_size
        
        X_vibration = np.load(file='../data/z24_clean/'+file_to_read+'_vibrations.npy')
        X_environmental = np.load(file='../data/z24_clean/'+file_to_read+'_env.npy')
        
        X_vibration_window = X_vibration[index_in_dataframe:index_in_dataframe+self.window_size,:]

        if self.normalize:
            X_vibration_window = (X_vibration_window - self.vibration_mean) / self.vibration_std
            X_environmental = (X_environmental - self.env_mean) / self.env_std
        
        X_vib_and_env = np.append(X_vibration_window.flatten(),X_environmental)
       
        return X_vib_and_env, X_vibration_window
        
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, z_size, output_size, dropout_p):
        super(Model, self).__init__()
        self.dropout_p = dropout_p

        self.h1 = nn.Linear(input_size, hidden_size)
        self.z  = nn.Linear(hidden_size, z_size)
        self.h2 = nn.Linear(z_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h1(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.z(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h2(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.h3(x)
        return x
        
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
training_dataset = z24Dataset(mode='training', window_size=100, normalize=True)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

validating_dataset = z24Dataset(mode='validating', window_size=100, normalize=True)
validating_dataloader = DataLoader(validating_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


space = {'learning_rate': hp.uniform('learningrate', low=0.00005, high=0.01),
         'dropout_p': hp.uniform('dropout', low=0.05, high=0.3),
         'hidden_size': hp.quniform('hidden_size', low=50,high=500,q=1),
         'z_size': hp.quniform('z_size', low=10,high=50,q=1),
         'weight_init': hp.choice('init', [init.xavier_normal_,
                                           init.xavier_uniform_,
                                           init.kaiming_normal_,
                                           init.kaiming_uniform_])}


def f(space):
    learning_rate = space['learning_rate']
    dropout_p = space['dropout_p']
    hidden_size = int(space['hidden_size'])
    z_size = int(space['z_size'])
    weight_init = space['weight_init']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #define model
    model = Model(input_size=7*100,
                  hidden_size=hidden_size,
                  z_size=z_size,
                  output_size=7*100,
                  dropout_p=dropout_p).to(device)
    
    Initializer.initialize(model=model, initialization=weight_init)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                betas=(0.9, 0.99),
                                eps=1e-08,
                                weight_decay=0,
                                amsgrad=True)
                                
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
            
        #scheduler.step()
        train_loss.append(np.mean(batchloss_train))
        
        batchloss_val = []
        with torch.no_grad():
            for X_val, Y_val in validating_dataloader:
                X_val_tensor = X_val.float().to(device)
                Y_val_tensor = Y_val.float().to(device)
                predicted_Y = model(X_val_tensor)
                valloss = loss_criterion(predicted_Y, Y_val_tensor)
                batchloss_val.append(valloss.item())
            
        #scheduler.step()
        val_loss.append(np.mean(batchloss_val))
        
        print('Epoch: {}'.format(epoch))
        print('train loss: {}'.format(np.mean(batchloss_train)))
        print('valiation loss: {}'.format(np.mean(batchloss_val)))
    
    final_val_loss = val_loss[-1]
    parameters = space
    parameters['train_loss'] = train_loss[-1]
    parameters['val_loss'] = val_loss[-1]
    print(str(parameters))
    
    return final_val_loss

trials = Trials()


best = fmin(
    fn=f,
    space=space,
    algo=tpe.suggest,
    max_evals = max_evals)
    
print('###Best model###')
print(best)