import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np

print('###Starting script###')
print('torch version: {}'.format(torch.__version__))
print('Number CUDA Devices: {}'.format(torch.cuda.device_count()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--nepochs', type=int, default=50)
args = parser.parse_args()

n_epochs = args.nepochs
batch_size = args.batchsize

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
        
        file_path_vib = '../data/z24_clean/'+file_to_read+'_vibrations.npy'
        file_path_env = '../data/z24_clean/'+file_to_read+'_env.npy'
        
        memmap_vib = np.memmap(file_path_vib, dtype=np.float64, mode='r', shape=(65536, 7))
        memmap_env = np.memmap(file_path_env, dtype=np.float64, mode='r', shape=(53,))

        X_environmental = np.array(memmap_env[:])
        X_vibration_window = np.array(memmap_vib[index_in_dataframe:index_in_dataframe+self.window_size,:])

        if self.normalize:
            X_vibration_window = (X_vibration_window - self.vibration_mean) / self.vibration_std
            X_environmental = (X_environmental - self.env_mean) / self.env_std
        
        X_vib_and_env = np.append(X_vibration_window.flatten(),X_environmental)
       
        return X_vib_and_env, X_vibration_window.flatten()
        
class Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, z_size, output_size, dropout_p):
        super(Model, self).__init__()
        self.dropout_p = dropout_p

        self.h1 = nn.Linear(input_size, hidden_size1)
        self.h2 = nn.Linear(hidden_size1, hidden_size2)
        self.z  = nn.Linear(hidden_size2, z_size)
        self.h4 = nn.Linear(z_size, hidden_size2)
        self.h5 = nn.Linear(hidden_size2, hidden_size1)
        self.h6 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h1(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h2(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.z(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h4(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h5(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.h6(x)
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
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

validating_dataset = z24Dataset(mode='validating', window_size=100, normalize=True)
validating_dataloader = DataLoader(validating_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#hyperparameters
hyperspace = {
'dropout_p': 0.1,
'learning_rate': 0.005,
'weight_init': init.kaiming_uniform_,
'hidden_size1': 256,
'hidden_size2': 128,
'z_size': 64}

#define model
model = Model(input_size=7*100+53,
              output_size=7*100,
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
    
#save trained model
torch.save(model, '../results/trained_autoencoder_bigger.pt')