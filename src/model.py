import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np

print('###Starting script###')
print('torch version: {}'.format(torch.__version__))
print('Device count: {}'.format(torch.cuda.device_count()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class z24Dataset(Dataset):
    def __init__(self, mode='training', window_size=100, lookahead=1, normalize=True):
        self.window_size = window_size
        self.lookahead = lookahead
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
        Y = X_vibration[index_in_dataframe+self.window_size+1,:]
        
        if self.normalize:
            X_vibration_window = (X_vibration_window - self.vibration_mean) / self.vibration_std
            X_environmental = (X_environmental - self.env_mean) / self.env_std
        
        #return X_vibration_window, X_environmental, Y
        return X_vibration_window.flatten(), Y
    

# Neural Network Model

class Model(nn.Module):
    def __init__(self, input_size, hidden_width, n_hidden_layers, output_size, dropout_p):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.n_hidden_layers = n_hidden_layers
        
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_width)])
        for _ in range(1, self.n_hidden_layers):
            self.layers.extend([nn.Linear(hidden_width, hidden_width)])
        self.layers.append(nn.Linear(hidden_width, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout_p, training=True)
            x = F.relu(layer(x))
        #no dropout for last layer
        output = self.layers[-1](x)
        
        return output
    
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
training_dataset = z24Dataset(mode='training', window_size=100, lookahead=1, normalize=True)
training_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=True, num_workers=4)

validating_dataset = z24Dataset(mode='validating', window_size=100, lookahead=1, normalize=True)
validating_dataloader = DataLoader(validating_dataset, batch_size=8, shuffle=False, num_workers=4)
        
#Hyperparameters for model
width = 50
depth = 2
l2 = 0.00001
dropout_p = 0.1
learning_rate = 0.01
weight_decay = 1e-6
batch_size = 8
n_epochs = 5
weight_init = init.xavier_uniform_

#define model
model = Model(input_size=7*100,
              hidden_width=500,
              n_hidden_layers = 1,
              output_size=7,
              dropout_p = 0.1).to(device)


Initializer.initialize(model=model, initialization=weight_init)

#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, amsgrad=True)
loss_criterion = torch.nn.MSELoss()

#training
train_loss = []
n_epochs = 10

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
    print('Epoch: {}'.format(epoch))
    print('train loss: {}'.format(np.mean(batchloss_train)))
    
#save trained model
with open('../results/trained_model.pt','w') as f:
    torch.save(model, f)