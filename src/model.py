import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class z24Dataset(Dataset):
    def __init__(self, window_size=100, lookahead=1):
        self.window_size = window_size
        self.lookahead = lookahead
        self.index_file = np.loadtxt('../data/name_to_index.txt',dtype=str)
        self.name_index_dict = dict(zip(range(len(self.index_file)),list(self.index_file)))
        self.slices_per_file = 65536 // self.window_size
        
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
        
        return X_vibration_window.flatten(), Y
    

# Neural Network Model

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p, weight_decay):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
    
    def forward(self, x):
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.fc2(x)
        
        return x

    
#Hyperparameters for model
width = 100
depth = 1
l2 = 0.01
dropout_p = 0.1
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9
batch_size = 32
n_epochs = 10

#define model
model = Model(input_size=8*100, 
              hidden_size=50, 
              output_size=8, 
              dropout_p = dropout_p, 
              weight_decay = weight_decay)

optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum)

loss_criterion = torch.nn.MSELoss()

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#Training 
train_loss = []

for epoch in range(n_epochs):
    print(f'Epoch {epoch}/{n_epochs}')
    batchloss_train = []
    
    for X_train, Y_train in dataloader:  
        X_train_tensor = X_train.float() 
        Y_train_tensor = Y_train.float()

        optimizer.zero_grad()  # zero the gradient buffer
        predicted_Y = model(X_train_tensor)
        trainloss = loss_criterion(predicted_Y, Y_train_tensor)
        
        trainloss.backward()
        optimizer.step()
    
        batchloss_train.append(trainloss.detach().numpy())
        
    #scheduler.step()
    train_loss.append(np.mean(batchloss_train))
    
#save trained model
with open('trained_model','w') as f
    torch.save(model, f)