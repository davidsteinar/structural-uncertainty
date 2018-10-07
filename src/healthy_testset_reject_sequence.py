import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from shallow_model import Model

import numpy as np

class z24Dataset_sequence(Dataset):
    def __init__(self, filename = '40C07', window_size=100, normalize=True):
        self.window_size = window_size
        self.slices_per_file = 65536 // self.window_size
        self.normalize = normalize
        self.filename = filename
            
        self.env_mean = np.load('../tools/env_mean.npy')
        self.env_std = np.load('../tools/env_std.npy')
        self.vibration_mean = np.load('../tools/vibration_mean.npy')
        self.vibration_std = np.load('../tools/vibration_std.npy')

    def __len__(self):
        return self.slices_per_file

    def __getitem__(self, index):
        index_to_read = index // self.slices_per_file
        index_in_dataframe = (index - index_to_read*self.slices_per_file) * self.window_size
        
        file_path_vib = '../data/z24_clean/'+self.filename+'_vibrations.npy'
        file_path_env = '../data/z24_clean/'+self.filename+'_env.npy'
        
        memmap_vib = np.memmap(file_path_vib, dtype=np.float64, mode='r', shape=(65536, 7))
        memmap_env = np.memmap(file_path_env, dtype=np.float64, mode='r', shape=(53,))

        X_environmental = np.array(memmap_env[:])
        X_vibration_window = np.array(memmap_vib[index_in_dataframe:index_in_dataframe+self.window_size,:])

        if self.normalize:
            X_vibration_window = (X_vibration_window - self.vibration_mean) / self.vibration_std
            X_environmental = (X_environmental - self.env_mean) / self.env_std
        
        X_vib_and_env = np.append(X_vibration_window.flatten(),X_environmental)
       
        return X_vib_and_env, X_vibration_window.flatten()


batch_size = 100
window_size = 200

device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
model = torch.load(f='../results/trained_shallow_relu200_100epochs.pt')#, map_location=device_string)

loss_criterion = torch.nn.MSELoss(reduce=False)
file_index = np.loadtxt('../tools/test_set_index.txt',dtype=str)


total = []
for filename in file_index:
    print(filename)
    dataset = z24Dataset_sequence(filename=filename, window_size=window_size, normalize=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    sequence_window_loss = []
    for X, Y in dataloader:
        X_tensor = X.float().to(device)
        Y_tensor = Y.float()#.to(device)
    
        batch_size, output_size = Y.shape
        N = 100
        N_predictions = torch.zeros([N, batch_size, output_size])
    
        for i in range(N):
            N_predictions[i,:,:] = model(X_tensor)
    
        prediction_mean = torch.mean(N_predictions, dim=0)
        prediction_std  = torch.std(N_predictions, dim=0)
    
        loss_full = loss_criterion(prediction_mean, Y_tensor)
    
        lower_y = prediction_mean - 2*prediction_std
        upper_y = prediction_mean + 2*prediction_std
        within_lower = Y_tensor > lower_y
        within_upper = Y_tensor < upper_y
        within_range = within_lower & within_upper
    
        loss_full[within_range] = 0
    
        for j in range(batch_size):
            window_loss = torch.sum(loss_full[j,:]) / torch.numel(loss_full[j,:])
            sequence_window_loss.append(window_loss.item())
    total.append(sequence_window_loss)
    
total_array = np.array(total)
np.save(file='../results/total_testsequence_window_losses', arr=total_array)