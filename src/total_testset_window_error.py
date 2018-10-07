import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats

from shallow_model import Model

class z24Dataset(Dataset):
    def __init__(self, filename = '04G03',window_size=100, normalize=True):
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

w_size = 200
batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_criterion = torch.nn.MSELoss(reduce=False)

model = torch.load(f='../results/trained_shallow_relu200_100epochs.pt')#,map_location='cpu')


file_index = np.loadtxt('../tools/test_set_index.txt',dtype=str)

total_errors = []
total_mean_errors = []
for test_file in file_index:
    print(test_file)
    dataset = z24Dataset(filename = test_file, window_size=w_size, normalize=True)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)

    all_window_loss = []
    all_mean_loss = []
    for X, Y in dataloader:
        X_tensor = X.float().to(device)
        Y_tensor = Y.float()#.to(device)
    
        batch_size, output_size = Y.shape
        N = 100
        N_predictions = torch.zeros([N, batch_size, output_size])
    
        for i in range(N):
            N_predictions[i,:,:] = model(X_tensor)
    
        prediction_mean = torch.mean(N_predictions, dim=0)
        prediction_std = torch.std(N_predictions, dim=0)
    
        loss_full = loss_criterion(prediction_mean, Y_tensor)
        
        for j in range(batch_size):
            mean_loss = torch.sum(loss_full[j,:]) / torch.numel(loss_full[j,:])
            all_mean_loss.append(mean_loss.item())
    
        lower_y = prediction_mean - 2*prediction_std
        upper_y = prediction_mean + 2*prediction_std
        within_lower = Y_tensor > lower_y
        within_upper = Y_tensor < upper_y
        within_range = within_lower & within_upper
    
        loss_full[within_range] = 0
    
        for j in range(batch_size):
            window_loss = torch.sum(loss_full[j,:]) / torch.numel(loss_full[j,:])
            all_window_loss.append(window_loss.item())

    total_errors.append(all_window_loss)
    total_mean_errors.append(all_mean_loss)
    print(test_file)
    print(np.max(all_window_loss))

total_errors_np = np.array(total_errors)
total_mean_np = np.array(total_mean_errors)
np.save('../tools/total_testset_window_errors.npy', total_errors_np)
np.save('../tools/total_testset_mean_errors.npy', total_mean_np)

