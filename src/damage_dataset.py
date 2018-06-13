import numpy as np
import os
import torch
from scipy import stats.ks_2samp



damage_case = '7'
file_to_read = '42E07'
        
env_mean = np.load('../tools/env_mean.npy')
env_std = np.load('../tools/env_std.npy')
vibration_mean = np.load('../tools/vibration_mean.npy')
vibration_std = np.load('../tools/vibration_std.npy')

file_path_vib = '../data/z24_damage/7_clean/'+file_to_read+'_vibrations.npy'
file_path_env = '../data/z24_damage/7_clean/'+file_to_read+'_env.npy'

memmap_vib = np.memmap(file_path_vib, dtype=np.float64, mode='r', shape=(65536, 7))
memmap_env = np.memmap(file_path_env, dtype=np.float64, mode='r', shape=(53,))



model = torch.load('../results/trained_autoencoder_memmap.pt')
loss_criterion = torch.nn.MSELoss()


losses = []
for i in range(65536 // 100):
    
    X_environmental = np.array(memmap_env[:])
    X_vibration_window = np.array(memmap_vib[i:i+100,:])

    X_vibration_window = (X_vibration_window - vibration_mean) / vibration_std
    X_environmental = (X_environmental - env_mean) / env_std

    X_vib_and_env = np.append(X_vibration_window.flatten(),X_environmental)
    
    
    X_train_tensor = torch.tensor(X_vib_and_env).float()
    Y_train_tensor = torch.tensor(X_vibration_window).float()
    predicted_Y = model(X_train_tensor)
    trainloss = loss_criterion(predicted_Y, Y_train_tensor)
    losses.append(trainloss.item())
