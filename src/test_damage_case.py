import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from scipy import stats

from shallow_model import Model

class z24Dataset(Dataset):
    def __init__(self, damage_case='1', filename = '40C07',window_size=100, normalize=True):
        self.window_size = window_size
        self.slices_per_file = 65536 // self.window_size
        self.normalize = normalize
        self.damage_case = damage_case
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
        
        file_path_vib = '../data/z24_damage/'+self.damage_case+'/'+self.filename+'_vibrations.npy'
        file_path_env = '../data/z24_damage/'+self.damage_case+'/'+self.filename+'_env.npy'

        memmap_vib = np.memmap(file_path_vib, dtype=np.float64, mode='r', shape=(65536, 7))
        memmap_env = np.memmap(file_path_env, dtype=np.float64, mode='r', shape=(53,))

        X_environmental = np.array(memmap_env[:])
        X_vibration_window = np.array(memmap_vib[index_in_dataframe:index_in_dataframe+self.window_size,:])

        if self.normalize:
            X_vibration_window = (X_vibration_window - self.vibration_mean) / self.vibration_std
            X_environmental = (X_environmental - self.env_mean) / self.env_std
        
        X_vib_and_env = np.append(X_vibration_window.flatten(),X_environmental)
       
        return X_vib_and_env, X_vibration_window.flatten()


damage_case = '1'

for damage_case in ['healthy','1','2','3','4','5','6','7','8','9','10','11','12','13','14']:
    file_index = np.loadtxt('../data/z24_damage/damage_'+damage_case+'_index.txt',dtype=str)
    w_size = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f='../results/trained_shallow_relu200_100epochs.pt', map_location='cpu')
    loss_criterion = torch.nn.MSELoss(reduce=False)
    
    classification_list = []
    for damage_file in file_index:
        dataset = z24Dataset(damage_case=damage_case, filename = damage_file, window_size=w_size, normalize=True)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    
        #loss_criterion = torch.nn.MSELoss(reduce=False)
        
        all_window_loss = []
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
        
            lower_y = prediction_mean - 2*prediction_std
            upper_y = prediction_mean + 2*prediction_std
            within_lower = Y_tensor > lower_y
            within_upper = Y_tensor < upper_y
            within_range = within_lower & within_upper
        
            loss_full[within_range] = 0
        
            for j in range(batch_size):
                window_loss = torch.sum(loss_full[j,:]) / torch.numel(loss_full[j,:])
                all_window_loss.append(window_loss.item())
                
        all_window_loss = np.array(all_window_loss)
        all_window_loss[all_window_loss > 0.33731913566589355] = 0.33731913566589355
        
        s = np.sum(all_window_loss)
    
        classification = 42.65411979705095 < s
    
        classification_list.append(classification)
        #print('Damage case {}, file {}, classified as {}, total error {}'.format(damage_case, damage_file, classification, np.round(s)))
        print(s)
    
    n_correct = np.sum(classification_list)
    print('Damage case: {}'.format(damage_case))
    print('Number of cases: {}'.format(len(classification_list)))
    print('Number of correct classifications: {}'.format(int(n_correct)))
    print('Ratio {}'.format(n_correct/len(classification_list)))