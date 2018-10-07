import glob
import os
import pandas as pd
import numpy as np


train_index = np.loadtxt('../tools/training_set_index.txt',dtype=str)

N = len(train_index)

total_vibration_mean = np.zeros([N,7])
total_vibration_variance = np.zeros([N,7])

total_env = np.zeros([N,53])

for index, stem in enumerate(train_index):

    file_path_vib = '../data/z24_clean/'+stem+'_vibrations.npy'
    memmap_vib = np.memmap(file_path_vib, dtype=np.float64, mode='r', shape=(65536, 7))
    vibrations = np.array(memmap_vib[:,:])
        
    total_vibration_mean[index,:] = np.mean(vibrations, axis=0)
    total_vibration_variance[index,:] = np.var(vibrations, axis=0)
    
    ###
    file_path_env = '../data/z24_clean/'+stem+'_env.npy'
    memmap_env = np.memmap(file_path_env, dtype=np.float64, mode='r', shape=(53,))
    environmental = np.array(memmap_env[:])
    
    total_env[index,:] = environmental
    print(index)
    
final_vibration_mean = np.mean(total_vibration_mean, axis=0)
final_vibration_variance = np.mean(total_vibration_variance, axis=0)
final_vibration_std = np.sqrt(final_vibration_variance)

final_env_mean = np.mean(total_env, axis=0)
final_env_variance = np.var(total_env, axis=0)
final_env_std = np.sqrt(final_env_variance)

np.save(file='../tools/vibration_mean', arr=final_vibration_mean)
np.save(file='../tools/vibration_std', arr=final_vibration_std)
np.save(file='../tools/env_mean', arr=final_env_mean)
np.save(file='../tools/env_std', arr=final_env_std)