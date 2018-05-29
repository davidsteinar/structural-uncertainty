import glob
import os
import pandas as pd
import numpy as np
from random import shuffle


train_index = np.loadtxt('training_set_index.txt',dtype=str)

N = len(train_index)

total_vibration_mean = np.zeros([N,8])
total_vibration_variance = np.zeros([N,8])

total_env = np.zeros([N,53])

for index, stem in enumerate(train_index):
    vibrations = np.load('../data/z24_clean/'+stem+'_vibrations.npy')
        
    total_vibration_mean[index,:] = np.mean(vibrations, axis=0)
    total_vibration_variance[index,:] = np.var(vibrations, axis=0)
    
    ###
    environmental = np.load('../data/z24_clean/'+stem+'_env.npy')
    total_env[index,:] = environmental
    
final_vibration_mean = np.mean(total_vibration_mean, axis=0)
final_vibration_variance = np.mean(total_vibration_variance, axis=0)
final_vibration_std = np.sqrt(final_vibration_variance)

final_env_mean = np.mean(total_env, axis=0)
final_env_variance = np.var(total_env, axis=0)
final_env_std = np.sqrt(final_env_variance)

np.save(file='vibration_mean', arr=final_vibration_mean)
np.save(file='vibration_std', arr=final_vibration_std)
np.save(file='env_mean', arr=final_env_mean)
np.save(file='env_std', arr=final_env_std)