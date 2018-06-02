import os
import zipfile
import pandas as pd
import numpy as np

for filename in os.listdir('../data/z24zipped/'): #permanent zipped files
    stem = filename.replace('.zip','')
    
    archive = zipfile.ZipFile('../data/z24zipped/'+filename, 'r')
    
    df_list = []
    for end in ['03','05','06', '07', '12', '14', '16']: # skip sensor 10
        df = pd.read_csv(archive.open(stem+end+'.aaa'), sep=' ', nrows=65536, skiprows=2)
        df.columns = [end]    
        df_list.append(df)
    data = pd.concat(df_list, axis=1).as_matrix()
    
    env = pd.read_csv(archive.open(stem+'PRE.env'), delim_whitespace=True, nrows=9, header=None, skiprows=1)
    env_mean_matrix = env.mean().as_matrix()
    
    np.save(file='../data/z24_clean/'+stem+'_env', arr=env_mean_matrix)
    np.save(file='../data/z24_clean/'+stem+'_vibrations', arr=data)
