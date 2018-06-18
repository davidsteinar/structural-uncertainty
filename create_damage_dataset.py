import os
import zipfile
import pandas as pd
import numpy as np
import glob

case = '9'
folder = 'damage_9/'

files = glob.glob(folder+"*")
files.sort(key=os.path.getmtime)
sorted_filenames = files

final = []

for name in sorted_filenames:
    temp = name.replace(folder,'')
    temp = temp.replace('.zip','')
    final.append(temp)

with open('damage_'+case+'_index.txt', 'w') as file:
    file.write('\n'.join(final))

for filename in os.listdir(folder): #permanent zipped files
    stem = filename.replace('.zip','')
    
    archive = zipfile.ZipFile(folder+filename, 'r')
    
    df_list = []
    for end in ['03','05','06', '07', '12', '14', '16']: # skip sensor 10
        df = pd.read_csv(archive.open(stem+end+'.aaa'), sep=' ', nrows=65536, skiprows=2)
        df.columns = [end]
        df_list.append(df)
    data = pd.concat(df_list, axis=1).as_matrix()
    
    env = pd.read_csv(archive.open(stem+'PRE.env'), delim_whitespace=True, nrows=9, header=None, skiprows=1)
    env_mean_matrix = env.mean().as_matrix()
    

    filename_vib = folder+stem+'_vibrations.npy'
    filename_env = folder+stem+'_env.npy'
    
    f_vib = np.memmap(filename_vib, dtype=np.float64, mode='w+', shape=(65536, 7))
    f_vib[:] = data
    
    f_env = np.memmap(filename_env, dtype=np.float64, mode='w+', shape=(53,))
    f_env[:] = env_mean_matrix