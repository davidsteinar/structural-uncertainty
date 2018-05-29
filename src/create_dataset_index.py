import glob
import os
import pandas as pd
import numpy as np
from random import shuffle


files = glob.glob("../data/z24zipped/*")
files.sort(key=os.path.getmtime)
sorted_filenames = files

final = []

for name in sorted_filenames:
    temp = name.replace('../data/z24zipped/','')
    temp = temp.replace('.zip','')
    final.append(temp)
    
shuffle(final)
split_index = int(0.8 * len(final))
train_index = final[:split_index]
test_index = final[split_index:]
   
with open('training_set_index.txt', 'w') as file:
    file.write('\n'.join(train_index))
    
with open('test_set_index.txt', 'w') as file:
    file.write('\n'.join(test_index))