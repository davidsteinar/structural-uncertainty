import glob
import os
import pandas as pd
import numpy as np
from random import shuffle
random.seed(1992)

files = glob.glob("../data/z24zipped/*")
files.sort(key=os.path.getmtime)
sorted_filenames = files

final = []

for name in sorted_filenames:
    temp = name.replace('../data/z24zipped/','')
    temp = temp.replace('.zip','')
    final.append(temp)
    
shuffle(final)
train_split = int(0.8 * len(final))
train_index = final[:train_split]
rest = final[train_split:]

val_split = int(0.5 * len(rest))
test_index = rest[:val_split]
validation_index = rest[val_split:]

   
with open('../tools/training_set_index.txt', 'w') as file:
    file.write('\n'.join(train_index))
    
with open('../tools/test_set_index.txt', 'w') as file:
    file.write('\n'.join(test_index))
    
with open('../tools/validation_set_index.txt', 'w') as file:
    file.write('\n'.join(validation_index))