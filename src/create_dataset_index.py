import glob
import os
import pandas as pd


files = glob.glob("../data/z24/permanent/*")
files.sort(key=os.path.getmtime)
sorted_filenames = files

my_set = set()
res = []
for filename in sorted_filenames:
    stem = filename[22:27]
    if stem not in my_set:
        res.append(stem)
        my_set.add(stem)
        
with open('name_to_index.txt', 'w') as file:
    file.write('\n'.join(res))