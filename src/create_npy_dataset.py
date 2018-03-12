import os

for filename in os.listdir('../data/z24/permanent_zipped/'):
    stem = filename.replace('.zip','')
    
    archive = zipfile.ZipFile('../data/z24/permanent_zipped/'+filename, 'r')
    
    df_list = []
    for end in ['03','05','06', '07', '10', '12', '14', '16']:
        df = pd.read_csv(archive.open(stem+end+'.aaa'), sep=' ', nrows=65536, skiprows=2)
        df.columns = [end]    
        df_list.append(df)
    data = pd.concat(df_list, axis=1).as_matrix()
    
    np.save(file=stem, arr=data)