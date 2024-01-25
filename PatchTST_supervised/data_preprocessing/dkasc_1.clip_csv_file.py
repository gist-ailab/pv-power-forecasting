import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = './dataset/DKASC'
    save_path = './dataset/DKASC_c'
    data_path_list = os.listdir(root_path)
    data_path_list.sort()
    
    for i, data_path in enumerate(data_path_list):
        # data_path = '0-96-Site_DKA-MasterMeter1.csv'
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
    
        df_list = df_raw.columns.to_list()
        
        start_idx, last_idx = 0, len(df_raw)-1
        
        start_year = df_raw['timestamp'][start_idx].split('-')[0]
        last_year = df_raw['timestamp'][last_idx].split('-')[0]

        if df_raw['timestamp'][start_idx] != f'{start_year}-01-01 00:00:00': start_year = int(start_year)+1
        if df_raw['timestamp'][last_idx] != f'{last_year}-12-31 23:00:00': last_year = int(last_year)-1
            
        start_idx = df_raw[df_raw['timestamp'] == f'{start_year}-01-01 00:00:00'].index[0]
        last_idx = df_raw[df_raw['timestamp'] == f'{last_year}-12-31 23:55:00'].index[0]
        
        df = df_raw.loc[start_idx:last_idx, :]
        
        df = df.reset_index(drop=True)
        
        df.to_csv(os.path.join(save_path, data_path), index=False)
        
        print(data_path, start_year, last_year)
        
    
    