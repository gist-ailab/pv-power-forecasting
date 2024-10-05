import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = 'dataset/DKASC_cp'
    save_path = 'dataset/DKASC_cpr'
    data_path_list = os.listdir(root_path)
    data_path_list.sort()
    data_path_list = data_path_list[20:30]        ##
    
    remove_path_list = os.listdir(save_path)
    for remove_path in remove_path_list:
        data_path_list.remove(remove_path)
        
    print()
    
    for i, data_path in enumerate(data_path_list):
        # data_path = '0-96-Site_DKA-MasterMeter1.csv'
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        print('-'*5, data_path, '-'*5)
        
        df_list = df_raw.columns.to_list()
        num_all = len(df_raw)
        
        for j, column in enumerate(df_list):
            num_null = df_raw[column].isnull().sum()
            
            print(column.ljust(40), round(num_null / num_all * 100, 2), '%')
        
        df = pd.DataFrame(columns=df_list)
        j = 0
        while True:
            date = df_raw.iloc[j]['timestamp'].split(' ')[0]
            hour = df_raw.iloc[j]['timestamp'].split(' ')[1].split(':')[0]
            first_idx = df_raw[(f'{date} {hour}:00:00' <= df_raw['timestamp'])].index[0]
            last_idx = df_raw[(df_raw['timestamp'] <= f'{date} {hour}:55:00')].index[-1]
            new_row = [f'{date} {hour}:00:00']
            
            small_df = df_raw.loc[first_idx:last_idx, :].drop(columns=['timestamp'])
            new_row.extend(small_df.mean().to_list())
            df.loc[j] = new_row
            
            j += (last_idx - first_idx +1)
            if j >= len(df_raw)-1: break
            
            
            
        df.to_csv(os.path.join(save_path, data_path), index=False)
        