import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = './dataset/DKASC_cpr'
    save_path = './dataset/DKASC_cprn'
    data_path_list = os.listdir(root_path)
    data_path_list.sort()
    print()
    
    for i, data_path in enumerate(data_path_list):
        # data_path = '0-96-Site_DKA-MasterMeter1.csv'
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        print('-'*5, data_path, '-'*5)
        
        df_list = df_raw.columns.to_list()
        num_all = len(df_raw)
        
        for j, column in enumerate(df_list):
            num_null = df_raw[column].isnull().sum()
            print(column.ljust(40), round(num_null / num_all * 100, 2), '%', end='')
            
            
            
            
            
            
            
            num_null = df_raw[column].isnull().sum()
            print(' -> ', round(num_null / num_all * 100, 2), '%')
            if num_null != 0:
                print('*** ', data_path, column, num_null)
                continue
            
        
        print()
        
        df.to_csv(os.path.join(save_path, data_path), index=False)
        