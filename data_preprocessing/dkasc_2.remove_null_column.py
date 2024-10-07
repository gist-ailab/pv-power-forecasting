import os
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = '/PV/DKASC_AliceSprings_1h'
    save_path = '/PV/DKASC_AliceSprings_1h'
    os.makedirs(save_path, exist_ok=True)
    data_path_list = os.listdir(root_path)
    data_path_list.sort()
    print()
    
    for i, data_path in tqdm(enumerate(data_path_list)):
        # data_path = '0-96-Site_DKA-MasterMeter1.csv'
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        print('-'*5, data_path, '-'*5)
        
        df_list = df_raw.columns.to_list()
        num_all = len(df_raw)
        
        for j, column in enumerate(df_list):
            num_null = df_raw[column].isnull().sum()
            
            print(column.ljust(40), round(num_null / num_all * 100, 2), '%')
        
        df = df_raw
        # for column in (['Performance_Ratio', 'Wind_Speed', 'Radiation_Global_Tilted', 'Radiation_Diffuse_Tilted']):
        for column in (['Performance_Ratio', 'Wind_Speed']):
            if column in df_list:
                df = df.drop(columns=[column])
        print('\n')
        col = df.pop('Active_Power')  
        df['Active_Power'] = col
        df.to_csv(os.path.join(save_path, data_path), index=False)
        

# %%
