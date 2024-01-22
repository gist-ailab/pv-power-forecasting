import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = './dataset/SolarDB'
    
    for i in range(1, 17):
        pp = f'pp{i}'
        print('='*30)
        print(pp)
    
        meta = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_meta.csv'), sep=';')
        meta_columns = meta.columns.tolist()
        power = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_power.csv'), sep=';')
        power_columns = power.columns.tolist()
        weather = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_weather.csv'), sep=';')
        weather_columns = weather.columns.tolist()
        exogenous = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_exogenous.csv'), sep=';')
        exogenous_columns = exogenous.columns.tolist()
        
        # print(meta)
        print(len(meta))
        # print(meta['location'][0])
        # print(power['dt'].head())
        # print(power['dt'].tail())

        print()