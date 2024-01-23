import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = './dataset/SolarDB'
    save_path = './dataset/SolarDB/pre-process'
    
    for i in range(1, 17):
        if i != 8: continue
        pp = f'pp{i}'
        print('='*30)
        print(pp)
    
        meta = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_meta.csv'), sep=';')
        power = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_power.csv'), sep=';')
        weather = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_weather.csv'), sep=';')
        exogenous = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_exogenous.csv'), sep=';')
        
        print(len(meta))
        # print(meta['location'][0])
        # print(power['dt'].head())
        # print(power['dt'].tail())

        df = pd.DataFrame(columns=['power_ac', 'temp', 'humidity', 'sun_irradiance'])
        weather.loc[:,['dt', 'temp', 'humidity']]
        power.loc[:, ['dt', 'power_ac']]
        exogenous.loc[:, ['dt', 'sun_irradiance']]

        print()

        date = start_date = power.loc[0, 'dt'].split(' ')[0]
        finish_date = power.loc[len(power)-1, 'dt']

        while True:
            start_dt = f'{date} 00:00:00'
            finish_dt = f'{date} 23:55:00'

            for hour in range(0, 24):
                power_first_idx = power[(f'{date} {str(hour).zfill(2)}:00:00' <= power['dt'])].index[0]



