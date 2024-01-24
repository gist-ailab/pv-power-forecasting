import os
import numpy as np
import pandas as pd
import warnings
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = './dataset/SolarDB/pre-process'
    save_path = './dataset/SolarDB/pre-process'
    
    for i in range(1, 17):
        if i != 8: continue
        pp = f'pp{i}'
        print('='*30)
        vis_path = f'visualize/SolarDB/{pp}'
        os.makedirs(vis_path, exist_ok=True)

        week = pd.read_csv(os.path.join(root_path, f'{pp}_week.csv'))
        print(pp, 'null', week.isnull().sum().sum())

        start_date = week.loc[0, 'timestamp'].split(' ')[0]
        year, month, day = list(map(int, start_date.split('-')))
        date = datetime.datetime(year, month, day)
        
        for j in range(4):
            _date = f'{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}'
            
            first_idx = week[(f'{_date} 00:00:00' <= week['timestamp'])].index[0]
            last_idx  = week[(f'{_date} 23:00:00' >= week['timestamp'])].index[-1]

            _hour    = week.loc[first_idx:last_idx, 'timestamp'].apply(lambda x: x.split(' ')[1].split(':')[0])
            _irrad   = week.loc[first_idx:last_idx, 'sun_irradiance']
            _pv      = week.loc[first_idx:last_idx, 'power_ac']
            _temp    = week.loc[first_idx:last_idx, 'temp']
            _humidity= week.loc[first_idx:last_idx, 'humidity']

            plt.clf()
            plt.title(f'{_date}')
            plt.plot(_hour, (_irrad - _irrad.min())/(_irrad.max() - _irrad.min()), label='sun_irradiance')
            plt.plot(_hour, (_pv - _pv.min())/(_pv.max() - _pv.min()), label='power_ac')
            plt.legend()
            plt.savefig(f'{vis_path}/Irrad-PV_{_date}.png')
            
            plt.plot(_hour, (_temp - _temp.min())/(_temp.max() - _temp.min()), label='temperature')
            plt.plot(_hour, (_humidity - _humidity.min())/(_humidity.max() - _humidity.min()), label='humidity')
            plt.legend()
            plt.savefig(f'{vis_path}/all_{_date}.png')

            date += datetime.timedelta(days=2)











