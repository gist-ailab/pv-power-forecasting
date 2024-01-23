import os
import time
import numpy as np
import pandas as pd
import warnings
import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = './dataset/SolarDB'
    save_path = './dataset/SolarDB/pre-process'
    
    for i in range(1, 17):
        if i != 8: continue
        start_time = time.time()
        pp = f'pp{i}'
        print('='*30)
        print(pp)
    
        meta = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_meta.csv'), sep=';')
        power = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_power.csv'), sep=';')
        weather = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_weather.csv'), sep=';')
        exogenous = pd.read_csv(os.path.join(root_path, f'{pp}/{pp}_solardb_exogenous.csv'), sep=';')
        
        num_inverters =  len(meta) -1
        power = power[power['inv_id'] == -1].reset_index(drop=True)
        # print(meta['location'][0])
        # print(power['dt'].head())
        # print(power['dt'].tail())

        df = pd.DataFrame(columns=['timestamp', 'power_ac', 'temp', 'humidity', 'sun_irradiance'])
        # weather.loc[:,['dt', 'temp', 'humidity']]
        # power.loc[:, ['dt', 'power_ac']]
        # exogenous.loc[:, ['dt', 'sun_irradiance']]

        print()

        start_date = power.loc[0, 'dt'].split(' ')[0]
        year, month, day = list(map(int, start_date.split('-')))
        date = start_date = datetime.datetime(year, month, day)

        finish_date_week  = start_date + datetime.timedelta(days=7)
        finish_date_month = start_date + relativedelta(months=3)
        finish_date_year  = start_date + relativedelta(years=1)

        df_idx = 0

        while True:
            _date = f'{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}'

            for hour in range(0, 24):
                ## timestamp
                new_row = [f'{_date} {str(hour).zfill(2)}:00:00']

                ## power_ac
                power_first_idx = power[(f'{_date} {str(hour).zfill(2)}:00:00' <= power['dt'])].index[0]
                power_last_idx  = power[(f'{_date} {str(hour).zfill(2)}:55:00' >= power['dt'])].index[-1]

                new_row.extend([power.loc[power_first_idx:power_last_idx, 'power_ac'].mean()])
                
                ## temp, humidity
                weather_first_idx = weather[(f'{_date} {str(hour).zfill(2)}:00:00' <= weather['dt'])].index[0]
                weather_last_idx  = weather[(f'{_date} {str(hour).zfill(2)}:55:00' >= weather['dt'])].index[-1]

                new_row.extend(weather.loc[weather_first_idx:weather_last_idx, ['temp', 'humidity']].mean().tolist())
                
                ## irradiance
                exogenous_first_idx = exogenous[(f'{_date} {str(hour).zfill(2)}:00:00' <= exogenous['dt'])].index[0]
                exogenous_last_idx  = exogenous[(f'{_date} {str(hour).zfill(2)}:55:00' >= exogenous['dt'])].index[-1]

                new_row.extend([exogenous.loc[exogenous_first_idx:exogenous_last_idx, 'sun_irradiance'].mean()])

                df.loc[df_idx] = new_row
                df_idx += 1
            

            ## next day
            date += datetime.timedelta(days=1)

            if date == finish_date_week:
                df.to_csv(os.path.join(save_path, f'{pp}_week.csv'), index=False)
                print('saved week')
            
            if date == finish_date_month:
                df.to_csv(os.path.join(save_path, f'{pp}_month.csv'), index=False)
                print('saved month')
            
            if date == finish_date_year:
                df.to_csv(os.path.join(save_path, f'{pp}_year.csv'), index=False)
                print('saved_year')
                break
        
        print(pp, round(time.time() - start_time, 2), ' sec')



