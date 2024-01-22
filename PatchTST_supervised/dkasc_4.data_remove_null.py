import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    root_path = './dataset/DKASC_cpr'
    save_path = './dataset/DKASC_cprn'
    data_path_list = os.listdir(root_path)
    data_path_list.sort()
    print()
    
    data_path_list = ['1A-91-Site_DKA-M9_B-Phase.csv',
                      '1B-87-Site_DKA-M9_A+C-Phases.csv',
                      '25-212-Site_DKA-M15_C-Phase_II.csv',
                      ]
    
    for i, data_path in enumerate(data_path_list):
        start_time = time.time()
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        print('-'*5, data_path, '-'*5)
        
        ## interpolate missing values
        df_interpolate = df_raw.copy()
        print(df_interpolate.isnull().sum().sum(), end='')
        df_interpolate = df_interpolate.interpolate()
        print('  ->  ', df_interpolate.isnull().sum().sum())
        
        ## visualize by day
        df = pd.DataFrame(columns=df_raw.columns)
        vis_path = f'visualize/{data_path[:-4]}'
        os.makedirs(vis_path, exist_ok=True)
        j, cnt = 0, 0
        while True:
            date = df_interpolate.iloc[j]['timestamp'].split(' ')[0]
            first_idx = df_interpolate[(f'{date} 00:00:00' <= df_interpolate['timestamp'])].index[0]
            last_idx = df_interpolate[(df_interpolate['timestamp'] <= f'{date} 23:00:00')].index[-1]
            
            AP = df_interpolate.loc[first_idx:last_idx, 'Active_Power']
            WTC = df_interpolate.loc[first_idx:last_idx, 'Weather_Temperature_Celsius']
            if AP.sum() == 0            \
            or AP.max() == AP.min()     \
            or WTC.min() < 0:
                j += (last_idx - first_idx + 1)
                if j >= len(df_interpolate)-1 : break
                continue
            df = pd.concat([df, df_interpolate.loc[first_idx:last_idx, :]], ignore_index=True)
            
            if cnt % 90 == 0:
                ## draw GHR-AP graph
                hour = df_interpolate.loc[first_idx:last_idx, 'timestamp'].apply(lambda x: x.split(' ')[1].split(':')[0])
                GHR = df_interpolate.loc[first_idx:last_idx, 'Global_Horizontal_Radiation']
                
                plt.clf()
                plt.title(f'{date}')
                plt.plot(hour, (GHR - GHR.min())/(GHR.max() - GHR.min()), label='GHR')
                plt.plot(hour, (AP - AP.min())/(AP.max() - AP.min()), label='AP')
                plt.legend()
                plt.savefig(f'{vis_path}/GHR-AP_{date}.png')
                
                
                ## draw all feature graph
                AEDR = df_interpolate.loc[first_idx:last_idx, 'Active_Energy_Delivered_Received']
                CPA = df_interpolate.loc[first_idx:last_idx, 'Current_Phase_Average']
                WRH = df_interpolate.loc[first_idx:last_idx, 'Weather_Relative_Humidity']
                DHR = df_interpolate.loc[first_idx:last_idx, 'Diffuse_Horizontal_Radiation']
                WD = df_interpolate.loc[first_idx:last_idx, 'Wind_Direction']
                WDR = df_interpolate.loc[first_idx:last_idx, 'Weather_Daily_Rainfall']
                
                plt.plot(hour, (AEDR - AEDR.min())/(AEDR.max() - AEDR.min()), label='AEDR')
                plt.plot(hour, (CPA - CPA.min())/(CPA.max() - CPA.min()), label='CPA')
                plt.plot(hour, (WTC - WTC.min())/(WTC.max() - WTC.min()), label='WTC')
                plt.plot(hour, (WRH - WRH.min())/(WRH.max() - WRH.min()), label='WRH')
                plt.plot(hour, (DHR - DHR.min())/(DHR.max() - DHR.min()), label='DHR')
                plt.plot(hour, (WD - WD.min())/(WD.max() - WD.min()), label='WD')
                if WDR.max() == 0: plt.plot(hour, WDR, label='WDR')
                else: plt.plot(hour, (WDR - WDR.min())/(WDR.max() - WDR.min()), label='WDR')
                plt.legend()
                plt.savefig(f'{vis_path}/all_{date}.png')
            
            j += (last_idx - first_idx + 1)
            if j >= len(df_interpolate)-1 : break
            cnt += 1
        
        ## clip humidity in 0 ~ 100
        idx_100 = df[df['Weather_Relative_Humidity'] > 100].index.tolist()
        idx_0 = df[df['Weather_Relative_Humidity'] < 0].index.tolist()
        df.loc[idx_100, 'Weather_Relative_Humidity'] = 100
        df.loc[idx_0, 'Weather_Relative_Humidity'] = 0
        
        ## save pre-processed dataframe
        df.to_csv(os.path.join(save_path, data_path), index=False)
        
        finish_time = time.time()
        print('pre-processing time:', round(finish_time - start_time, 2), 'sec')
        print()
        
        