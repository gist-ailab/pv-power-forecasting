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
        # data_path = '0-96-Site_DKA-MasterMeter1.csv'
        start_time = time.time()
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        print('-'*5, data_path, '-'*5)
        
        df_list = df_raw.columns.to_list()
        num_all = len(df_raw)
        
        df_ffill = pd.DataFrame(columns=df_raw.columns)
        # df_bfill = pd.DataFrame(columns=df_raw.columns)
        
        for j, column in enumerate(df_list):
            num_null = df_raw[column].isnull().sum()
            print(column.ljust(40), round(num_null / num_all * 100, 2), '%', end='')
            
            df_ffill[column] = df_raw[column].ffill()
            # df_bfill[column] = df_raw[column].bfill()
            
            
            print('ffill -> ', round(df_ffill[column].isnull().sum() / num_all * 100, 2), '%', df_ffill[column].isnull().sum())
            # print('bfill -> ', round(df_bfill[column].isnull().sum() / num_all * 100, 2), '%')
        
        df = pd.DataFrame(columns=df_raw.columns)
        vis_path = f'visualize/{data_path[:-4]}'
        os.makedirs(vis_path, exist_ok=True)
        j, cnt = 0, 0
        while True:
            date = df_ffill.iloc[j]['timestamp'].split(' ')[0]
            first_idx = df_ffill[(f'{date} 00:00:00' <= df_ffill['timestamp'])].index[0]
            last_idx = df_ffill[(df_ffill['timestamp'] <= f'{date} 23:00:00')].index[-1]
            
            pv = df_ffill.loc[first_idx:last_idx, 'Active_Power']
            if pv.sum() == 0 or pv.max() == pv.min(): 
                j += (last_idx - first_idx + 1)
                if j >= len(df_ffill)-1 : break
                continue
            df = pd.concat([df, df_ffill.loc[first_idx:last_idx, :]], ignore_index=True)
            
            hour = df_ffill.loc[first_idx:last_idx, 'timestamp'].apply(lambda x: x.split(' ')[1].split(':')[0])
            irrad = df_ffill.loc[first_idx:last_idx, 'Global_Horizontal_Radiation']
            
            if cnt % 90 == 0:
                plt.clf()
                plt.title(f'{date}')
                plt.plot(hour, (irrad - irrad.min())/(irrad.max() - irrad.min()), label='irrad')
                plt.plot(hour, (pv - pv.min())/(pv.max() - pv.min()), label='pv')
                plt.legend()
                plt.savefig(f'{vis_path}/irrad-pv_{date}.png')
                
                AEDR = df_ffill.loc[first_idx:last_idx, 'Active_Energy_Delivered_Received']
                CPA = df_ffill.loc[first_idx:last_idx, 'Current_Phase_Average']
                WTC = df_ffill.loc[first_idx:last_idx, 'Weather_Temperature_Celsius']
                WRH = df_ffill.loc[first_idx:last_idx, 'Weather_Relative_Humidity']
                DHR = df_ffill.loc[first_idx:last_idx, 'Diffuse_Horizontal_Radiation']
                WD = df_ffill.loc[first_idx:last_idx, 'Wind_Direction']
                WDR = df_ffill.loc[first_idx:last_idx, 'Weather_Daily_Rainfall']
                
                plt.plot(hour, (AEDR - AEDR.min())/(AEDR.max() - AEDR.min()), label='AEDR')
                plt.plot(hour, (CPA - CPA.min())/(CPA.max() - CPA.min()), label='CPA')
                plt.plot(hour, (WTC - WTC.min())/(WTC.max() - WTC.min()), label='WTC')
                plt.plot(hour, (WRH - WRH.min())/(WRH.max() - WRH.min()), label='WRH')
                plt.plot(hour, (DHR - DHR.min())/(DHR.max() - DHR.min()), label='DHR')
                plt.plot(hour, (WD - WD.min())/(WD.max() - WD.min()), label='WD')
                plt.plot(hour, (WDR - WDR.min())/(WDR.max() - WDR.min()), label='WDR')
                plt.legend()
                plt.savefig(f'{vis_path}/all_{date}.png')
            
            j += (last_idx - first_idx + 1)
            if j >= len(df_ffill)-1 : break
            cnt += 1
            
        df.to_csv(os.path.join(save_path, data_path), index=False)
        
        finish_time = time.time()
        print('pre-processing time:', round(finish_time - start_time, 2), 'sec')
        
        