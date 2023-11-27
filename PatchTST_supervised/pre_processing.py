import os
import sys

import pandas as pd



def wrapup(file_list, save_name):
    df = pd.DataFrame(columns=['date', 'time', 'Active_Power', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Relative_Humidity'])
    
    weather_info = pd.read_csv('dataset/GIST_dataset/GIST_AWS_TIM_2022.csv', encoding='unicode_escape')
    weather_info.columns = ['spot', 'spot_name', 'date', 'temperature', 'wind_speed', 'precipitation', 'humidity']
    # print(weather_info)
    
    pv_columns = ['time', 'radiation_horizontal', 'temperature_outdoor', 'radiation_incline', 'temperature_module', 
                  '1_soccer-field_cumulative_power', '1_soccer-field_hourly_power',
                  '2_student-union_cumulative_power', '2_student-union_hourly_power',
                  '3_center-warehouse_cumulative_power', '3_center-warehouse_hourly_power',
                  '4_undergraduate_cumulative_power', '4_undergraduate_hourly_power',
                  '5_dasan_cumulative_power', '5_dasan_hourly_power',
                  '6_sisuldong_cumulative_power', '6_sisuldong_hourly_power',
                  '7_univC_cumulative_power', '7_univC_hourly_power',
                  '8_animal-exp_cumulative_power', '8_animal-exp_hourly_power',
                  '9_main-library_cumulative_power', '9_main-library_hourly_power',
                  '10_LG-library_cumulative_power', '10_LG-library_hourly_power',
                  '11_renewable-energy_cumulative_power', '11_renewable-energy_hourly_power',
                  '12_samsung_cumulative_power', '12_samsung_hourly_power',
                  '13_junggidong_cumulative_power', '13_junggidong_hourly_power',
                  '14_industrial-cooperative_cumulative_power', '14_industrial-cooperative_hourly_power',
                  '15_dormB_cumulative_power', '15_dormB_hourly_power',
                  'daily_load']
    empty_rows = pd.concat([pd.DataFrame(df.columns)]*24, axis=1).T
    empty_rows.columns = df.columns
    # df = pd.DataFrame(columns=['date', 'time', 'Active_Power', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Relative_Humidity'])
    
    for i, file in enumerate(file_list):
        ## read pv info
        pv_info = pd.read_excel(file)
        pv_info.columns = pv_columns
        
        ## get date
        pv_date = file.split('_')[-2]
        weather_data = weather_info[weather_info['date'].str.contains(pv_date)]
        weather_data = weather_data.reset_index(drop=True)
        # weather_info['date'].str.contains(pv_date)
        
        ## check if the time is correct
        pv_time = [_time.split(' ')[0] for _time in pv_info.iloc[5:29]['time'].values]
        if i == 0:
            weather_time = ['00']
            weather_time.extend([_time.split(' ')[1].split(':')[0] for _time in weather_data['date'].values])
            weather_data = pd.concat([pd.DataFrame({
                'date': f'{pv_date} 00:00',
                'spot': 783,
                'spot_name': 'GIST',
                'humidity': weather_data.loc[0, 'humidity'],                
            }, index=[0]), weather_data], axis=0)
            weather_data = weather_data.reset_index(drop=True)
        else:
            weather_time = [_time.split(' ')[1].split(':')[0] for _time in weather_data['date'].values]
        
        
        ## handling the missing humidity data
        if weather_data['humidity'].isnull().sum():
            missing_idx = weather_data[weather_data['humidity'].isnull()].index
            for idx in missing_idx:
                weather_data.loc[idx, 'humidity'] = weather_data.loc[idx-1, 'humidity']
        
        ## handling the missing weather data
        missing_time = list(set(pv_time) - set(weather_time))
        missing_time.sort()
        
        for m_time in missing_time:
            m_time = int(m_time)
            ## forward filling
            tmp1 = weather_data.loc[weather_data.index < m_time]
            tmp2 = weather_data.loc[weather_data.index >= m_time]
            weather_data = pd.concat([tmp1, pd.DataFrame({
                'date': f'{pv_date} {str(m_time).zfill(2)}:00',
                'spot': 783,
                'spot_name': 'GIST',
                'humidity': weather_data.loc[m_time-1, 'humidity'],                
            }, index=[m_time])], axis=0)
            weather_data = pd.concat([weather_data, tmp2], axis=0)
            weather_data = weather_data.reset_index(drop=True)
        
        
        assert len(pv_time) == len(weather_data)
        
        ## add empty rows for 24 hours
        df = pd.concat([df, empty_rows], axis=0)
        df = df.reset_index(drop=True)
        
        ## add data
        df.loc[24*i:24*(i+1)-1,'date']                          = pv_date
        df.loc[24*i:24*(i+1)-1,'time']                          = [str(i).zfill(2) + ':00' for i in range(24)]
        df.loc[24*i:24*(i+1)-1,'Active_Power']                  = pv_info.iloc[5:29]['6_sisuldong_hourly_power'].values
        df.loc[24*i:24*(i+1)-1,'Weather_Temperature_Celsius']   = pv_info.iloc[5:29]['temperature_outdoor'].values
        df.loc[24*i:24*(i+1)-1,'Global_Horizontal_Radiation']   = pv_info.iloc[5:29]['radiation_horizontal'].values
        df.loc[24*i:24*(i+1)-1,'Diffuse_Horizontal_Radiation']  = pv_info.iloc[5:29]['radiation_incline'].values
        df.loc[24*i:24*(i+1)-1,'Weather_Relative_Humidity']     = weather_data['humidity'].values
            
            
    
    with open(save_name, 'w') as f:
        df.to_csv(f, index=False)
        

if __name__ == '__main__':
    path = 'dataset/GIST_dataset/일보/'
    file_list = [path + _ for _ in os.listdir(path)]
    file_list.sort()
    # print(file_list)
    
    wrapup(file_list, 'dataset/GIST_dataset/GIST_sisuldong.csv')
    


