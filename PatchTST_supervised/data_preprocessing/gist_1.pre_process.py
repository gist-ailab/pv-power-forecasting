import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm

def wrapup(file_list,
           weather_data,
           save_name):
    df = pd.DataFrame(columns=['date', 'time', 'Active_Power', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity'])
    
    weather_info = pd.read_csv(weather_data, encoding='unicode_escape')
    weather_info.columns = ['spot', 'spot_name', 'date', 'temperature', 'wind_direction', 'precipitation', 'humidity']
    # print(weather_info)
    
    pv_columns = ['time', 'radiation_horizontal', 'temperature_outdoor', 'radiation_incline', 'temperature_module',
                  '1_soccer-field_cumulative_power', '1_soccer-field_hourly_power',
                  '2_student-union(W6)_cumulative_power', '2_student-union(W6)_hourly_power',
                  '3_center-warehouse(W13)_cumulative_power', '3_center-warehouse(W13)_hourly_power',
                  '4_dormA(E11)_cumulative_power', '4_dormA(E11)_hourly_power',
                  '5_dasan(C9)_cumulative_power', '5_dasan(C9)_hourly_power',
                  '6_sisuldong(W11)_cumulative_power', '6_sisuldong(W11)_hourly_power',
                  '7_univC(N6)_cumulative_power', '7_univC(N6)_hourly_power',
                  '8_animal-exp(E2)_cumulative_power', '8_animal-exp(E2)_hourly_power',
                  '9_main-library(N1)_cumulative_power', '9_main-library(N1)_hourly_power',
                  '10_LG-library(N2)_cumulative_power', '10_LG-library(N2)_hourly_power',
                  '11_renewable-energy(C10)_cumulative_power', '11_renewable-energy(C10)_hourly_power',
                  '12_samsung-env(C7)_cumulative_power', '12_samsung-env(C7)_hourly_power',
                  '13_GAIA(C11)_cumulative_power', '13_GAIA(C11)_hourly_power',
                  '14_GTI(E3)_cumulative_power', '14_GTI(E3)_hourly_power',
                  '15_dormB(E12)_cumulative_power', '15_dormB(E12)_hourly_power',
                  '16_physics(E8)_cumulative_power', '16_physics(E8)_hourly_power',
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
        if pv_date == '2022-12-13': continue
                
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
        df.loc[24*i:24*(i+1)-1,'timestamp']                     = [pv_date + ' ' + str(i).zfill(2) + ':00' for i in range(24)]
        df.loc[24*i:24*(i+1)-1,'date']                          = pv_date
        df.loc[24*i:24*(i+1)-1,'time']                          = [str(i).zfill(2) + ':00' for i in range(24)]
        df.loc[24*i:24*(i+1)-1,'Active_Power']                  = pv_info.iloc[5:29]['6_sisuldong_hourly_power'].values
        df.loc[24*i:24*(i+1)-1,'Weather_Temperature_Celsius']   = pv_info.iloc[5:29]['temperature_outdoor'].values
        df.loc[24*i:24*(i+1)-1,'Global_Horizontal_Radiation']   = pv_info.iloc[5:29]['radiation_horizontal'].values
        # df.loc[24*i:24*(i+1)-1,'Diffuse_Horizontal_Radiation']  = pv_info.iloc[5:29]['radiation_incline'].values
        df.loc[24*i:24*(i+1)-1,'Weather_Relative_Humidity']     = weather_data['humidity'].values
            
    
    df.drop(['date', 'time'], axis=1, inplace=True)
    
    for i, column in enumerate(df.columns):
        if column == 'timestamp': continue
        missing_indices = df[df[column] == '-'].index.tolist()
        df.loc[missing_indices, column] = np.nan
        
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].interpolate()
    
    with open(save_name, 'w') as f:
        df.to_csv(f, index=False)


def create_combined_weather_csv(create_path, project_root):
    weather_data_dir = os.path.join(project_root, 'data/GIST_dataset/weather')
    weather_csv_files = [f for f in os.listdir(weather_data_dir) if f.endswith('.csv')]
    weather_csv_files.sort()

    data_frames = []
    for file in weather_csv_files:
        file_path = os.path.join(weather_data_dir, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8', skiprows=1, header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1', skiprows=1, header=None)

        # df = df.reindex(columns=expected_columns)   # Reindex the DataFrame to ensure consistent columns
        data_frames.append(df)
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Define the column names
    column_names = ['spot', 'spot_name', 'date', 'temperature', 'wind_direction', 'precipitation', 'humidity']

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(create_path, index=False, header=column_names)

def check_new_columns(pv_file_list):
    previous_elements = set()

    for i, file in tqdm(enumerate(pv_file_list), total=len(pv_file_list), desc='Checking for added new columns'):
        df = pd.read_excel(file, engine='xlrd')
        third_row = df.iloc[2]
        if i == 0:
            previous_elements = set(third_row.dropna().tolist())
            continue
        else:
            current_elements = set(third_row.dropna().tolist())
            new_elements = current_elements - previous_elements
            if new_elements == set():
                continue
            else:
                print(f'New elements in {file}: {new_elements}')
        previous_elements = current_elements

def conver_excel_to_csv(file_list):
    for i, xls_file in tqdm(enumerate(file_list), total=len(file_list), desc='Converting Excel to CSV'):
        df = pd.read_excel(xls_file, engine='xlrd')
        df = df.drop([0, 1])
        save_name = xls_file.split('/')[-1].replace('xls', 'csv')
        save_name = '.'.join(save_name.split('.')[1:])
        save_dir = xls_file.split('/')[:-2]
        save_dir.append('daily_PV_csv')
        save_dir = '/'.join(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        df.to_csv(save_path, index=False)
    print('Conversion completed!')
        # pv_file_list[0].split('/')[-1].split('.')[1]





        

if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    pv_xls_data_dir = os.path.join(project_root, 'data/GIST_dataset/daily_PV_xls')
    pv_file_list = [os.path.join(pv_xls_data_dir, _) for _ in os.listdir(pv_xls_data_dir)]
    pv_file_list.sort()

    # Define the path to save the combined CSV file
    weather_data = os.path.join(project_root, 'data/GIST_dataset/weather_data.csv')
    if not os.path.exists(weather_data):
        create_combined_weather_csv(weather_data, project_root)

    # Check for new columns in the PV data
    # check_new_columns(pv_file_list)

    conver_excel_to_csv(pv_file_list)

    pv_csv_data_dir = os.path.join(project_root, 'data/GIST_dataset/daily_PV_csv')
    pv_file_list = [os.path.join(pv_csv_data_dir, _) for _ in os.listdir(pv_csv_data_dir)]
    pv_file_list.sort()



    wrapup(pv_file_list,
           weather_data,
           'dataset/GIST/sisuldong.csv')
    


