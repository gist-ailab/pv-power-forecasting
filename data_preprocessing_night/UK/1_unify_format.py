import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm

def merge_raw_data(active_power_path, weather_path, site_list):
    
    weather_raw = pd.read_csv(weather_path, na_values='---')
    active_powers = pd.read_csv(active_power_path)

    # 열 이름 변경
    weather_rename_dict = {
        'TempOut': 'Weather_Temperature_Celsius',
        'SolarRad': 'Global_Horizontal_Radiation',
        'OutHum': 'Weather_Relative_Humidity',
        'WindSpeed': 'Wind_Speed',
        'Date': 'Date',
        'Time': 'Time',
        'Site': 'Site'
    }
    weather = weather_raw.rename(columns=weather_rename_dict)
    # 필요한 열만 선택
    weather_filtered = weather[['Site', 'Date', 'Time', 'Weather_Temperature_Celsius',
    'Wind_Speed', 
    'Global_Horizontal_Radiation', 'Weather_Relative_Humidity']].copy()
    # 데이터 타입 변환
    weather_filtered['Weather_Temperature_Celsius'] = pd.to_numeric(weather_filtered['Weather_Temperature_Celsius'], errors='coerce')
    weather_filtered['Global_Horizontal_Radiation'] = pd.to_numeric(weather_filtered['Global_Horizontal_Radiation'], errors='coerce')
    weather_filtered['Weather_Relative_Humidity'] = pd.to_numeric(weather_filtered['Weather_Relative_Humidity'], errors='coerce')
    weather_filtered['Wind_Speed'] = pd.to_numeric(weather_filtered['Wind_Speed'], errors='coerce')

    # 타임스탬프 생성
    weather_filtered['timestamp'] = pd.to_datetime(weather_filtered['Date'] + ' ' + weather_filtered['Time'])

    # 필요 없는 열 제거
    weather_filtered = weather_filtered[['Site', 'timestamp', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity',
    'Wind_Speed']]

    # 2. 날씨 데이터 리샘플링 (1시간 간격)
    weather_resampled = weather_filtered.groupby('Site').resample('1H', on='timestamp').mean().reset_index()

    
    # Active_Power 계산
    active_powers['Active_Power'] = \
    (active_powers['V_MIN_Filtered'] + active_powers['V_MAX_Filtered']) / 2 * \
    (active_powers['I_GEN_MIN_Filtered'] + active_powers['I_GEN_MAX_Filtered']) / 2
    # active_powers['Active_Power'] = active_powers['V_MIN_Filtered'] * active_powers['I_GEN_MIN_Filtered']


    # 열 이름 변경
    active_powers.rename(columns={'datetime': 'timestamp', 'Substation': 'Site'}, inplace=True)
    active_powers = active_powers[['Site', 'timestamp', 'Active_Power']]
    # 타임스탬프 형식 변환 (중요)
    active_powers['timestamp'] = pd.to_datetime(active_powers['timestamp'])
    weather_resampled['timestamp'] = pd.to_datetime(weather_resampled['timestamp'])  # 추가된 부분

    combined_data = pd.merge(active_powers, weather_resampled, on=['Site', 'timestamp'], how='left')

    return combined_data

def change_unit(combined_data, invertor_list):
    combined_data = combined_data.copy()
    # for invertor in invertor_list:
    #     combined_data[invertor] = combined_data[invertor].diff() # kWh to kW
    # combined_data['Global_Horizontal_Radiation'] *= 2.78 # J/cm^2 to w/m^2 
    # nothing to change
    combined_data['Active_Power'] /= 1000 # W to kW
    return combined_data

def make_unifrom_csv_files(merged_data, save_dir, invertor_list):
    os.makedirs(save_dir, exist_ok=True)

    for i, invertor_name in enumerate(invertor_list):
        # df = merged_data[['timestamp', invertor_name, 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity', 'Weather_Temperature_Celsius', 'Wind_Speed']]
        df = merged_data[merged_data['Site'] == invertor_name]
        df = df[['timestamp', 'Active_Power', 'Global_Horizontal_Radiation','Weather_Temperature_Celsius', 'Wind_Speed','Weather_Relative_Humidity']]
        df.to_csv(os.path.join(save_dir, invertor_name+".csv"), index=False)

if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    save_dir=os.path.join(project_root, 'data/UK_data/uniform_format_data')

    active_power_path = os.path.join(project_root, 'data/UK_data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv')
    # active_power_path = os.path.join(project_root, '/ailab_mat/dataset/PV/UK_data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv')
    weather_path = os.path.join(project_root, 'data/UK_data/Weather_Data_2014-11-30.csv')
    # weather_path = os.path.join(project_root, '/ailab_mat/dataset/PV/UK_data/Weather_Data_2014-11-30.csv')
    site_list = ['YMCA', 'Maple Drive East', 'Forest Road']
    merged_data = merge_raw_data(active_power_path, weather_path, site_list)
    active_power_path = os.path.join(project_root, 'data/UK_data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv')

    unit_changed_data = change_unit(merged_data, site_list)
    make_unifrom_csv_files(unit_changed_data, save_dir, site_list)

    
