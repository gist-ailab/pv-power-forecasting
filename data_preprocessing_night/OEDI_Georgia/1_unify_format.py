import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

from tqdm import tqdm
from copy import deepcopy
# 현재 파일에서 두 단계 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

# 이제 상위 폴더의 상위 폴더 내부의 utils 폴더의 파일 import 가능
from utils import plot_correlation_each, check_data

def combine_into_each_invertor(invertor_name, index_of_invertor,
                           save_dir, log_file_path, raw_df):
    os.makedirs(save_dir, exist_ok=True)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    # print(raw_df.head())

    '''1. Extract only necessary columns'''
    df = raw_df[['timestamp',invertor_name, 'Global_Horizontal_Radiation','Weather_Temperature_Celsius', 'Wind_Speed']]
    df = df.rename(columns={invertor_name: 'Active_Power'})

    df.to_csv(os.path.join(save_dir, f'{invertor_name}.csv'), index=False)


def merge_raw_data(active_power_path, env_path, irrad_path, meter_path):
    
    active_power = pd.read_csv(active_power_path)
    env = pd.read_csv(env_path)
    irrad = pd.read_csv(irrad_path)
    meter = pd.read_csv(meter_path)

    df_list = [active_power, env, irrad, meter]
    df_merged = df_list[0]

    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on='measured_on', how='left')
    columns_to_keep = [
        'measured_on',
        'weather_station_01_ambient_temperature_(sensor_1)_(c)_o_150245',   # Weather_Temperature_Celsius
        'pyranometer_(class_a)_12_ghi_irradiance_(w/m2)_o_150231',  # Global_Horizontal_Radiation
        'wind_sensor_12b_wind_speed_(m/s)_o_150271'
    ]
    # 인버터 열 추가 (inv1 ~ inv40)
    for i in range(1, 41):
        inv_col = f'inverter_{i:02d}_ac_power_(kw)_inv_{150952 + i}'
        columns_to_keep.append(inv_col)

    # df_merged에서 해당 열들만 남기기
    df_filtered = df_merged[columns_to_keep]
    # 열 이름 변경
    mydic = {
        'measured_on': 'timestamp',
        'weather_station_01_ambient_temperature_(sensor_1)_(c)_o_150245': 'Weather_Temperature_Celsius',
        'pyranometer_(class_a)_12_ghi_irradiance_(w/m2)_o_150231': 'Global_Horizontal_Radiation',
        'wind_sensor_12b_wind_speed_(m/s)_o_150271': 'Wind_Speed'
    }

    # 인버터 열 이름 변경 추가
    for i in range(1, 41):
        old_name = f'inverter_{i:02d}_ac_power_(kw)_inv_{150952 + i}'
        new_name = f'inv{i}'
        mydic[old_name] = new_name

    df_filtered.rename(columns=mydic, inplace=True)

    # 데이터 전처리 및 인버터별 처리
    # timestamp를 datetime 형식으로 변환
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # 데이터를 1시간 간격으로 리샘플링
    df_filtered.set_index('timestamp', inplace=True)
    df_resampled = df_filtered.resample('1H').mean()
    df_resampled.reset_index(inplace=True)
    # df_resampled['Active_Power'] = df_resampled['Active_Power']
    combined_data = df_resampled

    return combined_data


if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    # active_power_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_electrical_ac.csv')
    # env_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_environment_data.csv')
    # irrad_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_irradiance_data.csv')
    # meter_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_meter_data.csv')
    active_power_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_electrical_ac.csv')
    env_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_environment_data.csv')
    irrad_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_irradiance_data.csv')
    meter_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_meter_data.csv')
    merged_data = merge_raw_data(active_power_path, env_path, irrad_path, meter_path)

    # site_list = ['YMCA', 'Maple Drive East', 'Forest Road', 'Elm Crescent','Easthill Road']
    invertor_list = [f'inv{i}' for i in range(1,41)]

    # log_file_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/log.txt')
    for i, invertor_name in enumerate(invertor_list):
        combine_into_each_invertor(
            invertor_name, 
            i, 
            save_dir=os.path.join(project_root, 'data/OEDI/9069(Georgia)/preprocessed'),
            raw_df= merged_data
        )
    
    check_data.process_data_and_log(
    folder_path=os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/uniform_format_data'),
    log_file_path=os.path.join(project_root, 'data_preprocessing_night/OEDI_California/raw_info/raw_data_info.txt')
    )
    plot_correlation_each.plot_feature_vs_active_power(
            data_dir=os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/uniform_format_data'), 
            save_dir=os.path.join(project_root, 'data_preprocessing_night/OEDI_California/raw_info'), 
            features = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Wind_Speed', 'POA_Irradiance'],
            colors = ['blue', 'green', 'purple', 'yellow'],
            titles = ['Active Power [kW] vs Global Horizontal Radiation [w/m²]',
          'Active Power [kW] vs Weather Temperature [℃]',
          'Active Power [kW] vs Wind Speed [m/s]',
          'Active Power [kW] vs POA [w/m²]']
            )