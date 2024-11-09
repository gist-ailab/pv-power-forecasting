import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm

def merge_raw_data(active_power_path, env_path, irrad_path, meter_path):
    print("Start merging raw files")
    
    active_power = pd.read_csv(active_power_path)
    env = pd.read_csv(env_path)
    irrad = pd.read_csv(irrad_path)
    meter = pd.read_csv(meter_path)

    df_list = [active_power, env, irrad, meter]
    df_merged = df_list[0]

    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on='measured_on', how='outer')
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
    print("Merging Completed")

    return combined_data

def change_unit(combined_data, invertor_list):
    combined_data = combined_data.copy()
    # for invertor in invertor_list:
    #     combined_data[invertor] = combined_data[invertor].diff() # kWh to kW
    # combined_data['Global_Horizontal_Radiation'] *= 2.78 # J/cm^2 to w/m^2 
    # nothing to change
    return combined_data

def make_unifrom_csv_files(merged_data, save_dir, invertor_list):
    print("Start making uniform format files")
    os.makedirs(save_dir, exist_ok=True)

    for invertor_name in tqdm(invertor_list, desc="Saving CSV files"):
        # df = merged_data[['timestamp', invertor_name, 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity', 'Weather_Temperature_Celsius', 'Wind_Speed']]
        df = merged_data[['timestamp',invertor_name, 'Global_Horizontal_Radiation','Weather_Temperature_Celsius', 'Wind_Speed']]
        df = df.rename(columns={invertor_name: 'Active_Power'})
        df.to_csv(os.path.join(save_dir, invertor_name+".csv"), index=False)
    print("Finished!")

if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    save_dir=os.path.join(project_root, 'data/OEDI/9069(Georgia)/uniform_format_data')

    active_power_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_electrical_ac.csv')
    env_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_environment_data.csv')
    irrad_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_irradiance_data.csv')
    meter_path = os.path.join(project_root, 'data/OEDI/9069(Georgia)/9069_meter_data.csv')
    merged_data = merge_raw_data(active_power_path, env_path, irrad_path, meter_path)
    invertor_list = [f'inv{i}' for i in range(1,41)]

    unit_changed_data = change_unit(merged_data, invertor_list)
    make_unifrom_csv_files(unit_changed_data, save_dir, invertor_list)

    
