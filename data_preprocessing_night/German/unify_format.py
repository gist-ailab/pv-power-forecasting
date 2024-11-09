import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm

def merge_raw_data(active_power_path, temp_path, ghi_path, moisture_path, wind_path, invertor_list):
    active_powers = pd.read_csv(active_power_path)
    temp = pd.read_csv(temp_path, sep=";", na_values='-999')
    ghi = pd.read_csv(ghi_path, sep=";", na_values='-999')
    moisture = pd.read_csv(moisture_path, sep=";", na_values='-99.9')
    wind = pd.read_csv(wind_path, sep=";", na_values='-999')

    active_powers['utc_timestamp'] = pd.to_datetime(active_powers['utc_timestamp'], utc=True)
    active_powers['timestamp'] = active_powers['utc_timestamp'] + pd.Timedelta(hours=1)  # Convert to local time (UTC+1)
    active_powers['timestamp'] = active_powers['timestamp'].dt.tz_convert(None)
    active_powers = active_powers.drop(columns=['utc_timestamp'])
    temp['timestamp'] = pd.to_datetime(temp['MESS_DATUM'], format='%Y%m%d%H')
    ghi['timestamp'] = pd.to_datetime(ghi['MESS_DATUM'].str.split(":").str[0], format='%Y%m%d%H')
    moisture['timestamp'] = pd.to_datetime(moisture['MESS_DATUM'], format='%Y%m%d%H')
    wind['timestamp'] = pd.to_datetime(wind['MESS_DATUM'], format='%Y%m%d%H')

    active_powers = active_powers[['timestamp']+invertor_list]

    combined_data = ghi[['timestamp', 'FG_LBERG']].merge(
    moisture[['timestamp', 'RF_STD']], on='timestamp', how='left'
    ).merge(
    temp[['timestamp', 'TT_TU']], on='timestamp', how='left'
    ).merge(active_powers, on='timestamp', how='left').merge(
        wind[['timestamp', '   F']], on='timestamp', how='left'
    )

    rename_dict = {
    'FG_LBERG': 'Global_Horizontal_Radiation',  # Unit: J/cm^2
    'RF_STD': 'Weather_Relative_Humidity',      # Unit: %
    'TT_TU': 'Weather_Temperature_Celsius',      # Unit: tenths of degree Celsius
    '   F': 'Wind_Speed' # Unit: m/s
    }

    combined_data = combined_data.rename(columns=rename_dict)

    return combined_data

def change_unit(combined_data, invertor_list):
    combined_data = combined_data.copy()
    for invertor in invertor_list:
        combined_data[invertor] = combined_data[invertor].diff() # kWh to kW
    combined_data['Global_Horizontal_Radiation'] *= 2.78 # J/cm^2 to w/m^2 
    return combined_data

def make_unifrom_csv_files(merged_data, save_dir, invertor_list):
    os.makedirs(save_dir, exist_ok=True)

    for i, invertor_name in enumerate(invertor_list):
        df = merged_data[['timestamp', invertor_name, 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity', 'Weather_Temperature_Celsius', 'Wind_Speed']]
        df = df.rename(columns={invertor_name: 'Active_Power'})
        df.to_csv(os.path.join(save_dir, invertor_name+".csv"), index=False)

if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    save_dir=os.path.join(project_root, 'data/Germany_Household_Data/uniform_format_data')

    active_power_path = os.path.join(project_root, 'data/Germany_Household_Data/household_data_60min_singleindex(selected_column).csv')
    temp_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/air_temperature/stundenwerte_TU_02712_19710101_20231231_hist/produkt_tu_stunde_19710101_20231231_02712.txt')
    ghi_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/GHI_DHI/stundenwerte_ST_02712_row/produkt_st_stunde_19770101_20240731_02712.txt')
    moisture_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/moisture/stundenwerte_TF_02712_19520502_20231231_hist/produkt_tf_stunde_19520502_20231231_02712.txt')
    wind_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/wind/produkt_ff_stunde_19590701_20231231_02712.txt')
    invertor_list = [
    'DE_KN_industrial1_pv_1',
    'DE_KN_industrial1_pv_2',
    'DE_KN_industrial2_pv',
    'DE_KN_industrial3_pv_facade',
    'DE_KN_industrial3_pv_roof',
    'DE_KN_residential1_pv',
    'DE_KN_residential3_pv',
    'DE_KN_residential4_pv',
    'DE_KN_residential6_pv'
    ]
    merged_data = merge_raw_data(active_power_path, temp_path, ghi_path, moisture_path, wind_path, invertor_list)
    
    unit_changed_data = change_unit(merged_data, invertor_list)
    make_unifrom_csv_files(unit_changed_data, save_dir, invertor_list)

    
