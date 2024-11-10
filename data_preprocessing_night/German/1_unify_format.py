import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm


# Define functions for preprocessing
def cumulative_to_increment(df, col):
    """
    Convert cumulative readings to incremental values.
    Negative increments are set to zero.
    """
    df = df.copy()
    df[col] = df[col].diff().fillna(0)
    # Handle negative increments (e.g., due to meter resets or errors)
    df[col] = df[col].apply(lambda x: x if x >= 0 else 0)
    return df


def combine_into_each_invertor(invertor_name, index_of_invertor,
                           save_dir, raw_df):
    os.makedirs(save_dir, exist_ok=True)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])


    '''1. Extract only necessary columns'''
    df = raw_df[['timestamp', invertor_name, 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity', 'Weather_Temperature_Celsius', 'Wind_Speed']]
    df = cumulative_to_increment(df, invertor_name)
    df = df.rename(columns={invertor_name: 'Active_Power'})


    '''3. Drop days where any column has 2 consecutive NaN values'''
    # 1시간 단위 데이터이므로 연속된 2개의 Nan 값 존재 시 drop
    # Step 1: Replace empty strings or spaces with NaN
    df.replace(to_replace=["", " ", "  "], value=np.nan, inplace=True)

    # 5. 단위 조정
    df['Global_Horizontal_Radiation'] *= 2.78 # J/cm^2 to w/m^2 

    df.to_csv(os.path.join(save_dir, f'{invertor_name}.csv'), index=False)


def merge_raw_data(active_power_path, temp_path, ghi_path, moisture_path, wind_path):
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
    '   F': 'Wind_Speed'
    }

    combined_data = combined_data.rename(columns=rename_dict)

    return combined_data


# Detect 4 consecutive NaN values in any column
def detect_consecutive_nans(df, max_consecutive=4):
    """
    This function detects rows where any column has max_consecutive or more NaN values.
    It will return a boolean mask.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        # Get a boolean mask for NaN values
        is_nan = df[col].isna()
        # Rolling window to find consecutive NaNs
        nan_consecutive = is_nan.rolling(window=max_consecutive, min_periods=max_consecutive).sum() == max_consecutive
        mask[col] = nan_consecutive
    return mask.any(axis=1)


if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    active_power_path = os.path.join(project_root, 'data/Germany_Household_Data/household_data_60min_singleindex(selected_column).csv')
    temp_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/air_temperature/stundenwerte_TU_02712_19710101_20231231_hist/produkt_tu_stunde_19710101_20231231_02712.txt')
    ghi_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/GHI_DHI/stundenwerte_ST_02712_row/produkt_st_stunde_19770101_20240731_02712.txt')
    moisture_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/moisture/stundenwerte_TF_02712_19520502_20231231_hist/produkt_tf_stunde_19520502_20231231_02712.txt')
    wind_path = os.path.join(project_root, 'data/Germany_Household_Data/Konstanz_weather_data/wind/produkt_ff_stunde_19590701_20231231_02712.txt')
    merged_data = merge_raw_data(active_power_path, temp_path, ghi_path, moisture_path, wind_path)

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

    log_file_path = os.path.join(project_root, 'data/Germany_Household_Data/log.txt')
    for i, invertor_name in enumerate(invertor_list):
        combine_into_each_invertor(
            invertor_name, 
            i, 
            save_dir=os.path.join(project_root, 'data/Germany_Household_Data/uniform_format_data'),
            raw_df= merged_data
        )