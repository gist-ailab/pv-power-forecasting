import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
import pvlib

def combine_into_each_invertor(invertor_name, index_of_invertor,
                               save_dir, log_file_path, raw_df):
    os.makedirs(save_dir, exist_ok=True)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    tenth_largest_value = raw_df[invertor_name].nlargest(10).iloc[-1]
    capacity = tenth_largest_value

    '''1. Extract only necessary columns'''
    df = raw_df[['timestamp', invertor_name, 'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Wind_Speed']]
    df = df.rename(columns={invertor_name: 'Active_Power'})

    '''2. Convert Active Power values less than 0.001 to 0'''
    df['Active_Power'] = df['Active_Power'].abs()
    
    df.loc[df['Active_Power'] < 0.001, 'Active_Power'] = 0

    '''Calculate correlation between Active_Power and GHR'''
    correlation = df[['Active_Power', 'Global_Horizontal_Radiation']].corr().iloc[0, 1]
    print(f'{invertor_name} - Correlation with GHR: {correlation}')

    # **Skip saving if the correlation is below 0.9**
    # if correlation < 0.9:
    #     print(f'Skipping {invertor_name} due to low correlation with GHR.')
    #     return  # Skip this inverter

    '''3. Drop days where any column has 2 consecutive NaN values'''
    # Replace empty strings or spaces with NaN
    df.replace(to_replace=["", " ", "  "], value=np.nan, inplace=True)
    # Detect days with 2 consecutive NaNs
    consecutive_nan_mask = detect_consecutive_nans(df, max_consecutive=2)
    days_with_2_nan = df[consecutive_nan_mask]['timestamp'].dt.date.unique()
    df_cleaned = df[~df['timestamp'].dt.date.isin(days_with_2_nan)]
    # Interpolate up to 1 consecutive missing values
    df_cleaned_3 = df_cleaned.copy()
    numeric_cols = df_cleaned_3.select_dtypes(include=[float, int]).columns
    df_cleaned_3[numeric_cols] = df_cleaned_3[numeric_cols].interpolate(method='polynomial', limit=1, order=2)
    df_cleaned_3['Active_Power'] = df_cleaned_3['Active_Power'].where(df_cleaned_3['Active_Power'] >= 0, 0) # interpolate 이후 음수 발생시 0으로 대체

    '''4. AP 값이 있지만 GHR이 없는 날 제거'''
    # Step 1: AP > 0 and GHR = 0
    # rows_to_exclude = (df_cleaned_3['Active_Power'] > 0) & (df_cleaned_3['Global_Horizontal_Radiation'] == 0)
    rows_to_exclude = ((df_cleaned_3['Active_Power'] > 0) & (df_cleaned_3['Global_Horizontal_Radiation'] == 0)) | (df_cleaned_3['Global_Horizontal_Radiation'] > 2000)| ((df_cleaned_3['Active_Power']/capacity < 0.01/2) & (df_cleaned_3['Global_Horizontal_Radiation'] > 100)) | ((df_cleaned_3['Active_Power']/capacity < 0.2/2) & (df_cleaned_3['Global_Horizontal_Radiation'] > 500)) | (df_cleaned_3['Wind_Speed']>15) | (df_cleaned_3['Global_Horizontal_Radiation'] < 0)

    # Step 2: Find the days (dates) where the conditions are true
    days_to_exclude = df_cleaned_3[rows_to_exclude]['timestamp'].dt.date.unique()

    # Step 3: Exclude entire days where any row meets the conditions
    df_cleaned_4 = df_cleaned_3[~df_cleaned_3['timestamp'].dt.date.isin(days_to_exclude)]

    '''5. Proceed without further filtering'''
    df_cleaned_5 = df_cleaned_4

    '''6. Resample data hourly and adjust margins'''
    df_hourly = df_cleaned_5
    df_hourly = df_hourly.dropna(how='all', subset=df.columns[1:])
    df_cleaned_6 = df_hourly.groupby(df_hourly['timestamp'].dt.date).apply(adjust_daily_margin)
    df_cleaned_6 = df_cleaned_6.reset_index(drop=True)

    total_dates = df_cleaned_6['timestamp'].dt.date.nunique()
    print(total_dates)

    df_cleaned_6.to_csv(os.path.join(save_dir, f'{invertor_name}.csv'), index=False)


def merge_raw_data(active_power_path, env_path, irrad_path, meter_path):

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
        'ambient_temperature_o_149575',   # Weather_Temperature_Fahrenheit
        'wind_speed_o_149576',
        'poa_irradiance_o_149574',        # POA_Irradiance
    ]
    # Add inverter columns (inv1 ~ inv24)
    for i in range(1, 25):
        if i != 15:
            inv_col = f'inv_{i:02d}_ac_power_inv_{149583 + (i-1)*5}'
        else:
            # Special case ('inv_15_ac_power_iinv_149653')
            inv_col = 'inv_15_ac_power_iinv_149653'
        columns_to_keep.append(inv_col)

    # Keep only the relevant columns
    df_filtered = df_merged[columns_to_keep]
    # Rename columns
    rename_dict = {
        'measured_on': 'timestamp',
        'ambient_temperature_o_149575': 'Weather_Temperature_Fahrenheit',
        'wind_speed_o_149576': 'Wind_Speed',
        'poa_irradiance_o_149574': 'POA_Irradiance',  # Renamed for clarity
    }

    # Rename inverter columns
    for i in range(1, 25):
        if i != 15:
            old_name = f'inv_{i:02d}_ac_power_inv_{149583 + (i-1)*5}'
        else:
            old_name = 'inv_15_ac_power_iinv_149653'
        rename_dict[old_name] = f'inv{i}'

    df_filtered = df_filtered.rename(columns=rename_dict)

    # Data preprocessing and per-inverter processing
    # Convert timestamp to datetime format
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # Resample data at 1-hour intervals
    df_filtered.set_index('timestamp', inplace=True)
    df_resampled = df_filtered.resample('1H').mean()
    df_resampled.reset_index(inplace=True)
    combined_data = df_resampled

    # Convert Fahrenheit to Celsius
    combined_data['Weather_Temperature_Celsius'] = (combined_data['Weather_Temperature_Fahrenheit'] - 32) * 5 / 9
    combined_data.drop('Weather_Temperature_Fahrenheit', axis=1, inplace=True)

    # Calculate Global Horizontal Radiation (GHI) from POA Irradiance using pvlib
    # Set arbitrary values for latitude, longitude, timezone, tilt, and azimuth
    lat, lon = 38.996306, -122.134111  # Example coordinates
    tz = 'America/Los_Angeles'  # Pacific Time Zone
    tilt = 25  # degrees (arbitrary value)
    azimuth = 180  # degrees (south-facing)

    # Create a pvlib Location object
    site = pvlib.location.Location(lat, lon, tz=tz)

    # Get solar position data
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    combined_data = combined_data.set_index('timestamp')

    # Localize the index to make it tz-aware
    combined_data.index = combined_data.index.tz_localize(tz, nonexistent='shift_forward', ambiguous='NaT')
    times = combined_data.index  # Now both combined_data and times are tz-aware

    solar_position = site.get_solarposition(times)

    # Calculate Angle of Incidence (AOI)
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )

    # Estimate DNI and DHI from POA and AOI
    poa_global = combined_data['POA_Irradiance']
    # Use the isotropic sky model to estimate the diffuse component
    poa_diffuse = pvlib.irradiance.isotropic(tilt, poa_global)
    dhi = poa_diffuse
    # Avoid division by zero by replacing zeros in cos(aoi) with a small number
    cos_aoi = np.cos(np.radians(aoi))
    cos_aoi = cos_aoi.replace(0, 1e-6)
    dni = (poa_global - dhi) / cos_aoi

    # Compute GHI
    cos_zenith = np.cos(np.radians(solar_position['apparent_zenith']))
    cos_zenith = cos_zenith.replace(0, 1e-6)  # Avoid division by zero
    ghi = dni * cos_zenith + dhi

    # Replace 'Global_Horizontal_Radiation' with the calculated GHI
    combined_data['Global_Horizontal_Radiation'] = ghi
    combined_data['Wind_Speed'] *= 0.44704 # mph to m/s

    # Reset index
    combined_data.reset_index(inplace=True)
    combined_data['timestamp'] = combined_data['timestamp'].dt.tz_convert(None)

    return combined_data


def adjust_daily_margin(group):
    # Find indices where Active Power is greater than 0
    non_zero_power = group['Active_Power'] > 0

    if non_zero_power.any():
        # First and last occurrence of non-zero Active Power
        first_non_zero_idx = non_zero_power.idxmax()
        last_non_zero_idx = non_zero_power[::-1].idxmax()

        # Calculate start and end timestamps with 1-hour margin
        start_time = group.loc[first_non_zero_idx, 'timestamp'] - pd.Timedelta(hours=1)
        end_time = group.loc[last_non_zero_idx, 'timestamp'] + pd.Timedelta(hours=1)

        # Return data within this time window
        return group[(group['timestamp'] >= start_time) & (group['timestamp'] <= end_time)]
    else:
        # If all AP values are 0, return the entire day's data
        return group

# Detect consecutive NaN values in any column
def detect_consecutive_nans(df, max_consecutive=4):
    """
    Detect rows where any column has max_consecutive or more NaN values.
    Returns a boolean mask.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        # Boolean mask for NaN values
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

    active_power_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_electrical_data.csv') 
    env_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_environment_data.csv')
    irrad_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_irradiance_data.csv')
    meter_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_meter_15m_data.csv')
    # active_power_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/2107(Arbuckle_California)/2107_electrical_data.csv')
    # env_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/2107(Arbuckle_California)/2107_environment_data.csv')
    # irrad_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/2107(Arbuckle_California)/2107_irradiance_data.csv')
    # meter_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/2107(Arbuckle_California)/2107_meter_15m_data.csv')
    merged_data = merge_raw_data(active_power_path, env_path, irrad_path, meter_path)

    invertor_list = [f'inv{i}' for i in range(1, 25)]

    log_file_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107(Arbuckle_California)/log.txt')
    # log_file_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/2107(Arbuckle_California)/log.txt')
    for i, invertor_name in enumerate(invertor_list):
        combine_into_each_invertor(
            invertor_name,
            i,
            # save_dir=os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/2107(Arbuckle_California)/preprocessed'),
            save_dir=os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/preprocessed'),
            log_file_path=log_file_path,
            raw_df=merged_data
        )
