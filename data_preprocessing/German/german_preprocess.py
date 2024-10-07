import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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

def remove_negative_values(df, cols):
    """
    Set negative values in specified columns to zero.
    """
    df = df.copy()
    for col in cols:
        df.loc[df[col] < 0, col] = 0
    return df

def remove_invalid_days_with_nulls(df, cols_to_check, max_allowed_consecutive_nulls=2):
    """
    Remove days where any of the specified columns have more than
    'max_allowed_consecutive_nulls' consecutive missing values.
    """
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    valid_dates = []
    grouped = df.groupby('date')
    for date, group in grouped:
        is_valid = True
        for col in cols_to_check:
            if col in group.columns:
                # Check for consecutive nulls
                is_null = group[col].isnull()
                consecutive_nulls = is_null.astype(int).groupby((is_null != is_null.shift()).cumsum()).cumsum()
                max_consecutive_nulls = consecutive_nulls.max()
                if max_consecutive_nulls > max_allowed_consecutive_nulls:
                    is_valid = False
                    break
        if is_valid:
            valid_dates.append(date)
    df = df[df['date'].isin(valid_dates)]
    df = df.drop(columns=['date'])
    return df

def remove_invalid_days(df, pv_column, temperature_column):
    """
    Remove days where the PV output is zero throughout or the temperature is negative.
    """
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    grouped = df.groupby('date')
    valid_dates = []
    for date, group in grouped:
        AP = group[pv_column]
        WTC = group[temperature_column]
        if AP.sum() == 0 or AP.max() == AP.min() or WTC.min() < 0:
            continue  # Skip this day
        else:
            valid_dates.append(date)
    df = df[df['date'].isin(valid_dates)]
    df = df.drop(columns=['date'])
    return df

# Load the cumulative PV data
df_pv = pd.read_csv("PV/Germany_Household_Data/household_data_60min_singleindex(selected_column).csv")

# List of PV columns
pv_columns = [
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

# Convert 'utc_timestamp' to local time and remove timezone information
df_pv['utc_timestamp'] = pd.to_datetime(df_pv['utc_timestamp'], utc=True)
df_pv['timestamp'] = df_pv['utc_timestamp'] + pd.Timedelta(hours=1)  # Convert to local time (UTC+1)
df_pv['timestamp'] = df_pv['timestamp'].dt.tz_convert(None)
df_pv = df_pv.drop(columns=['utc_timestamp'])

# Load weather data
# Temperature
temp = pd.read_csv(
    "PV/Germany_Household_Data/Konstanz_weather_data/air_temperature/stundenwerte_TU_02712_19710101_20231231_hist/produkt_tu_stunde_19710101_20231231_02712.txt",
    sep=";",
    na_values='-999'
)
temp['timestamp'] = pd.to_datetime(temp['MESS_DATUM'], format='%Y%m%d%H')

# Global Horizontal Irradiance
ghi = pd.read_csv(
    "PV/Germany_Household_Data/Konstanz_weather_data/GHI_DHI/stundenwerte_ST_02712_row/produkt_st_stunde_19770101_20240731_02712.txt",
    sep=";",
    na_values='-999'
)
ghi['timestamp'] = pd.to_datetime(ghi['MESS_DATUM'].str.split(":").str[0], format='%Y%m%d%H')

# Relative Humidity
moisture = pd.read_csv(
    "PV/Germany_Household_Data/Konstanz_weather_data/moisture/stundenwerte_TF_02712_19520502_20231231_hist/produkt_tf_stunde_19520502_20231231_02712.txt",
    sep=";",
    na_values='-999'
)
moisture['timestamp'] = pd.to_datetime(moisture['MESS_DATUM'], format='%Y%m%d%H')

# Merge weather data
weather_data = ghi[['timestamp', 'FG_LBERG']].merge(
    moisture[['timestamp', 'RF_STD']], on='timestamp', how='left'
).merge(
    temp[['timestamp', 'TT_TU']], on='timestamp', how='left'
)

# Rename columns for consistency
rename_dict = {
    'FG_LBERG': 'Global_Horizontal_Radiation',  # Unit: J/cm^2
    'RF_STD': 'Weather_Relative_Humidity',      # Unit: %
    'TT_TU': 'Weather_Temperature_Celsius'      # Unit: tenths of degree Celsius
}
weather_data = weather_data.rename(columns=rename_dict)

# Convert 'Weather_Temperature_Celsius' from tenths of degrees to degrees Celsius
weather_data['Weather_Temperature_Celsius'] = weather_data['Weather_Temperature_Celsius'] / 10.0

# Create directory for visualizations
vis_path = 'visualize'
os.makedirs(vis_path, exist_ok=True)

# Process each PV column separately
for pv_column in pv_columns:
    print(f"Processing PV site: {pv_column}")
    
    # Copy PV data for the current site
    df_pv_site = df_pv[['timestamp', pv_column]].copy()
    
    # Convert cumulative PV readings to incremental values
    df_pv_site = cumulative_to_increment(df_pv_site, pv_column)
    
    # Merge PV data with weather data
    df_merged = df_pv_site.merge(weather_data, on='timestamp', how='left')
    
    # Select relevant columns
    df_processed = df_merged[['timestamp', pv_column, 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity', 'Weather_Temperature_Celsius']]
    
    # Handle negative values
    df_processed = remove_negative_values(df_processed, [pv_column, 'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity'])
    
    # Handle missing values
    print("처리 전 결측값 개수:", df_processed.isnull().sum().sum())
    
    # Remove days with more than 2 consecutive nulls in any column
    cols_to_check = [pv_column, 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity', 'Weather_Temperature_Celsius']
    df_processed = remove_invalid_days_with_nulls(df_processed, cols_to_check, max_allowed_consecutive_nulls=2)
    
    print("결측값 제거 후 남은 결측값 개수:", df_processed.isnull().sum().sum())
    
    # Set 'timestamp' as index before interpolation
    df_processed = df_processed.set_index('timestamp')
    
    # Interpolate missing values
    df_processed = df_processed.interpolate(method='time')
    print("보간 후 결측값 개수:", df_processed.isnull().sum().sum())
    
    # Reset index to bring 'timestamp' back as a column
    df_processed = df_processed.reset_index()
    
    # Remove invalid days based on PV output and temperature
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    df_processed = remove_invalid_days(df_processed, pv_column, 'Weather_Temperature_Celsius')
    
    # Reset index after all removals
    df_processed = df_processed.reset_index(drop=True)
    
    # Print the number of data after preprocessing
    print("전처리 후 총 데이터 수:", len(df_processed))
    
    # Create directory for the current PV site's visualizations
    site_vis_path = os.path.join(vis_path, pv_column)
    os.makedirs(site_vis_path, exist_ok=True)
    
    # Visualize data for inspection
    def visualize_data(df, pv_column, site_vis_path):
        cnt = 0
        for date, group in df.groupby(df['timestamp'].dt.date):
            AP = group[pv_column]
            WTC = group['Weather_Temperature_Celsius']
            GHR = group['Global_Horizontal_Radiation']
            if AP.sum() == 0 or AP.max() == AP.min() or WTC.min() < 0:
                continue  # Skip invalid days
            if cnt % 90 == 0:
                hour = group['timestamp'].dt.hour
                plt.figure(figsize=(10, 5))
                plt.title(f'Date: {date} - Site: {pv_column}')
                plt.plot(hour, (GHR - GHR.min()) / (GHR.max() - GHR.min() + 1e-8), label='GHR')
                plt.plot(hour, (AP - AP.min()) / (AP.max() - AP.min() + 1e-8), label='PV Output')
                plt.plot(hour, (WTC - WTC.min()) / (WTC.max() - WTC.min() + 1e-8), label='Temperature')
                plt.xlabel('Hour')
                plt.ylabel('Normalized Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                # Save the plot
                plot_filename = os.path.join(site_vis_path, f'plot_{date}.png')
                plt.savefig(plot_filename)
                plt.close()
            cnt += 1
    
    # Visualize data and save plots
    visualize_data(df_processed, pv_column, site_vis_path)
    
    df_processed['Active_Power'] = df_processed[pv_column]
    # Save preprocessed data
    output_filename = f'preprocessed_data_{pv_column}.csv'
    df_processed.to_csv(output_filename, index=False)
    print(f"전처리된 데이터가 '{output_filename}'에 저장되었습니다.\n")
