import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm


def combine_into_each_invertor(invertor_name, index_of_invertor,
                           save_dir, log_file_path, raw_df):
    os.makedirs(save_dir, exist_ok=True)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

    '''1. Extract only necessary columns'''
    raw_df = raw_df[raw_df['Site'] == invertor_name]
    df = raw_df[['timestamp', 'Active_Power', 'Global_Horizontal_Radiation','Weather_Temperature_Celsius', 'Wind_Speed','Weather_Relative_Humidity']]
    df = df.rename(columns={invertor_name: 'Active_Power'})

    '''2. AP가 0.001보다 작으면 0으로 변환'''
    df['Active_Power'] = df['Active_Power'].abs()    
    df.loc[df['Active_Power'] < 0.1, 'Active_Power'] = 0

    '''3. Drop days where any column has 2 consecutive NaN values'''
    # 1시간 단위 데이터이므로 연속된 2개의 Nan 값 존재 시 drop
    # Step 1: Replace empty strings or spaces with NaN
    df.replace(to_replace=["", " ", "  "], value=np.nan, inplace=True)
    # Step 2: Find days where any column has 4 consecutive NaN values
    consecutive_nan_mask = detect_consecutive_nans(df, max_consecutive=2)
    # Remove entire days where 4 consecutive NaNs were found
    days_with_2_nan = df[consecutive_nan_mask]['timestamp'].dt.date.unique()
    df_cleaned = df[~df['timestamp'].dt.date.isin(days_with_2_nan)]
    # Step 3: Interpolate up to 1 consecutive missing values
    # 날짜가 포함되지 않은 숫자형 열만 선택해서 보간
    df_cleaned_3 = df_cleaned.copy()
    numeric_cols = df_cleaned_3.select_dtypes(include=[float, int]).columns
    df_cleaned_3[numeric_cols] = df_cleaned_3[numeric_cols].interpolate(method='polynomial', limit=1, order=2)


    '''4. AP 값이 있지만 GHR이 없는 날 제거'''
    # Step 1: AP > 0 and GHR = 0
    rows_to_exclude = (df_cleaned_3['Active_Power'] > 0) & (df_cleaned_3['Global_Horizontal_Radiation'] == 0)

    # Step 2: Find the days (dates) where the conditions are true
    days_to_exclude = df_cleaned_3[rows_to_exclude]['timestamp'].dt.date.unique()

    # Step 3: Exclude entire days where any row meets the conditions
    df_cleaned_4 = df_cleaned_3[~df_cleaned_3['timestamp'].dt.date.isin(days_to_exclude)]

    '''5. 해가 떠 있는 시간 동안의 데이터만 추출하며 상대습도가 100이상인 날은 제거'''
    # # 날짜별로 그룹화
    # grouped = df_cleaned_4.groupby(df_cleaned_4['timestamp'].dt.date)

    # # 결과를 저장할 새로운 데이터프레임
    # df_cleaned_5 = pd.DataFrame()

    count_date = 0
    # for date, group in tqdm(grouped, desc=f'Processing {invertor_name}'):
    #     # Active Power가 0이 아닌 첫 번째 행 찾기
    #     first_non_zero = group[group['Active_Power'] != 0].first_valid_index()
    #     if first_non_zero is not None:
    #         start_time = group.loc[first_non_zero, 'timestamp']
    #         start_time_rounded = start_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            
    #         # Active Power가 0이 되는 마지막 행 찾기
    #         last_non_zero = group[group['Active_Power'] != 0].last_valid_index()
    #         if last_non_zero is not None:
    #             end_time = group.loc[last_non_zero, 'timestamp']
    #             # 종료 시간 설정 (마지막 기록 시간의 다음 시간대 끝까지)
    #             end_time_rounded = (end_time + timedelta(hours=1)).replace(minute=55, second=0, microsecond=0)
                
    #             # 시작 시간과 종료 시간 사이의 데이터만 선택
    #             day_data = group[(group['timestamp'] >= start_time_rounded) & 
    #                             (group['timestamp'] <= end_time_rounded)]
                
    #             # Weather_Relative_Humidity가 100 이상인 값이 있는지 확인
    #             if (day_data['Weather_Relative_Humidity'] >= 100).any() or (day_data['Weather_Temperature_Celsius'] < -10).any():
    #                 count_date += 1
    #                 continue  # 100 이상인 값이 있으면 이 날의 데이터를 건너뜁니다.
                
    #             # 결과 데이터프레임에 추가
    #             df_cleaned_5 = pd.concat([df_cleaned_5, day_data])
    df_cleaned_5 = df_cleaned_4 

    '''6. 1시간 단위로 데이터를 sampling. Margin은 1시간으로 유지'''
    # 1. 1시간 단위로 평균 계산
    # df_hourly = df_cleaned_5.resample('h', on='timestamp').mean().reset_index()
    df_hourly = df_cleaned_5
    df_hourly = df_hourly.dropna(how='all', subset=df.columns[1:])

    # 2. AP 값이 0.001보다 작은 경우 0으로 설정
    df_hourly.loc[df_hourly['Active_Power'] < 0.1, 'Active_Power'] = 0
    
    # 4. 날짜별로 그룹화하고 margin 조절
    df_cleaned_6 = df_hourly.groupby(df_hourly['timestamp'].dt.date).apply(adjust_daily_margin)
    df_cleaned_6 = df_cleaned_6.reset_index(drop=True)

    total_dates = df_cleaned_6['timestamp'].dt.date.nunique()
    print(count_date, total_dates)

    df_cleaned_6.to_csv(os.path.join(save_dir, f'{invertor_name}.csv'), index=False)


def merge_raw_data(active_power_path, weather_path):
    
    weather_raw = pd.read_csv(weather_path)
    weather_raw.replace('---', np.nan, inplace=True)

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
    'Global_Horizontal_Radiation', 'Weather_Relative_Humidity']]
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

    active_powers = pd.read_csv(active_power_path)
    # Active_Power 계산
    active_powers['Active_Power'] = \
    (active_powers['V_MIN_Filtered'] + active_powers['V_MAX_Filtered']) / 2 * \
    (active_powers['I_GEN_MIN_Filtered'] + active_powers['I_GEN_MAX_Filtered']) / 2
    # active_powers['Active_Power'] = active_powers['V_MIN_Filtered'] * active_powers['I_GEN_MIN_Filtered']
    active_powers['Active_Power'] = (active_powers['Active_Power'] - active_powers['Active_Power'].min())/1000


    # 열 이름 변경
    active_powers.rename(columns={'datetime': 'timestamp', 'Substation': 'Site'}, inplace=True)
    active_powers = active_powers[['Site', 'timestamp', 'Active_Power']]
    # 타임스탬프 형식 변환 (중요)
    active_powers['timestamp'] = pd.to_datetime(active_powers['timestamp'])
    weather_resampled['timestamp'] = pd.to_datetime(weather_resampled['timestamp'])  # 추가된 부분

    combined_data = pd.merge(active_powers, weather_resampled, on=['Site', 'timestamp'], how='left')


    return combined_data

def adjust_daily_margin(group):
    # Find the indices where Active Power is greater than 0
    non_zero_power = group['Active_Power'] > 0
    
    if non_zero_power.any():
        # First and last occurrence of non-zero Active Power
        first_non_zero_idx = non_zero_power.idxmax()
        last_non_zero_idx = non_zero_power[::-1].idxmax()
        
        # Calculate the start and end timestamps with 1-hour margin
        start_time = group.loc[first_non_zero_idx, 'timestamp'] - pd.Timedelta(hours=1)
        end_time = group.loc[last_non_zero_idx, 'timestamp'] + pd.Timedelta(hours=1)
        
        # Return only the data within this time window
        return group[(group['timestamp'] >= start_time) & (group['timestamp'] <= end_time)]
    
    else:
        # If all AP values are 0, return the entire day's data (or handle as needed)
        return group

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

    active_power_path = os.path.join(project_root, 'data/UK_data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv')
    # active_power_path = os.path.join(project_root, '/ailab_mat/dataset/PV/UK_data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv')
    weather_path = os.path.join(project_root, 'data/UK_data/Weather_Data_2014-11-30.csv')
    # weather_path = os.path.join(project_root, '/ailab_mat/dataset/PV/UK_data/Weather_Data_2014-11-30.csv')
    merged_data = merge_raw_data(active_power_path, weather_path)
    active_power_path = os.path.join(project_root, 'data/UK_data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv')
    

    # site_list = ['YMCA', 'Maple Drive East', 'Forest Road', 'Elm Crescent','Easthill Road']
    # site_list = ['YMCA', 'Maple Drive East', 'Forest Road']
    site_list = ['Maple Drive East', 'Forest Road']

    log_file_path = os.path.join(project_root, '/ailab_mat/dataset/PV/UK_data/log.txt')
    for i, invertor_name in enumerate(site_list):
        combine_into_each_invertor(
            invertor_name, 
            i, 
            # save_dir=os.path.join(project_root, '/ailab_mat/dataset/PV/UK_data/preprocessed'),
            save_dir=os.path.join(project_root, 'data/UK_data/preprocessed'),
            log_file_path=log_file_path,
            raw_df= merged_data
        )