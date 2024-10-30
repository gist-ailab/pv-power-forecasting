import os
import numpy as np
import pandas as pd
from datetime import timedelta

from tqdm import tqdm
from copy import deepcopy

def combine_humidity_data(raw_weather_list, save_path):
    # 모든 데이터프레임을 저장할 리스트 초기화
    humidity_dfs = []
    
    # wind speed 파일 리스트를 순회하면서 데이터프레임으로 읽어옵니다.
    for weather_file in raw_weather_list:
        df_weather = pd.read_csv(weather_file)
        df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
        humidity_dfs.append(df_weather)
        
    # 데이터프레임들을 하나로 합칩니다.
    combined_weather_df = pd.concat(humidity_dfs, ignore_index=True)
    combined_weather_df.sort_values('timestamp', inplace=True)
    combined_weather_df = combined_weather_df[['timestamp', 'Weather_Relative_Humidity']]
    combined_weather_df['Weather_Relative_Humidity'] = pd.to_numeric(combined_weather_df['Weather_Relative_Humidity'], errors='coerce')
    combined_weather_df['Weather_Relative_Humidity'] = combined_weather_df['Weather_Relative_Humidity'] * (1000 / 3600) # km/h to m/s
    
    # 1시간 단위로 리샘플링하여 평균을 계산합니다.
    combined_weather_df.set_index('timestamp', inplace=True)
    combined_weather_hourly = combined_weather_df.resample('h').mean().reset_index()
    
    # 결과를 CSV 파일로 저장합니다.
    combined_weather_hourly.to_csv(save_path, index=False)

def combine_into_each_site(file_path, index_of_site,
                           save_dir, log_file_path, combined_weather_hourly):
    os.makedirs(save_dir, exist_ok=True)
    file_name = file_path.split('/')[-1]
    raw_df = pd.read_csv(file_path, encoding='unicode_escape')
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    init_total_dates = raw_df['timestamp'].dt.date.nunique()


    '''1. 데이터의 맨 처음과 끝을 일 단위로 끊고 불필요한 열 제거'''
    # 데이터 시작 시간을 기준으로 다음날 00:00 이후 데이터 사용
    start_date = raw_df['timestamp'].dt.normalize().iloc[0] + pd.Timedelta(days=1)
    # 데이터 종료 시간을 기준으로 당일 23:59:59 이전 데이터 사용
    end_date = raw_df['timestamp'].dt.normalize().iloc[-1]
    raw_df = raw_df[(raw_df['timestamp'] >= start_date) & (raw_df['timestamp'] < end_date)]

    necessary_columns = ['Active_Power',
                         'Global_Horizontal_Radiation',
                        #  'Diffuse_Horizontal_Radiation',
                         'Weather_Temperature_Celsius']
                        #  'Weather_Relative_Humidity'] 1시간 단위로 변환후 추가
    df = raw_df.loc[:, ['timestamp'] + necessary_columns]
    
    '''2. AP가 0.001보다 작으면 0으로 변환'''
    df['Active_Power'] = df['Active_Power'].abs()    
    df.loc[df['Active_Power'] < 0.001, 'Active_Power'] = 0

    '''3. Drop days where any column has 4 consecutive NaN values'''
    # Step 1: Replace empty strings or spaces with NaN
    df.replace(to_replace=["", " ", "  "], value=np.nan, inplace=True)
    # Step 2: Find days where any column has 4 consecutive NaN values
    consecutive_nan_mask = detect_consecutive_nans(df, max_consecutive=4)
    # Remove entire days where 4 consecutive NaNs were found
    days_with_4_nan = df[consecutive_nan_mask]['timestamp'].dt.date.unique()
    df_cleaned = df[~df['timestamp'].dt.date.isin(days_with_4_nan)]
    # Step 3: Interpolate up to 3 consecutive missing values
    df_cleaned_3 = df_cleaned.interpolate(method='linear', limit=3)

    
    '''4. AP 값이 있지만 GHR이 있는 날 제거'''
    # Step 1: AP > 0 and GHR = 0
    rows_to_exclude = (df_cleaned_3['Active_Power'] > 0) & (df_cleaned_3['Global_Horizontal_Radiation'] == 0)

    # Step 2: Find the days (dates) where the conditions are true
    days_to_exclude = df_cleaned_3[rows_to_exclude]['timestamp'].dt.date.unique()

    # Step 3: Exclude entire days where any row meets the conditions
    df_cleaned_4 = df_cleaned_3[~df_cleaned_3['timestamp'].dt.date.isin(days_to_exclude)]


    '''5. 해가 떠 있는 시간 동안의 데이터만 추출하며 상대습도가 100이상인 날과 영하 -10도 이하는 제거'''
    # 날짜별로 그룹화
    grouped = df_cleaned_4.groupby(df_cleaned_4['timestamp'].dt.date)

    # 결과를 저장할 새로운 데이터프레임
    df_cleaned_5 = pd.DataFrame()

    count_date = 0
    for date, group in tqdm(grouped, desc=f'Processing {file_name}'):
        # Step 1: AP가 0이 아니고 GHR이 6보다 큰 행 찾기
        valid_rows = (group['Active_Power'] > 0) & (group['Global_Horizontal_Radiation'] > 6)

        if valid_rows.any():
            # AP > 0 이고 GHR > 6인 첫 번째 행 찾기
            first_valid_index = valid_rows.idxmax()
            start_time = group.loc[first_valid_index, 'timestamp']
            start_time_rounded = start_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            
            # AP > 0 이고 GHR > 6인 마지막 행 찾기
            last_valid_index = valid_rows[::-1].idxmax()
            end_time = group.loc[last_valid_index, 'timestamp']
            end_time_rounded = (end_time + timedelta(hours=1)).replace(minute=55, second=0, microsecond=0)

            # Step 2: 시작 시간과 종료 시간 사이의 데이터만 선택
            day_data = group[(group['timestamp'] >= start_time_rounded) &
                            (group['timestamp'] <= end_time_rounded)]
                
            if not day_data.empty:
                # Weather_Relative_Humidity가 100 이상이고 Weather_Temperature_Celsiusrk -10 이하인 값이 있는지 확인
                if (day_data['Weather_Temperature_Celsius'] < -10).any():
                    count_date += 1
                    continue  # 조건에 맞는 날의 데이터를 건너뜁니다.
                
                # 결과 데이터프레임에 추가
                df_cleaned_5 = pd.concat([df_cleaned_5, day_data])
    
    '''6. 1시간 단위로 데이터를 sampling. Margin은 1시간으로 유지'''
    # 1. 1시간 단위로 평균 계산
    df_hourly = df_cleaned_5.resample('h', on='timestamp').mean().reset_index()
    df_hourly = df_hourly.dropna(how='all', subset=df.columns[1:])

    # 2. AP 값이 0.001보다 작은 경우 0으로 설정
    df_hourly.loc[df_hourly['Active_Power'] < 0.001, 'Active_Power'] = 0
    
    # 4. 날짜별로 그룹화하고 margin 조절
    df_cleaned_6 = df_hourly.groupby(df_hourly['timestamp'].dt.date).apply(adjust_daily_margin)
    df_cleaned_6 = df_cleaned_6.reset_index(drop=True)

    total_dates = df_cleaned_6['timestamp'].dt.date.nunique()
    print(f'Total changes in the number of dates: {init_total_dates} -> {total_dates}')

    df_cleaned_6.to_csv(os.path.join(save_dir, f'{file_name}'), index=False)


def remove_empty_rows(df):
    # 모든 열에 대해 빈 문자열을 NaN으로 변환
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    
    # 모든 열이 NaN인 행 제거
    df_cleaned = df.dropna(how='all').reset_index(drop=True)
    
    return df_cleaned

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

    raw_csv_data_dir = os.path.join(project_root, 'data/DKASC_Yulara/raw')  # for local
    # raw_csv_data_dir = '/ailab_mat/dataset/PV/DKASC_AliceSprings/raw'
    raw_file_list = [os.path.join(raw_csv_data_dir, _) for _ in os.listdir(raw_csv_data_dir)]

    raw_weather_data_dir = os.path.join(project_root, 'data/DKASC_Yulara/weather_data')  # for local
    # raw_csv_data_dir = '/ailab_mat/dataset/PV/DKASC_AliceSprings/raw'
    raw_weather_list = [os.path.join(raw_weather_data_dir, _) for _ in os.listdir(raw_weather_data_dir)]
    raw_weather_list.sort()

    # combined_weather.csv를 저장할 경로 설정
    combined_weather_path = os.path.join(project_root, 'data/DKASC_Yulara/combined_weather.csv')
    # wind speed 데이터를 합치고 저장합니다.
    combine_humidity_data(raw_weather_list, combined_weather_path)

    # combined_weather.csv를 데이터프레임으로 읽어옵니다.
    combined_weather_hourly = pd.read_csv(combined_weather_path)
    combined_weather_hourly['timestamp'] = pd.to_datetime(combined_weather_hourly['timestamp'])

    log_file_path = os.path.join(project_root, 'data/DKASC_AliceSprings/log.txt') # for local
    # log_file_path = '/ailab_mat/dataset/PV/DKASC_AliceSprings/log.txt'
    for i, file_path in enumerate(raw_file_list):
        combine_into_each_site(file_path, i,
                               os.path.join(project_root, 'data/DKASC_Yulara/converted'),  # for local
                            #    '/ailab_mat/dataset/PV/DKASC_AliceSprings/converted',
                               log_file_path, combined_weather_hourly)
        # Weather_Relative_Humidity