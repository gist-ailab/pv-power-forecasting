import os
import numpy as np
import pandas as pd
from datetime import timedelta

from tqdm import tqdm
from copy import deepcopy


def combine_into_each_site(file_path, index_of_site,
                           save_dir, log_file_path):
    os.makedirs(save_dir, exist_ok=True)
    file_name = file_path.split('/')[-1]
    raw_df = pd.read_csv(file_path, encoding='unicode_escape')
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])


    '''1. Delete unnecessary columns'''
    unnecessary_columns = ['Active_Energy_Delivered_Received',
                           'Current_Phase_Average',
                           'Performance_Ratio',
                           'Wind_Speed',
                           'Wind_Direction',
                           'Weather_Daily_Rainfall',
                           'Radiation_Global_Tilted',
                           'Radiation_Diffuse_Tilted']
    df = raw_df.drop(columns=unnecessary_columns)
    
    '''2. AP가 0.0001보다 작으면 0으로 변환'''
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


    '''5. 해가 떠 있는 시간 동안의 데이터만 추출하며 상대습도가 100이상인 날은 제거'''
    # 날짜별로 그룹화
    grouped = df_cleaned_4.groupby(df_cleaned_4['timestamp'].dt.date)

    # 결과를 저장할 새로운 데이터프레임
    df_cleaned_5 = pd.DataFrame()

    count_date = 0
    for date, group in tqdm(grouped, desc=f'Processing {file_name}'):
        # Active Power가 0이 아닌 첫 번째 행 찾기
        first_non_zero = group[group['Active_Power'] != 0].first_valid_index()
        if first_non_zero is not None:
            start_time = group.loc[first_non_zero, 'timestamp']
            start_time_rounded = start_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            
            # Active Power가 0이 되는 마지막 행 찾기
            last_non_zero = group[group['Active_Power'] != 0].last_valid_index()
            if last_non_zero is not None:
                end_time = group.loc[last_non_zero, 'timestamp']
                # 종료 시간 설정 (마지막 기록 시간의 다음 시간대 끝까지)
                end_time_rounded = (end_time + timedelta(hours=1)).replace(minute=55, second=0, microsecond=0)
                
                # 시작 시간과 종료 시간 사이의 데이터만 선택
                day_data = group[(group['timestamp'] >= start_time_rounded) & 
                                (group['timestamp'] <= end_time_rounded)]
                
                # Weather_Relative_Humidity가 100 이상인 값이 있는지 확인
                if (day_data['Weather_Relative_Humidity'] >= 100).any():
                    count_date += 1
                    continue  # 100 이상인 값이 있으면 이 날의 데이터를 건너뜁니다.
                
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

    raw_csv_data_dir = os.path.join(project_root, 'data/DKASC_AliceSprings/sample')
    raw_file_list = [os.path.join(raw_csv_data_dir, _) for _ in os.listdir(raw_csv_data_dir)]
    raw_file_list.sort()

    log_file_path = os.path.join(project_root, 'data/DKASC_AliceSprings/log.txt')
    for i, file_path in enumerate(raw_file_list):
        combine_into_each_site(file_path, i,
                               os.path.join(project_root, 'data/DKASC_AliceSprings/preprocessed'),
                               log_file_path)