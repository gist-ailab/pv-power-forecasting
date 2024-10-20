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
    df = raw_df[['timestamp',invertor_name, 'Global_Horizontal_Radiation','Weather_Temperature_Celsius']]
    df = df.rename(columns={invertor_name: 'Active_Power'})

    '''2. AP가 0.001보다 작으면 0으로 변환'''
    df['Active_Power'] = df['Active_Power'].abs()    
    df.loc[df['Active_Power'] < 0.001, 'Active_Power'] = 0

    '''Calculate correlation between Active_Power and GHR'''
    correlation = df[['Active_Power', 'Global_Horizontal_Radiation']].corr().iloc[0, 1]
    print(f'{invertor_name} - Correlation with GHR: {correlation}')

    # **Skip saving if the correlation is below 0.9**
    if correlation < 0.9:
        print(f'Skipping {invertor_name} due to low correlation with GHR.')
        return  # Skip this inverter

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
    df_hourly.loc[df_hourly['Active_Power'] < 0.001, 'Active_Power'] = 0
    
    # 4. 날짜별로 그룹화하고 margin 조절
    df_cleaned_6 = df_hourly.groupby(df_hourly['timestamp'].dt.date).apply(adjust_daily_margin)
    df_cleaned_6 = df_cleaned_6.reset_index(drop=True)

    total_dates = df_cleaned_6['timestamp'].dt.date.nunique()
    print(count_date, total_dates)

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
        'ambient_temperature_o_149575',   # Weather_Temperature_Celsius
        'poa_irradiance_o_149574',        # POA (Plane of Array Irradiance)
    ]
    # 인버터 열 추가 (inv1 ~ inv24)
    for i in range(1, 25):
        if i != 15:
            inv_col = f'inv_{i:02d}_ac_power_inv_{149583 + (i-1)*5}'
        else:
            # 특수한 경우 처리 ('inv_15_ac_power_iinv_149653')
            inv_col = 'inv_15_ac_power_iinv_149653'
        columns_to_keep.append(inv_col)

    # df_merged에서 해당 열들만 남기기
    df_filtered = df_merged[columns_to_keep]
    # 열 이름 변경
    rename_dict = {
        'measured_on': 'timestamp',
        'ambient_temperature_o_149575': 'Weather_Temperature_Fahrenheit',
        'poa_irradiance_o_149574': 'Global_Horizontal_Radiation',
    }

    # 인버터 열 이름 변경
    for i in range(1, 25):
        if i != 15:
            old_name = f'inv_{i:02d}_ac_power_inv_{149583 + (i-1)*5}'
        else:
            old_name = 'inv_15_ac_power_iinv_149653'
        rename_dict[old_name] = f'inv{i}'

    df_filtered.rename(columns=rename_dict, inplace=True)

    # 데이터 전처리 및 인버터별 처리
    # timestamp를 datetime 형식으로 변환
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # 데이터를 1시간 간격으로 리샘플링
    df_filtered.set_index('timestamp', inplace=True)
    df_resampled = df_filtered.resample('1H').mean()
    df_resampled.reset_index(inplace=True)
    combined_data = df_resampled

    # 화씨 온도를 섭씨로 변환
    combined_data['Weather_Temperature_Celsius'] = (combined_data['Weather_Temperature_Fahrenheit'] - 32) * 5 / 9 
    combined_data.drop('Weather_Temperature_Fahrenheit', axis=1, inplace=True)

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

    active_power_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_electrical_data.csv')
    env_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_environment_data.csv')
    irrad_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_irradiance_data.csv')
    meter_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/2107_meter_15m_data.csv')
    merged_data = merge_raw_data(active_power_path, env_path, irrad_path, meter_path)

    invertor_list = [f'inv{i}' for i in range(1,25)]

    log_file_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/log.txt')
    for i, invertor_name in enumerate(invertor_list):
        combine_into_each_invertor(
            invertor_name, 
            i, 
            save_dir=os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/preprocessed'),
            log_file_path=log_file_path,
            raw_df= merged_data
        )