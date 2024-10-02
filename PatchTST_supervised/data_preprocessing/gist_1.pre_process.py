import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import deepcopy

def wrapup(file_list, index_of_site,
           kor_name, eng_name,
           weather_data,
           save_dir, log_file_path):
    df = pd.DataFrame(columns=['date', 'time', 'Active_Power', 'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity'])
    
    weather_info = pd.read_csv(weather_data, encoding='unicode_escape')
    weather_info.columns = ['datetime', 'temperature', 'wind_direction', 'precipitation', 'humidity']
    weather_info['datetime'] = pd.to_datetime(weather_info['datetime'])
    # print(weather_info)

    env_columns = ['time', 'radiation_horizontal', 'temperature_outdoor', 'radiation_incline', 'temperature_module',]

    empty_rows = pd.concat([pd.DataFrame(df.columns)]*24, axis=1).T
    empty_rows.columns = df.columns

    for i, file in tqdm(enumerate(file_list), total=len(file_list), desc=f'Processing {kor_name}. Out of {index_of_site+1}/16'):
        ## read pv info
        daily_pv_data = pd.read_csv(file)
        daily_pv_data.columns = daily_pv_data.iloc[0]
        daily_pv_data.columns.values[:len(env_columns)] = env_columns
        daily_pv_data = daily_pv_data.drop([0, 1, 2])
        daily_pv_data = daily_pv_data.reset_index(drop=True)

        if kor_name not in daily_pv_data.columns:
            continue

        columns_to_keep = daily_pv_data.columns[:5].tolist()  # 첫 5개의 영문 컬럼 유지
        columns_to_keep.append(kor_name)  # kor_name 컬럼 유지
        # 나머지 컬럼 삭제
        daily_pv_data = daily_pv_data[columns_to_keep]

        # 결측치 처리:'-' 또는 빈 값을 NaN으로 변환
        daily_pv_data = daily_pv_data.map(lambda x: np.nan if x == '-' else x)

        ## get date
        pv_date = file.split('_')[-2]
        pv_date = pd.to_datetime(pv_date).date()
        daily_weather_data = weather_info[weather_info['datetime'].dt.date == pv_date]
        daily_weather_data = daily_weather_data.reset_index(drop=True)

        # Check if there is no weather data for the given date
        if daily_weather_data.empty:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'No weather data for {pv_date}. Skipping...')
            continue

        # Step 1: daily_pv_data에 있는 결측치 처리
        flag = False
        for column in daily_pv_data.columns:
            missing_values_count = daily_pv_data[column].isnull().sum()
            if missing_values_count == 1:
                # 첫 번째 값이 NaN일 경우: 이후 값으로 채움
                if pd.isna(daily_pv_data[column].iloc[0]):
                    daily_pv_data[column].ffill()

                # 마지막 값이 NaN일 경우: 이전 값으로 채움
                elif daily_pv_data[column].iloc[-1] is np.nan:
                    daily_pv_data[column].bfill()

                else:   # 결측치가 중간에 있는 경우: 앞뒤 값의 평균으로 채움
                    ffill_values = daily_pv_data[column].ffill()
                    bfill_values = daily_pv_data[column].bfill()
                    daily_pv_data[column] = (ffill_values + bfill_values) / 2

            elif missing_values_count >= 2:
                # 결측치가 2개 이상인 경우: 해당 날을 스킵
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"Skipping {pv_date} due to too many missing values in {column}\n")
                flag = True
                break

        if flag:
            continue

        # Step 1: 'date'와 'time' 데이터를 추출하여 새로운 DataFrame 생성
        temp_df = pd.DataFrame(
            columns=['timestep', 'Active_Power', 'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                     'Weather_Relative_Humidity'])

        # Step 2: temp_df에 데이터 채워넣기
        temp_df['timestep'] = daily_weather_data['datetime']
        temp_df['Active_Power'] = daily_pv_data[kor_name]
        temp_df['Global_Horizontal_Radiation'] = daily_pv_data['radiation_horizontal']
        temp_df['Weather_Temperature_Celsius'] = daily_weather_data['temperature']
        temp_df['Weather_Relative_Humidity'] = daily_weather_data['humidity']

        # Step 3: DataFrame 결합 (concat)
        if df.empty:
            df = deepcopy(temp_df)
        else:
            df = pd.concat([df, temp_df], ignore_index=True)

    save_path = os.path.join(save_dir, f'{eng_name}.csv')
    with open(save_path, 'w') as f:
        df.to_csv(f, index=False)


def create_combined_weather_csv(create_path, project_root):
    weather_data_dir = os.path.join(project_root, 'data/GIST_dataset/weather')
    weather_csv_files = [f for f in os.listdir(weather_data_dir) if f.endswith('.csv')]
    weather_csv_files.sort()

    data_frames = []
    for file in weather_csv_files:
        file_path = os.path.join(weather_data_dir, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8', skiprows=1, header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1', skiprows=1, header=None)

        # df = df.reindex(columns=expected_columns)   # Reindex the DataFrame to ensure consistent columns
        data_frames.append(df)
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.drop(combined_df.columns[:2], axis=1, inplace=True)

    # Define the column names the add
    column_names = ['datetime', 'temperature', 'wind_direction', 'precipitation', 'humidity']
    combined_df.columns = column_names
    combined_df['datetime'] = pd.to_datetime(combined_df.iloc[:, 0])    # 3번째 컬럼을 datetime 형식으로 변환 (시간 관련 처리를 위해)

    # Step 1: datetime을 인덱스로 설정하고 1시간 단위로 리샘플링
    combined_df.set_index('datetime', inplace=True)
    # Step 2: 1시간 단위로 리샘플링하여 결측값을 확인
    df_resampled = combined_df.resample('1h').mean()
    # Step 3: 1시간이 빠진 경우 앞뒤 값의 평균으로 채우기. 단일 결측값에 대해 선형 보간을 사용하여 평균값으로 채움
    filled_values = df_resampled[df_resampled.isna()]  # 보간 전 결측값 기록
    df_resampled.interpolate(method='linear', inplace=True)

    # Step 4: 2시간 이상 연속 결측값이 있던 날짜는 삭제
    # 결측값이 2시간 이상 연속되면 해당 날짜를 제거
    mask = df_resampled.isna().astype(int).groupby(df_resampled.index.floor('D')).sum() >= 2
    dates_to_remove = mask[mask.any(axis=1)].index

    # Step 4: 해당 날짜들을 제거
    df_cleaned = df_resampled[~df_resampled.index.floor('D').isin(dates_to_remove)]

    # # Check for missing dates
    # unique_dates = pd.to_datetime(df_cleaned.index.date).unique()
    # missing_dates = pd.date_range(start=unique_dates[0], end=unique_dates[-1]).difference(unique_dates)
    # print(f'Missing dates: {missing_dates}') if missing_dates else print('No missing dates')
    # # Check for missing hours
    # full_time_range = pd.date_range(start=df_cleaned.index.min(), end=df_cleaned.index.max(), freq='h')
    # actual_times = df_cleaned.index
    # missing_times = full_time_range.difference(actual_times)
    # print(f'Missing times: {missing_times}') if missing_times else print('No missing times')

    # Save the combined DataFrame to a new CSV file
    df_cleaned.to_csv(create_path, index=True)

def check_new_columns(pv_file_list):
    previous_elements = set()

    for i, file in tqdm(enumerate(pv_file_list), total=len(pv_file_list), desc='Checking for added new columns'):
        df = pd.read_excel(file, engine='xlrd')
        third_row = df.iloc[2]
        if i == 0:
            previous_elements = set(third_row.dropna().tolist())
            continue
        else:
            current_elements = set(third_row.dropna().tolist())
            new_elements = current_elements - previous_elements
            if new_elements == set():
                continue
            else:
                print(f'New elements in {file}: {new_elements}')
        previous_elements = current_elements

def convert_excel_to_hourly_csv(file_list):
    for i, xls_file in tqdm(enumerate(file_list), total=len(file_list), desc='Converting Excel to CSV'):
        df = pd.read_excel(xls_file, engine='xlrd')
        df = df.drop([0, 1])

        start_column = 5  # 6번째 열의 인덱스는 5 (0부터 시작하므로)
        row_index = 0  # 1번째 행의 인덱스는 0
        for col in range(df.shape[1] - 1, start_column - 1, -1):
            if col + 1 < df.shape[1]:
                df.iloc[row_index, col + 1] = df.iloc[row_index, col]

        # 6번째 열부터 짝수 인덱스를 가진 열들을 삭제
        columns_to_drop = [i for i in range(start_column, df.shape[1]) if (i - start_column) % 2 == 0]
        df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

        last_valid_row = df[df.iloc[:, 0] == '23 시'].index  # '23 시'가 있는 행의 인덱스를 찾음
        df = df.iloc[:last_valid_row[-1]-1]  # '23 시'가 있는 행까지만 유지, 그 이후는 제거

        save_name = xls_file.split('/')[-1].replace('xls', 'csv')
        save_name = '.'.join(save_name.split('.')[1:])
        save_dir = xls_file.split('/')[:-2]
        save_dir.append('daily_PV_csv')
        save_dir = '/'.join(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        df.to_csv(save_path, index=False)
    print('Conversion completed!')


if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(current_file_path))

    pv_xls_data_dir = os.path.join(project_root, 'data/GIST_dataset/daily_PV_xls')
    pv_file_list = [os.path.join(pv_xls_data_dir, _) for _ in os.listdir(pv_xls_data_dir)]
    pv_file_list.sort()

    # Define the path to save the combined CSV file
    weather_data = os.path.join(project_root, 'data/GIST_dataset/GIST_weather_data.csv')
    if not os.path.exists(weather_data):
        create_combined_weather_csv(weather_data, project_root)

    # Check for new columns in the PV data
    # check_new_columns(pv_file_list)

    # convert_excel_to_hourly_csv(pv_file_list)

    pv_csv_data_dir = os.path.join(project_root, 'data/GIST_dataset/daily_PV_csv')
    pv_file_list = [os.path.join(pv_csv_data_dir, _) for _ in os.listdir(pv_csv_data_dir)]
    pv_file_list.sort()

    site_dict = {
        '축구장': 'Soccer-Field',
        '학생회관': 'W06_Student-Union',
        '중앙창고': 'W13_Centeral-Storage',
        '학사과정': 'E11_DormA',
        '다산빌딩': 'C09_Dasan',
        '시설관리동': 'W11_Facility-Maintenance-Bldg',
        '대학C동': 'N06_College-Bldg',
        '동물실험동': 'E02_Animal-Recource-Center',
        '중앙도서관': 'N01_Central-Library',
        'LG도서관': 'N02_LG-Library',
        '신재생에너지동': 'C10_Renewable-E-Bldg',
        '삼성환경동': 'C07_Samsung-Env-Bldg',
        '중앙연구기기센터': 'C11_GAIA',
        '산업협력관': 'E03_GTI',
        '학사B동': 'E12_DormB',
        '자연과학동': 'E8_Natural-Science-Bldg'
    }

    log_file_path = os.path.join(project_root, 'data/GIST_dataset/log.txt')
    for i, (kor_name, eng_name) in enumerate(site_dict.items()):
        wrapup(pv_file_list, i,
               kor_name, eng_name,
               weather_data,
               os.path.join(project_root, 'data/GIST_dataset'),
               log_file_path)
    


