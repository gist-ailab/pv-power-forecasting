import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import deepcopy

def wrapup(file_list,
           weather_data,
           save_name):
    df = pd.DataFrame(columns=['date', 'time', 'Active_Power', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity'])
    
    weather_info = pd.read_csv(weather_data, encoding='unicode_escape')
    weather_info.columns = ['datetime', 'temperature', 'wind_direction', 'precipitation', 'humidity']
    # print(weather_info)

    '''
    pv_columns = ['time', 'radiation_horizontal', 'temperature_outdoor', 'radiation_incline', 'temperature_module',
                  '1_soccer-field_hourly_power',
                  '2_student-union(W6)_hourly_power',
                  '3_center-warehouse(W13)_hourly_power',
                  '4_dormA(E11)_hourly_power',
                  '5_dasan(C9)_hourly_power',
                  '6_sisuldong(W11)_hourly_power',
                  '7_univC(N6)_hourly_power',
                  '8_animal-exp(E2)_hourly_power',
                  '9_main-library(N1)_hourly_power',
                  '10_LG-library(N2)_hourly_power',
                  '11_renewable-energy(C10)_hourly_power',
                  '12_samsung-env(C7)_hourly_power',
                  '13_GAIA(C11)_hourly_power',
                  '14_GTI(E3)_hourly_power',
                  '15_dormB(E12)_hourly_power',
                  '16_physics(E8)_hourly_power',
                  'daily_load']
    '''
    env_columns = ['time', 'radiation_horizontal', 'temperature_outdoor', 'radiation_incline', 'temperature_module',]
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
        '자연과학동': 'E8_Natural-Science-Bldg'}
    # pv_columns = ['time', 'radiation_horizontal', 'temperature_outdoor', 'radiation_incline', 'temperature_module',
    #               'Soccer-Field_hourly_power',
    #               'W06_Student-Union_hourly_power',
    #               'W13_Centeral-Storage_hourly_power',
    #               'E11_DormA_hourly_power',
    #               'C09_Dasan_hourly_power',
    #               'W11_Facility-Maintenance-Bldg_hourly_power',
    #               'N06_College-Bldg_hourly_power',
    #               'E02_Animal-Recource-Center_hourly_power',
    #               'N01_Central-Library_hourly_power',
    #               'N02_LG-Library_hourly_power',
    #               'C10_Renewable-E-Bldg_hourly_power',
    #               'C07_Samsung-Env-Bldg_hourly_power',
    #               'C11_GAIA_hourly_power',
    #               'E03_GTI_hourly_power',
    #               'E12_DormB_hourly_power',
    #               'E8_Natural-Science-Bldg_hourly_power',
    #               'daily_load']
    empty_rows = pd.concat([pd.DataFrame(df.columns)]*24, axis=1).T
    empty_rows.columns = df.columns
    # df = pd.DataFrame(columns=['date', 'time', 'Active_Power', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Relative_Humidity'])
    
    for i, file in enumerate(file_list):
        ## read pv info
        pv_info = pd.read_csv(file)
        pv_info.columns.values[:len(env_columns)] = env_columns
        
        ## get date
        pv_date = file.split('_')[-2]
        weather_data = weather_info[weather_info['datetime'].str.contains(pv_date)]
        weather_data = weather_data.reset_index(drop=True)


        # weather_info['date'].str.contains(pv_date)
        if pv_date == '2024-09-25': continue
                
        ## check if the time is correct
        pv_time = [_time.split(' ')[0] for _time in pv_info.iloc[3:27]['time'].values]
        if i == 0:
            weather_time = ['00']
            weather_time.extend([_time.split(' ')[1].split(':')[0] for _time in weather_data['date'].values])
            weather_data = pd.concat([pd.DataFrame({
                'date': f'{pv_date} 00:00',
                'spot': 783,
                'spot_name': 'GIST',
                'humidity': weather_data.loc[0, 'humidity'],                
            }, index=[0]), weather_data], axis=0)
            weather_data = weather_data.reset_index(drop=True)
        else:
            weather_time = [_time.split(' ')[1].split(':')[0] for _time in weather_data['date'].values]
        
        ## handling the missing humidity data. 앞에 있는 값을 그대로 가져옴
        if weather_data['humidity'].isnull().sum():
            missing_idx = weather_data[weather_data['humidity'].isnull()].index
            for idx in missing_idx:
                weather_data.loc[idx, 'humidity'] = weather_data.loc[idx-1, 'humidity']
        
        ## handling the missing weather data
        missing_time = list(set(pv_time) - set(weather_time))
        missing_time.sort()
        
        for m_time in missing_time:
            m_time = int(m_time)
            ## forward filling
            tmp1 = weather_data.loc[weather_data.index < m_time]
            tmp2 = weather_data.loc[weather_data.index >= m_time]
            weather_data = pd.concat([tmp1, pd.DataFrame({
                'date': f'{pv_date} {str(m_time).zfill(2)}:00',
                'spot': 783,
                'spot_name': 'GIST',
                'humidity': weather_data.loc[m_time-1, 'humidity'],                
            }, index=[m_time])], axis=0)
            weather_data = pd.concat([weather_data, tmp2], axis=0)
            weather_data = weather_data.reset_index(drop=True)
        
        
        assert len(pv_time) == len(weather_data)
        
        ## add empty rows for 24 hours
        df = pd.concat([df, empty_rows], axis=0)
        df = df.reset_index(drop=True)
        
        ## add data
        df.loc[24*i:24*(i+1)-1,'timestamp']                     = [pv_date + ' ' + str(i).zfill(2) + ':00' for i in range(24)]
        df.loc[24*i:24*(i+1)-1,'date']                          = pv_date
        df.loc[24*i:24*(i+1)-1,'time']                          = [str(i).zfill(2) + ':00' for i in range(24)]
        df.loc[24*i:24*(i+1)-1,'Active_Power']                  = pv_info.iloc[5:29]['6_sisuldong_hourly_power'].values
        df.loc[24*i:24*(i+1)-1,'Weather_Temperature_Celsius']   = pv_info.iloc[5:29]['temperature_outdoor'].values
        df.loc[24*i:24*(i+1)-1,'Global_Horizontal_Radiation']   = pv_info.iloc[5:29]['radiation_horizontal'].values
        # df.loc[24*i:24*(i+1)-1,'Diffuse_Horizontal_Radiation']  = pv_info.iloc[5:29]['radiation_incline'].values
        df.loc[24*i:24*(i+1)-1,'Weather_Relative_Humidity']     = weather_data['humidity'].values
            
    
    df.drop(['date', 'time'], axis=1, inplace=True)
    
    for i, column in enumerate(df.columns):
        if column == 'timestamp': continue
        missing_indices = df[df[column] == '-'].index.tolist()
        df.loc[missing_indices, column] = np.nan
        
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].interpolate()
    
    with open(save_name, 'w') as f:
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

    # 3번째 컬럼을 datetime 형식으로 변환 (시간 관련 처리를 위해)
    combined_df['datetime'] = pd.to_datetime(combined_df.iloc[:, 0])

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

    # Step 5: 보간된 날짜 및 시간과 제거된 날짜를 로그로 저장
    with open('log_filled_and_removed.txt', 'w') as log_file:
        log_file.write("===== 보간된 날짜 및 시간 =====\n")
        filled_times = filled_values.index
        for time in filled_times:
            log_file.write(f"{time}\n")

        log_file.write("\n===== 제거된 날짜 =====\n")
        for date in dates_to_remove:
            log_file.write(f"{date}\n")

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

        # 맨 마지막 행에 있는 하루 동안 총 발전량은 제거
        last_row_value = df.iloc[-1, 0]
        if last_row_value != '23 시':
            df = df.drop(df.index[-1])

        save_name = xls_file.split('/')[-1].replace('xls', 'csv')
        save_name = '.'.join(save_name.split('.')[1:])
        save_dir = xls_file.split('/')[:-2]
        save_dir.append('daily_PV_csv')
        save_dir = '/'.join(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        df.to_csv(save_path, index=False)
    print('Conversion completed!')
        # pv_file_list[0].split('/')[-1].split('.')[1]







if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

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

    wrapup(pv_file_list,
           weather_data,
           'dataset/GIST/sisuldong.csv')
    


