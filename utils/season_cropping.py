import pandas as pd
import os
import glob

def process_seasonal_split(input_dir, output_dir):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 계절 매핑
    month_groups = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }
    
    # 계절별 디렉토리 생성
    for season_name in month_groups.keys():
        season_dir = os.path.join(output_dir, season_name)
        os.makedirs(season_dir, exist_ok=True)
    
    # CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for csv_file in csv_files:
        print(f"Processing: {os.path.basename(csv_file)}")
        
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        
        # 각 계절별로 분할
        for season_name, months in month_groups.items():
            season_df = df[df['month'].isin(months)].copy()
            
            if len(season_df) > 0:
                # 계절별 디렉토리에 원본 파일명으로 저장
                season_dir = os.path.join(output_dir, season_name)
                output_path = os.path.join(season_dir, os.path.basename(csv_file))
                
                # month 컬럼 제거 후 저장
                season_df = season_df.drop('month', axis=1)
                season_df.to_csv(output_path, index=False)
                
                print(f"  {season_name}: {len(season_df)} records")

# 실행
input_directory = '/home/bak/Projects/pv-power-forecasting/data/GISTchrono/processed_data_split/test_1year/'
output_directory = '/home/bak/Projects/pv-power-forecasting/data/GISTchrono/processed_data_split/seasonal/'

process_seasonal_split(input_directory, output_directory)