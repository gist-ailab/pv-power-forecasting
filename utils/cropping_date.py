import pandas as pd
import os
import glob

def filter_csv_by_date_range(input_directory, output_directory, start_date='2023-09-01', end_date='2024-08-31'):
    """
    CSV 파일들을 지정된 날짜 범위로 필터링
    
    Parameters:
    input_directory (str): 입력 CSV 파일들이 있는 디렉토리
    output_directory (str): 필터링된 파일들을 저장할 디렉토리
    start_date (str): 시작 날짜 (YYYY-MM-DD)
    end_date (str): 종료 날짜 (YYYY-MM-DD)
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_directory, exist_ok=True)
    
    # CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
    
    print(f"총 {len(csv_files)}개 파일을 처리합니다.")
    print(f"필터링 기간: {start_date} ~ {end_date}")
    
    for file_path in csv_files:
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            
            # timestamp를 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 날짜 범위로 필터링
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date + ' 23:59:59')
            df_filtered = df[mask].copy()
            
            # 결과 저장
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_directory, filename)
            df_filtered.to_csv(output_path, index=False)
            
            print(f"{filename}: {len(df)} → {len(df_filtered)} rows")
            
        except Exception as e:
            print(f"오류 - {os.path.basename(file_path)}: {e}")
    
    print("완료!")

# 사용 예시
filter_csv_by_date_range(
    input_directory="/home/bak/Projects/pv-power-forecasting/data/GISTchrono/processed_data_split/test/",
    output_directory="/home/bak/Projects/pv-power-forecasting/data/GISTchrono/processed_data_split/test_cropping/"
)