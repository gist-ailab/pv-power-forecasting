import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

def process_csv_files(input_directory, output_directory=None):
    """
    디렉토리 내 CSV 파일들을 읽어서 하루종일 Active_Power가 0인 날을 제거
    
    Parameters:
    input_directory (str): 입력 CSV 파일들이 있는 디렉토리 경로
    output_directory (str): 처리된 파일을 저장할 디렉토리 경로 (None이면 원본 파일 덮어쓰기)
    """
    
    # 출력 디렉토리 설정
    if output_directory is None:
        output_directory = input_directory
    else:
        os.makedirs(output_directory, exist_ok=True)
    
    # CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
    
    if not csv_files:
        print(f"디렉토리 '{input_directory}'에서 CSV 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(csv_files)}개의 CSV 파일을 처리합니다.")
    
    for file_path in csv_files:
        try:
            # 파일 읽기
            df = pd.read_csv(file_path)
            
            # timestamp 컬럼을 datetime 타입으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 날짜별로 그룹화
            df['Date'] = df['timestamp'].dt.date
            
            # 원본 데이터 수
            original_count = len(df)
            
            # 각 날짜별로 Active_Power의 합계 계산
            daily_power = df.groupby('Date')['Active_Power'].sum()
            
            # Active_Power가 0보다 큰 날짜들만 선택
            valid_dates = daily_power[daily_power > 0].index
            
            # 유효한 날짜의 데이터만 필터링
            df_filtered = df[df['Date'].isin(valid_dates)].copy()
            
            # Date 컬럼 제거 (임시로 만든 컬럼이므로)
            df_filtered = df_filtered.drop('Date', axis=1)
            
            # 처리 결과 출력
            removed_count = original_count - len(df_filtered)
            removed_days = len(daily_power) - len(valid_dates)
            
            print(f"\n파일: {os.path.basename(file_path)}")
            print(f"  - 원본 데이터 포인트: {original_count:,}개")
            print(f"  - 제거된 데이터 포인트: {removed_count:,}개")
            print(f"  - 남은 데이터 포인트: {len(df_filtered):,}개")
            print(f"  - 제거된 날짜 수: {removed_days}일")
            print(f"  - 유효한 날짜 수: {len(valid_dates)}일")
            
            # 결과 파일 저장
            output_path = os.path.join(output_directory, os.path.basename(file_path))
            df_filtered.to_csv(output_path, index=False)
            
            print(f"  - 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({file_path}): {str(e)}")
    
    print("\n모든 파일 처리가 완료되었습니다.")

def analyze_zero_power_days(file_path):
    """
    특정 파일의 0 power 날짜들을 분석하는 함수
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Date'] = df['timestamp'].dt.date
    
    # 날짜별 Active_Power 합계
    daily_power = df.groupby('Date')['Active_Power'].sum()
    
    # 0인 날짜들
    zero_power_dates = daily_power[daily_power == 0].index
    valid_dates = daily_power[daily_power > 0].index
    
    print(f"전체 날짜 수: {len(daily_power)}")
    print(f"Active_Power가 0인 날짜 수: {len(zero_power_dates)}")
    print(f"유효한 날짜 수: {len(valid_dates)}")
    
    if len(zero_power_dates) > 0:
        print(f"\n0 power 날짜 범위:")
        print(f"  시작: {min(zero_power_dates)}")
        print(f"  끝: {max(zero_power_dates)}")
    
    if len(valid_dates) > 0:
        print(f"\n유효한 날짜 범위:")
        print(f"  시작: {min(valid_dates)}")
        print(f"  끝: {max(valid_dates)}")
    
    return daily_power

# 사용 예시:
if __name__ == "__main__":
    # 1. 업로드된 파일 분석
    # print("=== 업로드된 파일 분석 ===")
    # 1. 디렉토리 내 모든 CSV 파일 분석
    print("=== 디렉토리 내 모든 CSV 파일 분석 ===")
    input_dir = '/home/bak/Downloads/new_processed_data_all'
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for file_path in csv_files:
        print(f"\n{'='*50}")
        print(f"파일: {os.path.basename(file_path)}")
        print('='*50)
        daily_power = analyze_zero_power_days(file_path)
    # daily_power = analyze_zero_power_days('/home/bak/Downloads/processed_data_all-20250527T034932Z-1-001/processed_data_all/4.75_DE_KN_industrial1_pv_1.csv')
    
    # 2. 디렉토리 내 모든 CSV 파일 처리
    # 사용법:
    # process_csv_files("input_directory_path", "output_directory_path")
    
    # 예시 (실제 경로로 변경해서 사용):
    # process_csv_files("/home/bak/Downloads/processed_data_all", "/home/bak/Downloads/new_processed_data_all")
    
    # 원본 파일을 덮어쓰고 싶다면:
    # process_csv_files("/path/to/your/csv/files")