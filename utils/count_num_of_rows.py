import os
import pandas as pd

def count_rows_in_csv(directory_path):
    total_rows = 0
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            print(f"Reading {file_path}")
            try:
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                # 첫 줄 제외한 row 수 계산
                total_rows += len(df) - 1
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return total_rows

# 디렉토리 경로 설정
directory = "/ailab_mat/dataset/PV/UK/processed_data_all"  # 여기에 디렉토리 경로를 입력하세요.
total_row_count = count_rows_in_csv(directory)
print("===============================================")
print(f"Total rows (excluding header): {total_row_count}")
print("===============================================")
