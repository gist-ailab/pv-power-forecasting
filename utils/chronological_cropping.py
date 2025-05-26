import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path

def split_pv_data_by_period(input_folder, output_folder):
    """
    PV 데이터를 기간별로 train/validation/test로 분할하는 함수
    
    Parameters:
    input_folder (str): 원본 CSV 파일들이 있는 폴더 경로
    output_folder (str): 분할된 데이터를 저장할 폴더 경로
    """
    
    # 출력 폴더 생성
    output_path = Path(output_folder)
    train_path = output_path / 'train'
    # val_path = output_path / 'val'
    test_path = output_path / 'test'
    
    # for path in [train_path, val_path, test_path]:
    for path in [train_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # 기준 날짜 설정 (2024년 9월 24일을 기준으로)
    end_date = datetime(2024, 9, 24)
    # test_start_date = end_date - timedelta(days=2*365)  # 최근 2년
    # val_start_date = test_start_date - timedelta(days=365)  # 그 전 1년
    test_start_date = end_date - timedelta(days=365)  # 최근 1년
    # val_start_date = test_start_date - timedelta(days=365)  # 그 전 1년
    
    print(f"데이터 분할 기준:")
    print(f"Test set: {test_start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    # print(f"Validation set: {val_start_date.strftime('%Y-%m-%d')} ~ {test_start_date.strftime('%Y-%m-%d')}")
    # print(f"Train set: ~ {val_start_date.strftime('%Y-%m-%d')}")
    print(f"Train set: ~ {test_start_date.strftime('%Y-%m-%d')}")
    print("-" * 50)
    
    # CSV 파일 목록 가져오기
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        try:
            print(f"처리 중: {csv_file}")
            
            # 데이터 읽기
            file_path = os.path.join(input_folder, csv_file)
            df = pd.read_csv(file_path)
            
            # 'time' 컬럼을 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 데이터 기간 확인
            data_start = df['timestamp'].min()
            data_end = df['timestamp'].max()
            print(f"  데이터 기간: {data_start.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")
            
            # 기간별로 데이터 분할
            # train_data = df[df['timestamp'] < val_start_date].copy()
            # val_data = df[(df['timestamp'] >= val_start_date) & (df['timestamp'] < test_start_date)].copy()
            # test_data = df[df['timestamp'] >= test_start_date].copy()
            train_data = df[df['timestamp'] < test_start_date].copy()
            test_data = df[df['timestamp'] >= test_start_date].copy()


            # 각 데이터셋 크기 출력
            print(f"  Train: {len(train_data)} rows")
            # print(f"  Validation: {len(val_data)} rows")
            print(f"  Test: {len(test_data)} rows")
            
            # 파일명에서 확장자 제거
            file_name_without_ext = os.path.splitext(csv_file)[0]
            
            # 각 분할된 데이터 저장 (데이터가 있는 경우만)
            if len(train_data) > 0:
                # train_file_path = train_path / f"{file_name_without_ext}_train.csv"
                train_file_path = train_path / f"{file_name_without_ext}.csv"
                train_data.to_csv(train_file_path, index=False)
                print(f"  Train 데이터 저장: {train_file_path}")
            else:
                print(f"  Train 데이터 없음")
            
            # if len(val_data) > 0:
            #     # val_file_path = val_path / f"{file_name_without_ext}_val.csv"
            #     val_file_path = val_path / f"{file_name_without_ext}_1.csv"
            #     val_data.to_csv(val_file_path, index=False)
            #     print(f"  Validation 데이터 저장: {val_file_path}")
            # else:
            #     print(f"  Validation 데이터 없음")
            
            if len(test_data) > 0:
                test_file_path = test_path / f"{file_name_without_ext}_test.csv"
                # test_file_path = test_path / f"{file_name_without_ext}_2.csv"
                test_data.to_csv(test_file_path, index=False)
                print(f"  Test 데이터 저장: {test_file_path}")
            else:
                print(f"  Test 데이터 없음")
            
            print("-" * 30)
            
        except Exception as e:
            print(f"오류 발생 - {csv_file}: {str(e)}")
            continue
    
    print("모든 파일 처리 완료!")

# 사용 예시
if __name__ == "__main__":
    # 입력 및 출력 폴더 경로 설정
    # input_folder = "/home/bak/Projects/pv-power-forecasting/data/GIST/processed_data_all"  # 원본 CSV 파일들이 있는 폴더 경로
    # output_folder = "/home/bak/Projects/pv-power-forecasting/data/GIST_chronological/processed_data_split"  # 분할된 데이터를 저장할 폴더 경로

    input_folder = "/home/bak/Projects/pv-power-forecasting/data/Germany/processed_data_all"  # 원본 CSV 파일들이 있는 폴더 경로
    output_folder = "/home/bak/Projects/pv-power-forecasting/data/Germanychrono/processed_data_split"  # 분할된 데이터를 저장할 폴더 경로

    # 실제 경로로 변경해서 사용하세요
    # input_folder = "/home/user/pv_data/"
    # output_folder = "/home/user/pv_data_split/"
    
    split_pv_data_by_period(input_folder, output_folder)