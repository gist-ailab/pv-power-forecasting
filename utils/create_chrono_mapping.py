import pandas as pd
import os
from pathlib import Path

def create_gist_chrono_mapping(processed_data_folder, output_file="mapping_all.csv"):
    """
    GISTchrono 데이터셋에 대한 매핑 파일 생성 (train/val/test 각각 별도 항목)
    
    Parameters:
    processed_data_folder (str): train/val/test 파일들이 있는 폴더 경로
    output_file (str): 생성할 매핑 파일 이름
    """
    
    # 폴더 경로 설정
    data_path = Path(processed_data_folder)
    
    # CSV 파일 목록 가져오기
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    # train, val, test 파일들을 분류
    # train_files = [f for f in csv_files if f.endswith('_train.csv')]
    train_files = [f for f in csv_files if f.endswith('.csv') and not f.endswith('_1.csv') and not f.endswith('_2.csv')]
    # val_files = [f for f in csv_files if f.endswith('_1.csv')]
    test_files = [f for f in csv_files if f.endswith('_test.csv')]
    
    # 파일명 정렬
    train_files.sort()
    # val_files.sort()
    test_files.sort()
    
    # 매핑 데이터 생성
    mapping_data = []
    current_idx = 1

    dataset_name = 'Germanychrono'  # 데이터셋 이름
    print(f"dataset_name: {dataset_name}")

    # Train 파일들 처리
    print("=== Train 파일들 ===")
    for train_file in train_files:
        mapping_name = f"{current_idx:02d}_{train_file}"
        mapping_data.append({
            'dataset': dataset_name,
            'original_name': train_file,
            'mapping_name': mapping_name
        })
        print(f"{current_idx:02d}: {train_file} -> {mapping_name}")
        current_idx += 1
    
    # # Val 파일들 처리
    # print("\n=== Validation 파일들 ===")
    # for val_file in val_files:
    #     mapping_name = f"{current_idx:02d}_{val_file}"
    #     mapping_data.append({
    #         'dataset': 'GISTchrono',
    #         'original_name': val_file,
    #         'mapping_name': mapping_name
    #     })
    #     print(f"{current_idx:02d}: {val_file} -> {mapping_name}")
    #     current_idx += 1
    
    # Test 파일들 처리
    print("\n=== Test 파일들 ===")
    for test_file in test_files:
        mapping_name = f"{current_idx:02d}_{test_file}"
        mapping_data.append({
            'dataset': dataset_name,
            'original_name': test_file,
            'mapping_name': mapping_name
        })
        print(f"{current_idx:02d}: {test_file} -> {mapping_name}")
        current_idx += 1
    
    # 데이터프레임 생성
    mapping_df = pd.DataFrame(mapping_data)
    
    # CSV 파일로 저장
    output_path = data_path.parent / output_file  # processed_data_all의 상위 폴더에 저장
    mapping_df.to_csv(output_path, index=False)
    
    print(f"\n매핑 파일 생성 완료: {output_path}")
    print(f"총 {len(mapping_data)}개의 파일 매핑")
    print(f"- Train: {len(train_files)}개")
    # print(f"- Validation: {len(val_files)}개") 
    print(f"- Test: {len(test_files)}개")
    
    # 결과 확인
    print("\n=== 생성된 매핑 파일 내용 ===")
    print(mapping_df.to_string(index=False))
    
    return mapping_df

def verify_file_structure(processed_data_folder, mapping_df):
    """
    파일 구조 확인 및 통계 출력
    """
    data_path = Path(processed_data_folder)
    
    print("\n=== 파일 구조 통계 ===")
    
    # 각 타입별 파일 수 계산
    train_count = len([row for _, row in mapping_df.iterrows() if '_train.csv' in row['original_name']])
    # val_count = len([row for _, row in mapping_df.iterrows() if '_val.csv' in row['original_name']])
    test_count = len([row for _, row in mapping_df.iterrows() if '_test.csv' in row['original_name']])
    
    print(f"Train 파일 수: {train_count}")
    # print(f"Validation 파일 수: {val_count}")
    print(f"Test 파일 수: {test_count}")
    print(f"총 파일 수: {len(mapping_df)}")
    
    # 파일 존재 여부 확인
    missing_files = []
    for _, row in mapping_df.iterrows():
        file_path = data_path / row['original_name']
        if not file_path.exists():
            missing_files.append(row['original_name'])
    
    if missing_files:
        print(f"\n⚠️ 누락된 파일들 ({len(missing_files)}개):")
        for missing_file in missing_files:
            print(f"  - {missing_file}")
    else:
        print("\n✅ 모든 매핑된 파일이 존재합니다!")

# 사용 예시
if __name__ == "__main__":
    # 실제 폴더 경로로 변경하세요
    # processed_data_folder = "/home/bak/Projects/pv-power-forecasting/data/GISTchrono/processed_data_all"
    # processed_data_folder = "/home/bak/Projects/pv-power-forecasting/data/GIST_chronological/processed_data_all"
    processed_data_folder = "/home/bak/Projects/pv-power-forecasting/data/Germanychrono/processed_data_all"
    
    # 매핑 파일 생성
    mapping_df = create_gist_chrono_mapping(processed_data_folder)
    
    # 파일 구조 확인
    verify_file_structure(processed_data_folder, mapping_df)