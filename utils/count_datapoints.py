import pandas as pd
import os
from pathlib import Path

def count_datapoints_in_folders(base_directory):
    """
    train, val, test 폴더 내 CSV 파일들의 데이터 포인트 개수를 계산
    
    Parameters:
    base_directory (str): train, val, test 폴더가 있는 상위 디렉토리 경로
    """
    
    base_path = Path(base_directory)
    
    # 폴더별 결과 저장
    folder_stats = {}
    
    # train, val, test 폴더 순회
    for folder_name in ['train', 'val', 'test']:
        folder_path = base_path / folder_name
    # for folder_name in ['train', 'test']:
    #     folder_path = base_path / folder_name
        
        if not folder_path.exists():
            print(f"⚠️ {folder_name} 폴더가 존재하지 않습니다: {folder_path}")
            continue
        
        print(f"\n=== {folder_name.upper()} 폴더 분석 ===")
        print(f"경로: {folder_path}")
        
        # CSV 파일 목록 가져오기
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"CSV 파일이 없습니다.")
            folder_stats[folder_name] = {'files': 0, 'total_rows': 0, 'file_details': []}
            continue
        
        csv_files.sort()  # 파일명 정렬
        
        total_rows = 0
        file_details = []
        
        for csv_file in csv_files:
            try:
                file_path = folder_path / csv_file
                df = pd.read_csv(file_path)
                rows = len(df)
                total_rows += rows
                
                file_details.append({
                    'filename': csv_file,
                    'rows': rows
                })
                
                print(f"  {csv_file}: {rows:,} rows")
                
            except Exception as e:
                print(f"  ❌ {csv_file}: 오류 발생 - {str(e)}")
                file_details.append({
                    'filename': csv_file,
                    'rows': 0,
                    'error': str(e)
                })
        
        folder_stats[folder_name] = {
            'files': len(csv_files),
            'total_rows': total_rows,
            'file_details': file_details
        }
        
        print(f"\n📊 {folder_name} 폴더 요약:")
        print(f"  - 파일 수: {len(csv_files)}개")
        print(f"  - 총 데이터 포인트: {total_rows:,}개")
        print(f"  - 평균 데이터 포인트/파일: {total_rows/len(csv_files):.0f}개")
    
    return folder_stats

def print_summary_table(folder_stats):
    """
    전체 요약 테이블 출력
    """
    print("\n" + "="*60)
    print("📋 전체 요약")
    print("="*60)
    
    # 테이블 헤더
    print(f"{'폴더':<10} {'파일 수':<10} {'총 데이터 포인트':<15} {'평균/파일':<12}")
    print("-" * 60)
    
    total_files = 0
    total_datapoints = 0
    
    for folder_name, stats in folder_stats.items():
        files = stats['files']
        rows = stats['total_rows']
        avg = rows / files if files > 0 else 0
        
        total_files += files
        total_datapoints += rows
        
        print(f"{folder_name:<10} {files:<10} {rows:<15,} {avg:<12.0f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_files:<10} {total_datapoints:<15,} {total_datapoints/total_files if total_files > 0 else 0:<12.0f}")
    
    # 비율 계산
    if total_datapoints > 0:
        print(f"\n📈 데이터 분할 비율:")
        for folder_name, stats in folder_stats.items():
            ratio = (stats['total_rows'] / total_datapoints) * 100
            print(f"  {folder_name}: {ratio:.1f}%")

def save_detailed_report(folder_stats, output_file="datapoint_report.csv"):
    """
    상세 리포트를 CSV 파일로 저장
    """
    report_data = []
    
    for folder_name, stats in folder_stats.items():
        for file_detail in stats['file_details']:
            report_data.append({
                'folder': folder_name,
                'filename': file_detail['filename'],
                'rows': file_detail.get('rows', 0),
                'error': file_detail.get('error', '')
            })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_file, index=False)
    print(f"\n💾 상세 리포트 저장: {output_file}")
    
    return report_df

# 사용 예시
if __name__ == "__main__":
    # 기본 디렉토리 경로 설정 (train, val, test 폴더가 있는 상위 경로)
    base_directory = "/home/bak/Projects/pv-power-forecasting/data/GISTchrono/processed_data_split/"  # 실제 경로로 변경하세요
    # base_directory = "/home/bak/Projects/pv-power-forecasting/data/Germanychrono/processed_data_split/"  # 실제 경로로 변경하세요
    
    # 데이터 포인트 개수 계산
    stats = count_datapoints_in_folders(base_directory)
    
    # 요약 테이블 출력
    print_summary_table(stats)
    
    # 상세 리포트 저장
    # save_detailed_report(stats)