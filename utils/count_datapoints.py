import pandas as pd
from datetime import datetime, timedelta
import os
import glob
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


def find_missing_dates(file_path, timestamp_column='timestamp'):
    """
    CSV 파일에서 빠진 날짜를 찾는 함수
    
    Parameters:
    file_path (str): CSV 파일 경로
    timestamp_column (str): 날짜 컬럼명
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        if timestamp_column not in df.columns:
            print(f"오류: '{timestamp_column}' 컬럼을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {list(df.columns)}")
            return None
        
        # timestamp를 datetime으로 변환
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # 날짜만 추출 (시간 제거)
        df['date_only'] = df[timestamp_column].dt.date
        
        # 고유한 날짜들만 추출하고 정렬
        unique_dates = sorted(df['date_only'].unique())
        
        # 전체 기간에서 연속된 날짜 생성
        start_date = min(unique_dates)
        end_date = max(unique_dates)
        
        # 연속된 모든 날짜 생성
        date_range = pd.date_range(start=start_date, end=end_date, freq='D').date
        
        # 빠진 날짜들 찾기
        missing_dates = [date for date in date_range if date not in unique_dates]
        
        # 결과 출력
        print(f"\n=== 파일: {os.path.basename(file_path)} ===")
        print(f"데이터 기간: {start_date} ~ {end_date}")
        print(f"전체 기간: {len(date_range)}일")
        print(f"실제 데이터가 있는 날짜: {len(unique_dates)}일")
        print(f"빠진 날짜 수: {len(missing_dates)}일")
        
        if missing_dates:
            print(f"\n빠진 날짜들:")
            for i, missing_date in enumerate(missing_dates, 1):
                print(f"  {i:3d}. {missing_date}")
                
            # 연속된 빠진 날짜 구간 찾기
            consecutive_gaps = find_consecutive_gaps(missing_dates)
            if consecutive_gaps:
                print(f"\n연속으로 빠진 날짜 구간:")
                for gap in consecutive_gaps:
                    if gap['start'] == gap['end']:
                        print(f"  - {gap['start']} (1일)")
                    else:
                        print(f"  - {gap['start']} ~ {gap['end']} ({gap['days']}일)")
        else:
            print("✅ 빠진 날짜가 없습니다!")
        
        return {
            'missing_dates': missing_dates,
            'total_days': len(date_range),
            'actual_days': len(unique_dates),
            'missing_count': len(missing_dates),
            'start_date': start_date,
            'end_date': end_date
        }
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None

def find_consecutive_gaps(missing_dates):
    """
    연속된 빠진 날짜 구간을 찾는 함수
    """
    if not missing_dates:
        return []
    
    gaps = []
    current_start = missing_dates[0]
    current_end = missing_dates[0]
    
    for i in range(1, len(missing_dates)):
        # 이전 날짜와 연속인지 확인
        if missing_dates[i] == current_end + timedelta(days=1):
            current_end = missing_dates[i]
        else:
            # 연속이 끊어짐, 현재 구간 저장
            gaps.append({
                'start': current_start,
                'end': current_end,
                'days': (current_end - current_start).days + 1
            })
            current_start = missing_dates[i]
            current_end = missing_dates[i]
    
    # 마지막 구간 저장
    gaps.append({
        'start': current_start,
        'end': current_end,
        'days': (current_end - current_start).days + 1
    })
    
    return gaps

def analyze_directory(directory_path, timestamp_column='timestamp'):
    """
    디렉토리 내 모든 CSV 파일의 빠진 날짜 분석
    
    Parameters:
    directory_path (str): CSV 파일들이 있는 디렉토리 경로
    timestamp_column (str): 날짜 컬럼명
    """
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"디렉토리 '{directory_path}'에서 CSV 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(csv_files)}개의 CSV 파일을 분석합니다.")
    print("="*60)
    
    all_results = {}
    
    for file_path in csv_files:
        result = find_missing_dates(file_path, timestamp_column)
        if result:
            all_results[os.path.basename(file_path)] = result
    
    # 전체 요약 출력
    print("\n" + "="*60)
    print("📊 전체 요약")
    print("="*60)
    
    if all_results:
        print(f"{'파일명':<30} {'전체일수':<8} {'실제일수':<8} {'빠진일수':<8} {'비율':<8}")
        print("-" * 70)
        
        total_expected = 0
        total_actual = 0
        total_missing = 0
        
        for filename, result in all_results.items():
            ratio = (result['missing_count'] / result['total_days']) * 100
            print(f"{filename[:29]:<30} {result['total_days']:<8} {result['actual_days']:<8} {result['missing_count']:<8} {ratio:<7.1f}%")
            
            total_expected += result['total_days']
            total_actual += result['actual_days']
            total_missing += result['missing_count']
        
        print("-" * 70)
        overall_ratio = (total_missing / total_expected) * 100 if total_expected > 0 else 0
        print(f"{'전체 합계':<30} {total_expected:<8} {total_actual:<8} {total_missing:<8} {overall_ratio:<7.1f}%")

def save_missing_dates_report(directory_path, output_file="missing_dates_report.csv", timestamp_column='timestamp'):
    """
    빠진 날짜 분석 결과를 CSV 파일로 저장
    """
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    report_data = []
    
    for file_path in csv_files:
        result = find_missing_dates(file_path, timestamp_column)
        if result and result['missing_dates']:
            for missing_date in result['missing_dates']:
                report_data.append({
                    'filename': os.path.basename(file_path),
                    'missing_date': missing_date,
                    'start_date': result['start_date'],
                    'end_date': result['end_date']
                })
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_file, index=False)
        print(f"\n💾 빠진 날짜 리포트 저장: {output_file}")
    else:
        print("\n✅ 모든 파일에 빠진 날짜가 없습니다!")


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
    dir_path = os.path.join(base_directory, "test")  # test 폴더 경로
    analyze_directory(dir_path, "timestamp")

    # 상세 리포트 저장
    # save_detailed_report(stats)
    # save_missing_dates_report(directory_path, "missing_dates_report.csv", "timestamp")