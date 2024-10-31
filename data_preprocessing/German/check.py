import pandas as pd
import os

# 폴더 경로 지정
# folder_path = '../../data/Germany_Household_Data/preprocessed'  # 원하는 폴더 경로로 변경 가능
# folder_path = '../../data/OEDI/2107(Arbuckle_California)/preprocessed'
# folder_path = '../../data/OEDI/9069(Georgia)/preprocessed'
folder_path = '../../data/UK_data/preprocessed'

# 해당 폴더 내 모든 CSV 파일 가져오기
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

# 파일별 날짜 개수를 저장할 딕셔너리
date_counts = {}

# 데이터프레임들을 저장할 리스트
dfs = []

# 각 파일을 불러와 날짜 개수와 데이터프레임 저장
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    # timestamp를 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 날짜만 추출하고 유일한 날짜 수 계산
    unique_dates = df['timestamp'].dt.date.nunique()
    date_counts[file_name] = unique_dates

    # 데이터프레임 리스트에 추가 (날짜만 포함하여 join에 사용)
    dfs.append(df[['timestamp']].copy())

# 파일명을 기준으로 정렬하여 날짜 수 출력
print("파일별 날짜 수:")
for file_name, count in sorted(date_counts.items()):
    print(f"{file_name}: {count}일")

# 여러 데이터프레임을 inner join으로 병합
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='timestamp', how='inner')

# 병합된 데이터의 날짜 개수 출력
merged_date_count = merged_df['timestamp'].dt.date.nunique()
print(f"\n병합된 데이터의 총 날짜 수: {merged_date_count}일")