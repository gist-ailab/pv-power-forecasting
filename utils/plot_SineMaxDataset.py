import sys
import os

# 현재 스크립트 위치를 기준으로 data_provider 경로 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../"))
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np
from data_provider.data_loader import SinMaxDataset

seq_len, label_len, pred_len = 24, 0, 12  # 하루 단위 + 12시간 예측
total_len = 10000  # 총 데이터 길이 (약 416일)

# 데이터셋 초기화
dataset = SinMaxDataset(seq_len, label_len, pred_len, total_len)

# 1. 그래프 그리기
# 그래프 그리기 (2~3일 출력)
days_to_plot = 3  # 3일만 출력
hours_to_plot = days_to_plot * 24  # 시간으로 변환
x_data = np.arange(hours_to_plot)  # 0부터 n일까지의 시간
y_data = dataset.y_data[:hours_to_plot].flatten()  # y_data에서 해당 구간만 가져오기

plt.figure(figsize=(10, 4))
plt.plot(x_data, y_data, label=f"{days_to_plot} Days of $y = \\max(0, \\sin(\\theta))$")
plt.title(f"{days_to_plot}-Day View of $y = \\max(0, \\sin(\\theta))$")
plt.xlabel("Hours")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

# 2. __getitem__ 출력 확인
sample_idx = 10  # 확인하고 싶은 데이터셋의 인덱스
output = dataset[sample_idx]

# __getitem__ 출력 확인
seq_x, seq_y, seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds = output

print(f"seq_x (shape: {seq_x.shape}):\n{seq_x}")
print(f"seq_y (shape: {seq_y.shape}):\n{seq_y}")
print(f"seq_x_mark (shape: {seq_x_mark.shape}):\n{seq_x_mark}")
print(f"seq_y_mark (shape: {seq_y_mark.shape}):\n{seq_y_mark}")
print(f"site (shape: {site.shape}):\n{site}")
print(f"seq_x_ds (shape: {seq_x_ds.shape}):\n{seq_x_ds}")
print(f"seq_y_ds (shape: {seq_y_ds.shape}):\n{seq_y_ds}")
