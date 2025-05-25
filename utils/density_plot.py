import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_and_process_data(file_path):
    """CSV 파일을 로드하고 Active_Power 컬럼을 추출"""
    try:
        df = pd.read_csv(file_path)
        if 'Active_Power' in df.columns:
            # NaN 값 제거
            active_power = df['Active_Power'].dropna()
            return active_power
        else:
            print(f"Warning: 'Active_Power' column not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_data(data, method='minmax'):
    """데이터 정규화"""
    data_array = np.array(data).reshape(-1, 1)
    
    if method == 'minmax':
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data_array).flatten()
    elif method == 'zscore':
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data_array).flatten()
    else:
        raise ValueError("Method should be 'minmax' or 'zscore'")
    
    return normalized

def plot_density_comparison(csv_directory, output_dir=None):
    """
    지정된 디렉토리의 모든 CSV 파일에 대해 Active_Power의 density plot 비교
    
    Parameters:
    csv_directory (str): CSV 파일들이 있는 디렉토리 경로
    output_dir (str): 그래프를 저장할 디렉토리 (None이면 저장하지 않음)
    """
    
    # CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # 데이터 로드
    data_dict = {}
    for file_path in csv_files:
        file_name = os.path.basename(file_path).replace('.csv', '')
        active_power = load_and_process_data(file_path)
        if active_power is not None and len(active_power) > 0:
            data_dict[file_name] = active_power
    
    if not data_dict:
        print("No valid data found")
        return
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Active Power Density Plots Comparison', fontsize=16, fontweight='bold')
    
    # 색상 팔레트 설정
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    # 1. 원본 데이터
    ax1 = axes[0, 0]
    for i, (name, data) in enumerate(data_dict.items()):
        ax1.hist(data, bins=50, alpha=0.6, density=True, label=name, color=colors[i])
        # KDE 추가
        data_clean = data[np.isfinite(data)]
        if len(data_clean) > 1:
            sns.kdeplot(data=data_clean, ax=ax1, color=colors[i], linewidth=2)
    
    ax1.set_title('Original Data')
    ax1.set_xlabel('Active Power')
    ax1.set_ylabel('Density')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Min-Max Normalization
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(data_dict.items()):
        normalized_data = normalize_data(data, 'minmax')
        ax2.hist(normalized_data, bins=50, alpha=0.6, density=True, label=name, color=colors[i])
        # KDE 추가
        normalized_clean = normalized_data[np.isfinite(normalized_data)]
        if len(normalized_clean) > 1:
            sns.kdeplot(data=normalized_clean, ax=ax2, color=colors[i], linewidth=2)
    
    ax2.set_title('Min-Max Normalization')
    ax2.set_xlabel('Normalized Active Power (0-1)')
    ax2.set_ylabel('Density')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Z-score Standardization
    ax3 = axes[1, 0]
    for i, (name, data) in enumerate(data_dict.items()):
        standardized_data = normalize_data(data, 'zscore')
        ax3.hist(standardized_data, bins=50, alpha=0.6, density=True, label=name, color=colors[i])
        # KDE 추가
        standardized_clean = standardized_data[np.isfinite(standardized_data)]
        if len(standardized_clean) > 1:
            sns.kdeplot(data=standardized_clean, ax=ax3, color=colors[i], linewidth=2)
    
    ax3.set_title('Z-score Standardization')
    ax3.set_xlabel('Standardized Active Power')
    ax3.set_ylabel('Density')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. 통계 정보 표시
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 통계 정보 테이블 생성
    stats_data = []
    for name, data in data_dict.items():
        stats = {
            'PV Array': name,
            'Count': len(data),
            'Mean': f"{data.mean():.2f}",
            'Std': f"{data.std():.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}"
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # 테이블 그리기
    table = ax4.table(cellText=stats_df.values,
                     colLabels=stats_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('Statistics Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 그래프 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'active_power_density_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_dir}")
    
    plt.show()
    
    # 개별 정규화 방법별 상세 비교
    plot_detailed_comparison(data_dict, output_dir)

def plot_detailed_comparison(data_dict, output_dir=None):
    """정규화 방법별 상세 비교 그래프"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Detailed Normalization Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    # Min-Max Normalization 상세
    ax1 = axes[0]
    for i, (name, data) in enumerate(data_dict.items()):
        normalized_data = normalize_data(data, 'minmax')
        sns.kdeplot(data=normalized_data, ax=ax1, label=name, color=colors[i], linewidth=2)
    
    ax1.set_title('Min-Max Normalization (KDE only)')
    ax1.set_xlabel('Normalized Active Power (0-1)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Z-score Standardization 상세
    ax2 = axes[1]
    for i, (name, data) in enumerate(data_dict.items()):
        standardized_data = normalize_data(data, 'zscore')
        sns.kdeplot(data=standardized_data, ax=ax2, label=name, color=colors[i], linewidth=2)
    
    ax2.set_title('Z-score Standardization (KDE only)')
    ax2.set_xlabel('Standardized Active Power')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'detailed_normalization_comparison.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

# 사용 예시
if __name__ == "__main__":
    # CSV 파일들이 있는 디렉토리 경로를 지정하세요
    csv_directory = "/home/bak/Projects/pv-power-forecasting/data/GIST/processed_data_all"  # 여기에 실제 경로를 입력하세요
    output_directory = "/home/bak/Projects/pv-power-forecasting/data/GIST/output_plots"  # 그래프를 저장할 디렉토리 (선택사항)
    
    # 함수 실행
    plot_density_comparison(csv_directory, output_directory)