
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.colors as mcolors
def get_ylabel_position(fig, ax):
    """
    모든 그래프에서 일관된 y축 레이블 위치를 계산합니다.
    """
    fig.canvas.draw()  # 레이아웃 업데이트
    bbox = ax.get_position()
    # 모든 그래프에서 동일한 절대 위치 사용
    return 0.08, bbox.y0 + bbox.height/2
def create_subplot_with_broken_axis(fig, gs, data, title, break_points, y_ranges, show_ylabel=True):
    """
    서브플롯을 생성하고 필요한 경우 축을 끊어서 표현하는 함수
    
    Parameters:
    - fig: matplotlib figure 객체
    - gs: GridSpec 객체
    - data: 그래프에 표시할 데이터
    - title: 서브플롯 제목
    - break_points: 축을 끊을 지점
    - y_ranges: y축의 범위
    - show_ylabel: y축 레이블 표시 여부
    """
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    x = np.arange(len(categories))
    width = 0.7
    
    # 최소값을 찾아서 해당 막대만 빨간색으로 표시
    min_idx = np.argmin(data)
    colors = ['#D3D3D3'] * len(data)  # 기본 색상은 연한 회색
    colors[min_idx] = '#FF4B4B'  # 최소값은 빨간색
    
    # 막대 그래프 그리기
    for i, (value, color) in enumerate(zip(data, colors)):
        ax1.bar(x[i], value, width, color=color)
        ax2.bar(x[i], value, width, color=color)
    
    # y축 범위 설정
    ax1.set_ylim(y_ranges[1])
    ax2.set_ylim(y_ranges[0])
    
    # 축이 끊어진 표시 추가
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    
    # 끊어진 부분 표시
    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='gray', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    # 제목과 레이블 설정
    ax1.set_title(title, pad=20, fontweight='bold')
    
    # y축 레이블 중앙 정렬
    if show_ylabel:
        # fig.canvas.draw()
        # x_pos, y_pos = get_ylabel_position(fig, ax2)
        x_pos, y_pos = calculate_mape_position(fig, [ax1, ax2])
        fig.text(x_pos, y_pos, 'MAPE', rotation=90, va='center')

        # ax1_pos = ax1.get_position()
        # ax2_pos = ax2.get_position()
        # ylabel_x = ax2_pos.x0 - 0.04
        # ylabel_y = (ax1_pos.y0 + ax1_pos.y1 + ax2_pos.y0 + ax2_pos.y1) / 4
        # fig.text(ylabel_x, ylabel_y, 'MAPE', rotation=90, va='center')
    
    # x축 설정
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    
    # 그리드 추가
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 불필요한 테두리 제거
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
def calculate_mape_position(fig, axes):
    """
    MAPE 레이블의 적절한 위치를 계산합니다.
    
    Parameters:
    - fig: matplotlib figure 객체
    - axes: 하나의 축 또는 두 개의 축(끊어진 그래프의 경우) 리스트
    
    Returns:
    - x_pos: MAPE 레이블의 x 좌표
    - y_pos: MAPE 레이블의 y 좌표
    """
    fig.canvas.draw()  # 레이아웃 업데이트를 위해 필요
    
    if isinstance(axes, list):  # 끊어진 그래프의 경우
        ax1, ax2 = axes
        bbox1 = ax1.get_position()
        bbox2 = ax2.get_position()
        # y 위치는 두 축의 중간점
        y_pos = (bbox1.y0 + bbox1.y1 + bbox2.y0 + bbox2.y1) / 4
    else:  # 일반 그래프의 경우
        bbox = axes.get_position()
        # y 위치는 축의 중간점
        y_pos = bbox.y0 + bbox.height / 2
    
    # 모든 그래프에 대해 동일한 x 위치 사용
    x_pos = 0.08  # 이 값은 필요에 따라 조정 가능
    
    return x_pos, y_pos
# 데이터 정의
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'zero-shot']
data = {
    'GIST': [1.6293, 1.6008, 1.6438, 1.6514, 1.7037, 1.8012, 2.0653, 2.7289, 5.4627, 6.127],
    'Konstanz': [1.466, 1.0643, 1.0794, 1.0956, 1.1274, 1.1795, 1.298, 1.5509, 2.4999, 3.2284],
    'Miryang': [3.1986, 2.3044, 2.3204, 2.3256, 2.3348, 2.3628, 2.4544, 2.7459, 3.4867, 3.7475],
    'California': [5.2096, 5.2434, 5.2638, 5.2827, 5.3013, 5.3254, 5.3791, 5.1777, 5.1154, 5.2956],
    'Georgia': [6.4877, 6.4395, 6.463, 6.4671, 6.49, 6.506, 6.4227, 6.2644, 6.1461, 6.1932],
    'UK': [9.0415, 8.3173, 8.308, 8.2876, 8.3863, 8.5353, 8.5398, 7.4664, 7.8598, 8.6286]
}

# 그래프 설정
plot_settings = {
    'GIST': {
        'break_points': [1.8, 2.6],
        'y_ranges': [(1.5, 1.8), (2.6, 6.5)]
    },
    'Konstanz': {
        'break_points': [1.3, 2.4],
        'y_ranges': [(1.0, 1.3), (2.4, 3.5)]
    },
    'Miryang': {
        'break_points': [2.5, 2.7],
        'y_ranges': [(2.25, 2.5), (2.7, 4.0)]
    },
    'California': {
        'break_points': None,
        'y_ranges': [(5.1, 5.4)]
    },
    'Georgia': {
        'break_points': None,
        'y_ranges': [(6.0, 6.6)]
    },
    'UK': {
        'break_points': None,
        'y_ranges': [(7, 9.2)]
    }
}

# 그래프 순서 정의
plot_order = ['GIST', 'Konstanz', 'Miryang', 'California', 'Georgia', 'UK']

# 플롯 스타일 설정
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# 메인 figure 생성
fig = plt.figure(figsize=(15, 12))
outer_grid = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)

# 서브플롯 생성
for idx, region in enumerate(plot_order):
    settings = plot_settings[region]
    values = data[region]
    show_ylabel = idx % 2 == 0  # 왼쪽 열에만 y축 레이블 표시
    
    if settings['break_points']:
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=outer_grid[idx],
                                                    height_ratios=[1, 1],
                                                    hspace=0.1)
        create_subplot_with_broken_axis(fig, inner_grid, values, region,
                                      settings['break_points'],
                                      settings['y_ranges'],
                                      show_ylabel)
    else:
        ax = fig.add_subplot(outer_grid[idx])
        
        # 최소값 강조
        min_idx = np.argmin(values)
        colors = ['#D3D3D3'] * len(values)
        colors[min_idx] = '#FF4B4B'
        
        x = np.arange(len(categories))
        for i, (value, color) in enumerate(zip(values, colors)):
            ax.bar(x[i], value, width=0.7, color=color)
        
        ax.set_title(region, pad=10, fontweight='bold')
        if show_ylabel:
            x_pos, y_pos = calculate_mape_position(fig, ax)
            fig.text(x_pos, y_pos, 'MAPE', rotation=90, va='center')
            # ax.yaxis.set_label_coords(-0.1, 0.5)
            # ax.set_ylabel('MAPE')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylim(settings['y_ranges'][0])
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# 공통 x축 레이블 추가
fig.text(0.5, 0.02, 'Common x-axis: Number of freeze layers', ha='center', va='center', fontsize=12)

# 저장
# plt.savefig('region_metrics_updated.pdf', bbox_inches='tight', dpi=300)
plt.savefig('1.png', bbox_inches='tight', dpi=300)
plt.close()
# %%
