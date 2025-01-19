import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio

# notebook에서 인라인으로 표시되도록 설정
pio.renderers.default = "notebook"

# 데이터 정의
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
data = {
    'GIST': [1.6293, 1.6008, 1.6438, 1.6514, 1.7037, 1.8012, 2.0653, 2.7289, 5.4627],
    'Germany': [1.466, 1.0643, 1.0794, 1.0956, 1.1274, 1.1795, 1.298, 1.5509, 2.4999],
    'Miryang': [3.1986, 2.3044, 2.3204, 2.3256, 2.3348, 2.3628, 2.4544, 2.7459, 3.4867],
    'California': [5.2096, 5.2434, 5.2638, 5.2827, 5.3013, 5.3254, 5.3791, 5.1777, 5.1154],
    'Georgia': [6.4877, 6.4395, 6.463, 6.4671, 6.49, 6.506, 6.4227, 6.2644, 6.1461],
    'UK': [9.0415, 8.3173, 8.308, 8.2876, 8.3863, 8.5353, 8.5398, 7.4664, 7.8598]
}

def assign_highlight_colors(values):
    sorted_indices = np.argsort(values)
    min_idx = sorted_indices[0]
    second_min_idx = sorted_indices[1]
    colors = ['#5E6064'] * len(values)  
    colors[second_min_idx] = '#6A282C'
    colors[min_idx] = '#B31A23'
    return colors

# 서브플롯 생성
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('<b>GIST</b>', '<b>Germany</b>', '<b>Miryang</b>', 
                   '<b>California</b>', '<b>Georgia</b>', '<b>UK</b>'),
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

# 각 지역별 데이터 시각화
plot_order = ['GIST', 'Germany', 'Miryang', 'California', 'Georgia', 'UK']
row_col_pairs = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]

for (region, (row, col)) in zip(plot_order, row_col_pairs):
    values = data[region]
    colors = assign_highlight_colors(values)
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            showlegend=False
        ),
        row=row, col=col
    )

# 레이아웃 업데이트
fig.update_layout(
    height=1200,
    width=1000,
    font=dict(
        family="Liberation Serif",
        size=20
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
)

# x축, y축 업데이트
fig.update_xaxes(
    title_text="Number of freeze layers",
    tickangle=45,
    showgrid=True,
    gridcolor='lightgray',
)

fig.update_yaxes(
    title_text="MAPE [%]",
    showgrid=True,
    gridcolor='lightgray',
)

# y축 범위 설정
y_ranges = {
    'GIST': [1.5, 5.5],
    'Germany': [1.0, 2.5],
    'Miryang': [2.0, 3.5],
    'California': [5.1, 5.4],
    'Georgia': [6.0, 6.6],
    'UK': [7.0, 9.2]
}

for idx, region in enumerate(plot_order):
    row = idx // 2 + 1
    col = idx % 2 + 1
    fig.update_yaxes(range=y_ranges[region], row=row, col=col)

fig.show()