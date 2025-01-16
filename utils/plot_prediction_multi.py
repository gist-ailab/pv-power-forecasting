#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
import numpy as np
import random

input_dummy = np.random.rand(3)
pred_dummy = np.random.rand(24)
gt_dummy = np.random.rand(24)

plot_order = ['DKASC', 'DKASC', 'DKASC', 'DKASC',
              'GIST', 'GIST', 'California', 'California',
              'Germany', 'Germany', 'Georgia', 'Georgia',
              'Miryang', 'Miryang', 'UK', 'UK']

plt.style.use('default')

plt.rcParams['font.size'] = 44
plt.rcParams['font.family'] = 'Liberation Serif'

fig = plt.figure(figsize=(50, 30)) # 50?

outer_grid = gridspec.GridSpec(4, 4, figure=fig, hspace=0.2, wspace=0.2)

for idx, region in enumerate(plot_order):
    show_ylabel = idx % 4 == 0 # 왼쪽 열에 y축 레이블 표시
    ax = fig.add_subplot(outer_grid[idx])

   
    for spine in ax.spines.values():
            spine.set_linewidth(3)  # 원하는 굵기로 설정 (예: 2)
    dash, = ax.plot(input_dummy, '--', color='#4B6D41', label='Input', linewidth=3)
    dash_pattern = [8, 4]  # 선 길이 10, 빈 간격 5
    dash.set_dashes(dash_pattern)
    ax.plot(np.arange(len(input_dummy), len(input_dummy) + len(gt_dummy)), gt_dummy, color='#4B6D41', label='Ground Truth', linewidth=3)
    ax.plot(np.arange(len(input_dummy), len(input_dummy) + len(gt_dummy)), pred_dummy, color='#77202E', label='Prediction', linewidth=3)

    ax.set_title(region, pad=20, fontweight='bold')
    
    # if show_ylabel:
    #     ax.set_ylabel ('MAPE [%]', rotation=90, va='center', ha='center', labelpad=30)
    
    if idx in [12, 13, 14, 15]:
        ax.tick_params(axis='x', length=10, width=2, pad=7)
        # ax.set_xlabel('Timestamp')
    else: 
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    
   
    ax.tick_params(axis='y', direction='in', length=10, width=2, pad=7)

    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

fig.text(
    0.5, 0.95, 'Timestamp', ha='center', va='center', fontdict={'fontsize': 70}
)

fig.text(
    0.08, 0.5, 'MAPE [%]', ha='center', va='center', rotation='vertical', fontdict={'fontsize': 70}
)

colors = ['#5E6064', '#B31A23', '#5E6064']
fig_legend = fig.legend(
     labels=['Input', 'Prediction', 'Ground Truth'],
     loc='lower center',
     bbox_to_anchor=(0.5, -0.03),
     ncols=3,
     edgecolor='black',
    fontsize=60,
)

fig_legend.get_frame().set_linewidth(2)


plt.show()
                 
# %%
