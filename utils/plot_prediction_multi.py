#%%
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import random

input_dummy = np.random.rand(5)
pred_dummy = np.random.rand(24)
gt_dumpy = np.random.rand(24)

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

    x = np.arange(len(np.concatenate([input_dummy, pred_dummy])))
    for spine in ax.spines.values():
            spine.set_linewidth(2)  # 원하는 굵기로 설정 (예: 2)
    ax.plot(x, np.concatenate([input_dummy, pred_dummy]), color='blue', label='Input')
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
    0.5, 0.04, 'Timestamp', ha='center', va='center', fontdict={'fontsize': 60}
)

fig.text(
    0.08, 0.5, 'MAPE [%]', ha='center', va='center', rotation='vertical', fontdict={'fontsize': 60}
)
fig.legend(
     category
)
plt.show()
                 
# %%
