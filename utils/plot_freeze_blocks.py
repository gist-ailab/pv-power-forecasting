# 
# %%
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import matplotlib.patches as patches
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data
categories = ['all', '1', '2', '3', '4', '5', '6', '7', '8', 'zero-shot']
data = {
    'GIST': [1.6293, 1.6008, 1.6438, 1.6514, 1.7037, 1.8012, 2.0653, 2.7289, 5.4627, 6.127],
    'Miryang': [3.1986, 2.3044, 2.3204, 2.3256, 2.3348, 2.3628, 2.4544, 2.7459, 3.4867, 3.7475],
    'Konstanz': [1.466, 1.0643, 1.0794, 1.0956, 1.1274, 1.1795, 1.298, 1.5509, 2.4999, 3.2284],
    'California': [5.2096, 5.2434, 5.2638, 5.2827, 5.3013, 5.3254, 5.3791, 5.1777, 5.1154, 5.2956],
    'Georgia': [6.4877, 6.4395, 6.463, 6.4671, 6.49, 6.506, 6.4227, 6.2644, 6.1461, 6.1932],
    'UK': [9.0415, 8.3173, 8.308, 8.2876, 8.3863, 8.5353, 8.5398, 7.4664, 7.8598, 8.6286]
}

sps1, sps2, sps3, sps4, sps5, sps6 = GridSpec(2, 3)

bax = brokenaxes(ylims=((1.5, 2), (5, 6.5)), subplot_spec=sps1)
bax.bar(categories, data['GIST'], label='GIST', color='blue')

from matplotlib.ticker import FixedLocator

fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=300, constrained_layout=True)

ax = axs[0, 0]
ax.bar(categories, data['GIST'], color='blue', alpha=0.8)

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.colors as mcolors

def make_color_gradient(n_colors, start_hex='#fdbb2d', end_hex="#22c1c3"):
    """Create a smooth color gradient between two colors"""
    start_rgb = np.array(mcolors.to_rgb(start_hex))
    end_rgb = np.array(mcolors.to_rgb(end_hex))
    
    # Create logarithmic spacing for more natural color progression
    t = np.logspace(-0.5, 0, n_colors)
    t = (t - t.min()) / (t.max() - t.min())
    
    colors = [tuple(start_rgb * (1-i) + end_rgb * i) for i in t]
    return colors

def make_broken_axis(fig, gs, data, title, break_points, y_ranges):
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    x = np.arange(len(categories))
    width = 0.7
    
    # Create smooth color gradient
    colors = make_color_gradient(len(x))
    
    # Plot bars with gradient colors
    for i, (value, color) in enumerate(zip(data, colors)):
        ax1.bar(x[i], value, width, color=color)
        ax2.bar(x[i], value, width, color=color)

    # Set different scales
    ax1.set_ylim(y_ranges[1])  # upper plot
    ax2.set_ylim(y_ranges[0])  # lower plot
    
    # Hide the spines between ax1 and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    
    # Add broken axis marks
    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='gray', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    # Set title and labels
    ax1.set_title(title, pad=20, fontweight='bold')
    ax2.set_xlabel('# of freeze')
    
    # Set x-ticks
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    
    # Add grid
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)


# Data
categories = ['fully-finetune', '1', '2', '3', '4', '5', '6', '7', '8', 'zero-shot']
data = {
    'GIST': [1.6293, 1.6008, 1.6438, 1.6514, 1.7037, 1.8012, 2.0653, 2.7289, 5.4627, 6.127],
    'Miryang': [3.1986, 2.3044, 2.3204, 2.3256, 2.3348, 2.3628, 2.4544, 2.7459, 3.4867, 3.7475],
    'Konstanz': [1.466, 1.0643, 1.0794, 1.0956, 1.1274, 1.1795, 1.298, 1.5509, 2.4999, 3.2284],
    'California': [5.2096, 5.2434, 5.2638, 5.2827, 5.3013, 5.3254, 5.3791, 5.1777, 5.1154, 5.2956],
    'Georgia': [6.4877, 6.4395, 6.463, 6.4671, 6.49, 6.506, 6.4227, 6.2644, 6.1461, 6.1932],
    'UK': [9.0415, 8.3173, 8.308, 8.2876, 8.3863, 8.5353, 8.5398, 7.4664, 7.8598, 8.6286]
}

# Plot settings for each region
plot_settings = {
    'GIST': {
        'break_points': [1.8, 2.6],
        'y_ranges': [(1.5, 1.8), (2.6, 6.5)]
    },
    'Miryang': {
        'break_points': [2.5, 2.7],
        'y_ranges': [(2.25, 2.5), (2.7, 4.0)]
    },
    'Konstanz': {
        'break_points': [1.3, 2.4],
        'y_ranges': [(1.0, 1.3), (2.4, 3.5)]
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

# Create figures
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Create figure
fig = plt.figure(figsize=(15, 10))
outer_grid = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

for idx, (region, values) in enumerate(data.items()):
    settings = plot_settings[region]
    if settings['break_points']:
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=outer_grid[idx],
                                                    height_ratios=[1, 1],
                                                    hspace=0.1)
        make_broken_axis(fig, inner_grid, values, region,
                        settings['break_points'],
                        settings['y_ranges'])
    else:
        ax = fig.add_subplot(outer_grid[idx])
        x = np.arange(len(categories))
        
        # Create smooth color gradient for regular subplot
        colors = make_color_gradient(len(x))
        
        for i, (value, color) in enumerate(zip(values, colors)):
            ax.bar(x[i], value, width=0.7, color=color)
            
        ax.set_title(region, pad=10, fontweight='bold')
        ax.set_xlabel('# of freeze')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylim(settings['y_ranges'][0])
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.savefig('region_metrics_smooth_color.pdf', bbox_inches='tight', dpi=300)
plt.savefig('region_metrics_smooth_color.png', bbox_inches='tight', dpi=300)
plt.close()# %%

# %%
# %%
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle, PathPatch
# from matplotlib.path import Path
# import matplotlib.patches as patches
# from brokenaxes import brokenaxes
# from matplotlib.gridspec import GridSpec

# plt.rcParams.update({
#     'font.size': 10,
#     'axes.labelsize': 11,
#     'axes.titlesize': 11,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'figure.figsize': (12, 8),
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linestyle': '--',
#     'font.family': 'sans-serif',
#     'axes.axisbelow': True,
#     'axes.spines.top': False,
#     'axes.spines.right': False,
# })

# # Data
# categories = ['all', '1', '2', '3', '4', '5', '6', '7', '8', 'zero-shot']
# data = {
#     'GIST': [1.6293, 1.6008, 1.6438, 1.6514, 1.7037, 1.8012, 2.0653, 2.7289, 5.4627, 6.127],
#     'Miryang': [3.1986, 2.3044, 2.3204, 2.3256, 2.3348, 2.3628, 2.4544, 2.7459, 3.4867, 3.7475],
#     'Konstanz': [1.466, 1.0643, 1.0794, 1.0956, 1.1274, 1.1795, 1.298, 1.5509, 2.4999, 3.2284],
#     'California': [5.2096, 5.2434, 5.2638, 5.2827, 5.3013, 5.3254, 5.3791, 5.1777, 5.1154, 5.2956],
#     'Georgia': [6.4877, 6.4395, 6.463, 6.4671, 6.49, 6.506, 6.4227, 6.2644, 6.1461, 6.1932],
#     'UK': [9.0415, 8.3173, 8.308, 8.2876, 8.3863, 8.5353, 8.5398, 7.4664, 7.8598, 8.6286]
# }

# sps1, sps2, sps3, sps4, sps5, sps6 = GridSpec(2, 3)

# bax = brokenaxes(ylims=((1.5, 2), (5, 6.5)), subplot_spec=sps1)
# bax.bar(categories, data['GIST'], label='GIST', color='blue')

# from matplotlib.ticker import FixedLocator

# fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=300, constrained_layout=True)

# ax = axs[0, 0]
# ax.bar(categories, data['GIST'], color='blue', alpha=0.8)

# #%%
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import gridspec

# def make_broken_axis(fig, gs, data, title, break_points, y_ranges):
#     # Create two subplots
#     ax1 = fig.add_subplot(gs[0])
#     ax2 = fig.add_subplot(gs[1])
    
#     x = np.arange(len(categories))
#     width = 0.7
    
#     # Plot the same data on both axes
#     ax1.bar(x, data, width, color='#2171b5', alpha=0.8)
#     ax2.bar(x, data, width, color='#2171b5', alpha=0.8)

#     # Set different scales
#     ax1.set_ylim(y_ranges[1])  # upper plot
#     ax2.set_ylim(y_ranges[0])  # lower plot
    
#     # Hide the spines between ax1 and ax2
#     ax1.spines.bottom.set_visible(False)
#     ax2.spines.top.set_visible(False)
#     ax1.xaxis.tick_top()
#     ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    
#     # Add broken axis marks
#     d = 0.01
#     kwargs = dict(transform=ax1.transAxes, color='gray', clip_on=False)
#     ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#     ax1.plot((1-d, 1+d), (-d, +d), **kwargs)      # top-right diagonal
    
#     kwargs.update(transform=ax2.transAxes)
#     ax2.plot((-d, +d), (1-d, 1+d), **kwargs)      # bottom-left diagonal
#     ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)    # bottom-right diagonal
    
#     # Set title and labels
#     ax1.set_title(title, pad=20, fontweight='bold')
#     ax2.set_xlabel('# of freeze')
    
#     # Set x-ticks
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(categories, rotation=45, ha='right')
    
#     # Add grid
#     ax1.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
#     ax2.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
#     # Remove top and right spines
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax2.spines['right'].set_visible(False)

# # Data
# categories = ['all', '1', '2', '3', '4', '5', '6', '7', '8', 'zero-shot']
# data = {
#     'GIST': [1.6293, 1.6008, 1.6438, 1.6514, 1.7037, 1.8012, 2.0653, 2.7289, 5.4627, 6.127],
#     'Miryang': [3.1986, 2.3044, 2.3204, 2.3256, 2.3348, 2.3628, 2.4544, 2.7459, 3.4867, 3.7475],
#     'Konstanz': [1.466, 1.0643, 1.0794, 1.0956, 1.1274, 1.1795, 1.298, 1.5509, 2.4999, 3.2284],
#     'California': [5.2096, 5.2434, 5.2638, 5.2827, 5.3013, 5.3254, 5.3791, 5.1777, 5.1154, 5.2956],
#     'Georgia': [6.4877, 6.4395, 6.463, 6.4671, 6.49, 6.506, 6.4227, 6.2644, 6.1461, 6.1932],
#     'UK': [9.0415, 8.3173, 8.308, 8.2876, 8.3863, 8.5353, 8.5398, 7.4664, 7.8598, 8.6286]
# }

# # Plot settings for each region
# plot_settings = {
#     'GIST': {
#         'break_points': [2.0, 2.6],
#         'y_ranges': [(1.5, 2.0), (2.6, 6.5)]
#     },
#     'Miryang': {
#         'break_points': [2.5, 2.7],
#         'y_ranges': [(2.25, 2.5), (2.7, 4.0)]
#     },
#     'Konstanz': {
#         'break_points': [1.55, 2.2],
#         'y_ranges': [(1.0, 1.55), (2.2, 3.5)]
#     },
#     'California': {
#         'break_points': None,
#         'y_ranges': [(5.1, 5.4)]
#     },
#     'Georgia': {
#         'break_points': None,
#         'y_ranges': [(6.1, 6.6)]
#     },
#     'UK': {
#         'break_points': None,
#         'y_ranges': [(7.4, 9.2)]
#     }
# }

# # Create figures
# plt.style.use('default')
# plt.rcParams.update({
#     'font.size': 10,
#     'axes.labelsize': 11,
#     'axes.titlesize': 11,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10
# })

# # Create subplot grid
# fig = plt.figure(figsize=(15, 10))
# outer_grid = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

# for idx, (region, values) in enumerate(data.items()):
#     settings = plot_settings[region]
#     if settings['break_points']:
#         # Create broken axis subplot
#         inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
#                                                     subplot_spec=outer_grid[idx],
#                                                     height_ratios=[1, 1],
#                                                     hspace=0.1)
#         make_broken_axis(fig, inner_grid, values, region,
#                         settings['break_points'],
#                         settings['y_ranges'])
#     else:
#         # Create regular subplot
#         ax = fig.add_subplot(outer_grid[idx])
#         x = np.arange(len(categories))
#         ax.bar(x, values, width=0.7, color='#2171b5', alpha=0.8)
#         ax.set_title(region, pad=10, fontweight='bold')
#         ax.set_xlabel('# of freeze')
#         ax.set_xticks(x)
#         ax.set_xticklabels(categories, rotation=45, ha='right')
#         ax.set_ylim(settings['y_ranges'][0])
#         ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

# plt.savefig('region_metrics_broken.pdf', bbox_inches='tight', dpi=300)
# plt.savefig('region_metrics_broken.png', bbox_inches='tight', dpi=300)
# plt.close()