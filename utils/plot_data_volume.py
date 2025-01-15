#%%
import matplotlib.pyplot as plt
import numpy as np



# Data
months = ['3months', '6months', '9months', '12months']
data_1_16 = [7.1363, 7.1455, 6.7053, 6.4413]
data_1_8 = [6.6736, 6.8629, 6.468, 6.2141]
data_entire = [6.1724, 6.1783, 5.8742, 5.7292]

# Set width of each bar and positions of bars
width = 0.2
x = np.arange(len(months))

# Create figure and axis
plt.figure(figsize=(10, 6))
plt.style.use('default')
plt.rcParams['font.size'] = 22  
plt.rcParams['font.family'] = 'Liberation Serif'

plt.bar(x - width, data_1_16, width, label='1/16', color='#BEBDBD')
plt.bar(x, data_1_8, width, label='1/8', color='#103B58')
plt.bar(x + width, data_entire, width, label='Entire Volume', color='#B31A23')

# Customize plot
plt.ylabel('MAPE [%]')
plt.xlabel('Train Set Size of Target Data', labelpad=15)
# plt.title('GIST')
plt.xticks(x, months)
plt.tick_params(axis='y', direction='in', length=10, width=2, pad=7)
plt.tick_params(axis='x', bottom=False)
plt.legend(title='Source Data Size', handlelength=2, edgecolor='black')

# Set y-axis limits
plt.ylim(5, 8)
plt.yticks(np.arange(5, 9, 1))

# Add grid
plt.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)

# Set spines linewidth (변의 굵기 조정)
ax = plt.gca()  # Get current axis
for spine in ax.spines.values():
    spine.set_linewidth(2)  # 원하는 굵기 설정 (예: 2)

# Adjust layout
plt.tight_layout()

# Save plots
plt.show()
# %%
