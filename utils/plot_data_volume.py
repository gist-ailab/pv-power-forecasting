#%%
import matplotlib.pyplot as plt
import numpy as np

# Data
months = ['6months', '9months', '12months']
data_1_16 = [7.1455, 6.7053, 6.4413]
data_1_8 = [6.8629, 6.468, 6.2141]
data_entire = [6.1783, 5.8742, 5.7292]

# Set width of each bar and positions of bars
width = 0.25
x = np.arange(len(months))

# Create figure and axis
plt.figure(figsize=(10, 6))
plt.style.use('default')

# Create bars
# plt.bar(x - width, data_1_16, width, label='1/16', color='#3191C7')
# plt.bar(x, data_1_8, width, label='1/8', color='#1A73A5')
# plt.bar(x + width, data_entire, width, label='Entire Volume', color='#0C527D')
plt.bar(x - width, data_1_16, width, label='1/16', color='#3191C7')
plt.bar(x, data_1_8, width, label='1/8', color='#2C3E50')
plt.bar(x + width, data_entire, width, label='Entire Volume', color='#27AE60')

# Customize plot
plt.ylabel('MAPE')
plt.xlabel('Train Set Size of Target Data', labelpad=15)
plt.title('GIST')
plt.xticks(x, months)
plt.legend(title='Source Data Size')

# Set y-axis limits
plt.ylim(5, 8)
plt.yticks(np.arange(5, 9, 1))

# Add grid
plt.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save plots
plt.savefig('gist_plot1.png', dpi=300, bbox_inches='tight')
plt.savefig('gist_plot1.pdf', dpi=300, bbox_inches='tight')
plt.close()
# %%
