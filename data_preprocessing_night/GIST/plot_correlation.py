import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the root directory (assuming the root is two levels up from the current file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

data_name = 'GIST'
data_dir = os.path.join(project_root, 'data/GIST_dataset/uniform_format_data')

data_list = os.listdir(data_dir)
data_list = [file for file in data_list if file.endswith('.csv')]

# List of features to plot
features = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Wind_Speed']
colors = ['blue', 'green', 'red', 'purple']
titles = ['Active Power [kW] vs Global Horizontal Radiation [w/m²]',
          'Active Power [kW] vs Weather Temperature [℃]',
          'Active Power [kW] vs Weather Relative Humidity [%]',
          'Active Power [kW] vs Wind Speed [m/s]']

# Iterate over each file in data_list and visualize separately
for file in tqdm(data_list, desc='Processing data files'):
    data_path = os.path.join(data_dir, file)
    df = pd.read_csv(data_path)
    df.drop(columns=['timestamp'], inplace=True)

    # Normalize Active_Power by dividing by the maximum value for each site
    df['Normalized_Active_Power'] = df['Active_Power'] / df['Active_Power'].max()

    # Calculate the correlation between Active_Power and other features
    correlations = df.corr()['Active_Power']

    # Plot the relationship between Active_Power and other features
    plt.figure(figsize=(18, 12))

    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i+1)
        corr_value = correlations.get(feature, 0)  # Get correlation, default to 0 if not present
        plt.scatter(df[feature], df['Normalized_Active_Power'], color=colors[i], marker='x', alpha=0.6)
        plt.xlabel(f'{feature}')
        plt.ylabel('Active Power (kW)')
        plt.title(f'{titles[i]} (Correlation: {corr_value:.2f})')
        plt.grid(True, alpha=0.3)

    plt.subplots_adjust(hspace=0.5, top=0.93)
    plt.suptitle(f'Feature Analysis vs Normalized Active Power for {file}', fontsize=20, y=1)

    # Save the plot as an image file (e.g., PNG format) with a tight bounding box
    plot_filename = f'{file.split(".")[0]}_feature_vs_power_plot.png'
    plot_path = os.path.join('./raw_info', plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()

    print(f"Plot saved for {file} at: {plot_path}")
