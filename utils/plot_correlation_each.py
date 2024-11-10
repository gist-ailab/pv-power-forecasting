import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_feature_vs_active_power(data_dir, save_dir, features, colors, titles):
    """
    Plots the relationship between Active Power and various features for CSV files in a directory.

    Parameters:
    - data_dir (str): The directory containing the CSV files.
    - save_dir (str): The directory to save the plots.
    - features (list): The list of features to plot against Active Power.
    - colors (list): The list of colors for each plot.
    - titles (list): The list of titles for each plot.
    """
    data_list = os.listdir(data_dir)
    data_list = [file for file in data_list if file.endswith('.csv')]
    
    os.makedirs(save_dir, exist_ok=True)

    for file in tqdm(data_list, desc='Processing data files'):
        data_path = os.path.join(data_dir, file)
        df = pd.read_csv(data_path)
        
        # Drop 'timestamp' column if present
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)

        # Normalize Active_Power by dividing by the maximum value for each site
        df['Normalized_Active_Power'] = df['Active_Power'] / df['Active_Power'].max()

        # Calculate the correlation between Active_Power and other features
        correlations = df.corr()['Active_Power']

        # Plot the relationship between Active Power and other features
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
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

        print(f"Plot saved for {file} at: {plot_path}")

# Example usage:
# plot_feature_vs_active_power(data_dir='/path/to/data', save_dir='/path/to/save',
#                              features=['Feature1', 'Feature2'], colors=['blue', 'green'], 
#                              titles=['Title1', 'Title2'])
