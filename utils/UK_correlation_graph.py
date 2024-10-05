import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

UK_data = '/home/bak/Dataset/UK_data/PV Data/PV Data - csv files only/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv'
pv_df = pd.read_csv(UK_data)

weather_data = '/home/bak/Dataset/UK_data/Weather_Data_2014-11-30.csv'
weather_df = pd.read_csv(weather_data)

value_to_check = 'Forest Road'
pv_data = pv_df[pv_df['Substation'] == value_to_check]

weather_data = weather_df[weather_df['Site'] == value_to_check]
weather_data = weather_data[weather_data['Time'].str.endswith(':00:00')]

# Define the date range
start_date = '2013-12-10'
end_date = '2014-10-02'

# Filter the DataFrame for the specified date range
filtered_weather_data = weather_data[(weather_data['Date'] >= start_date) & (weather_data['Date'] <= end_date)].copy()
filtered_pv_data = pv_data[(pv_data['t_date'] >= start_date) & (pv_data['t_date'] <= end_date)].copy()
filtered_pv_data.loc[:, 'P_GEN_Filtered'] = ((filtered_pv_data['V_MIN_Filtered'] + filtered_pv_data['V_MAX_Filtered']) / 2) * \
                                    ((filtered_pv_data['I_GEN_MAX_Filtered'] + filtered_pv_data['I_GEN_MIN_Filtered']) / 2)
filtered_pv_data.loc[:, 'P_GEN_MEAN'] = filtered_pv_data['P_GEN_MIN'] + filtered_pv_data['P_GEN_MAX'] / 2

merged_data = pd.merge(filtered_pv_data, filtered_weather_data, left_on=['t_date', 't_h'], right_on=['Date', 'Hour'], suffixes=('_pv', '_weather'))

# Convert 'P_GEN_Filtered' and 'SolarRad' to numeric, replacing non-numeric values with NaN
merged_data['P_GEN_Filtered'] = pd.to_numeric(merged_data['P_GEN_Filtered'], errors='coerce')
merged_data['SolarRad'] = pd.to_numeric(merged_data['SolarRad'], errors='coerce')
merged_data['P_GEN_MEAN'] = pd.to_numeric(merged_data['P_GEN_MEAN'], errors='coerce')

# Create a new DataFrame with daily maximum values
daily_max_data = merged_data.groupby('Date').agg({
    'P_GEN_Filtered': 'max',
    'SolarRad': 'max',
    'P_GEN_MEAN': 'max'
}).reset_index()

# Rename columns for clarity
daily_max_data.columns = ['Date', 'Max_Power_calculated', 'Max_Solar_Radiation', 'Max_Power_kW']
daily_max_data['Max_Power_calculated_kW'] = daily_max_data['Max_Power_calculated'] / 1000  # Convert W to kW

# Remove rows with NaN values
daily_max_data = daily_max_data.dropna()

# Display the first few rows of the new DataFrame
print(daily_max_data.head())

# Plot 1: Scatter plot with both power measurements vs solar radiation
plt.figure(figsize=(12, 8))
plt.scatter(daily_max_data['Max_Solar_Radiation'], daily_max_data['Max_Power_calculated_kW'],
            color='blue', alpha=0.6, label='Calculated Power')
plt.scatter(daily_max_data['Max_Solar_Radiation'], daily_max_data['Max_Power_kW'],
            color='red', alpha=0.6, label='Measured Power')

plt.xlabel('Max Solar Radiation (W/m²)')
plt.ylabel('Max Power Generated (kW)')
plt.title('Comparison of Calculated vs Measured Max Power Generated\nwith respect to Max Solar Radiation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Scatter plot of calculated power vs measured power
plt.figure(figsize=(12, 8))
plt.scatter(daily_max_data['Max_Power_kW'], daily_max_data['Max_Power_calculated_kW'],
            color='purple', alpha=0.6)

# Add 1:1 line
max_value = max(daily_max_data['Max_Power_calculated_kW'].max(), daily_max_data['Max_Power_kW'].max())
plt.plot([0, max_value], [0, max_value], color='green', linestyle='--', alpha=0.5, label='1:1 Line')

plt.xlabel('Measured Max Power (kW)')
plt.ylabel('Calculated Max Power (kW)')
plt.title('Calculated vs Measured Max Power')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Scatter plot of calculated power vs measured power
plt.figure(figsize=(12, 8))
plt.scatter(daily_max_data['Max_Power_kW'], daily_max_data['Max_Power_calculated_kW'],
            color='purple', alpha=0.6)

# Add 1:1 line
max_value = max(daily_max_data['Max_Power_calculated_kW'].max(), daily_max_data['Max_Power_kW'].max())
plt.plot([0, max_value], [0, max_value], color='green', linestyle='--', alpha=0.5, label='1:1 Line')

plt.xlabel('Measured Max Power (kW)')
plt.ylabel('Calculated Max Power (kW)')
plt.title('Calculated vs Measured Max Power')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate and print the correlation coefficients
correlation_calculated = daily_max_data['Max_Solar_Radiation'].corr(daily_max_data['Max_Power_calculated_kW'])
correlation_measured = daily_max_data['Max_Solar_Radiation'].corr(daily_max_data['Max_Power_kW'])

print(f"Correlation coefficient (Calculated Power vs Solar Radiation): {correlation_calculated:.2f}")
print(f"Correlation coefficient (Measured Power vs Solar Radiation): {correlation_measured:.2f}")

# Calculate and print the RMSE between calculated and measured power
rmse = ((daily_max_data['Max_Power_calculated_kW'] - daily_max_data['Max_Power_kW']) ** 2).mean() ** 0.5
print(f"RMSE between calculated and measured power: {rmse:.2f} kW")

# Calculate correlation between calculated and measured power
correlation_power = daily_max_data['Max_Power_calculated_kW'].corr(daily_max_data['Max_Power_kW'])
print(f"Correlation coefficient (Calculated Power vs Measured Power): {correlation_power:.2f}")