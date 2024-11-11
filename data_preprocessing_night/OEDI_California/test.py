# Process all processed CSV files and create an aggregated DataFrame with average values
def create_aggregated_average_df(csv_dir, timestamp_col='timestamp'):
    # List to store individual DataFrames
    df_list = []

    # Iterate through all CSV files in the directory
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Ensure timestamp column is in datetime format
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            # Append DataFrame to the list
            df_list.append(df)

    # Concatenate all DataFrames and calculate the mean for each timestamp
    combined_df = pd.concat(df_list)
    aggregated_df = combined_df.groupby(timestamp_col).mean()

    # Reset index to have timestamp as a column
    aggregated_df.reset_index(inplace=True)

    return aggregated_df



aggregated_df = create_aggregated_average_df(save_dir)
    aggregated_output_path = os.path.join(project_root, 'data/OEDI/2107(Arbuckle_California)/processed_data/aggregated_average_data.csv')
    os.makedirs(os.path.dirname(aggregated_output_path), exist_ok=True)
    aggregated_df.to_csv(aggregated_output_path, index=False)