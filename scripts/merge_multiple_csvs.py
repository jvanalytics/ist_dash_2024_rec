import os
import pandas as pd

# Folder path where your CSV files are located
folder_path = r'C:\Users\nosta\CODING\datasets\ga4_merch'

# List to store dataframes
csv_list = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Ensure we are working with CSV files
        file_path = os.path.join(folder_path, filename)
        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        print(f'filename read: {filename}')
        
        # Append the DataFrame to the list
        csv_list.append(df)
        
        print(f'filename write: {filename}')


# Concatenate all DataFrames into one
merged_df = pd.concat(csv_list, ignore_index=True)

# Save the combined DataFrame to a single CSV file
merged_df.to_csv(r'C:\Users\nosta\CODING\datasets\ga4_merch\google_merch_store_raw_merge.csv', index=False)

print("CSV files merged successfully!")