import pandas as pd

# Open the .jsonl file and read it with pyarrow
file_input=r'notebooks\amazon_reviews\meta_Electronics.jsonl'
file_output=r'notebooks\amazon_reviews\meta_Electronics.parquet'


print('df load')
# Read the .jsonl file into a pandas DataFrame
df = pd.read_json(file_input, lines=True)

# Check if the 'price' column exists
if 'price' in df.columns:
    print('Cleaning price column')
    # Convert 'price' to numeric, coercing invalid parsing to NaN
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

print('df complete. converting to parquet')



# Convert the DataFrame to a Parquet file
df.to_parquet(file_output, engine='pyarrow')

print('df conversion complete')
