import pickle
import numpy as np
import pandas as pd

file_path = './dataset/Drink/Drink/raw/ind.custom.x'

target = int(input())

# Attempting to load the content of the file using pickle again
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Checking the type and a small sample of the loaded data
data_type = type(data)
data_sample = data if isinstance(data, np.ndarray) else data[:5]  # Display a sample if it's not too large

print(data_sample)

# Converting the data_sample to a pandas DataFrame
if isinstance(data_sample, np.ndarray):
    df = pd.DataFrame(data_sample)
else:
    df = pd.DataFrame(data_sample)

print(len(df.index))
# Specifying the rows to write to the CSV file (e.g., rows 1 to 3)
rows_to_write = df.iloc[target:(target + 1)]  # Adjust the index range as needed

# Writing the specified rows to a CSV file
output_csv_path = './node_features.csv'
rows_to_write.to_csv(output_csv_path, index=False)

print(f"Specified rows have been written to {output_csv_path}")