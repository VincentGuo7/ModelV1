import numpy as np
import pandas as pd

# # Load the .npy file
# data_y = np.load("y_data_final.npy")  # Replace with your actual file path
# data_X = np.load("X_data_final.npy")

# # Print the shape to understand its structure
# print(f"Outputs Data shape: {data_y.shape}")  # Expected: (num_samples, 21, num_features)
# print(f"Input Data shape: {data_X.shape}")  # Expected: (num_samples, 21, num_features)



# # Get number of samples
# num_samples = data.shape[0]
# print(f"Number of samples: {num_samples}")


# top5 = data[:5]  # Shape: (5, time_steps, features)

# # Save to text file in a readable format
# with open("top5_y_data.txt", "w") as f:
#     for i, sample in enumerate(top5):
#         f.write(f"Sample {i}:\n")
#         np.savetxt(f, sample, fmt="%.4f")
#         f.write("\n" + "-"*40 + "\n")



# column_means = data[:, 3:].mean(axis=0)

# # Print the result
# print("Column means from column 3 onwards:\n", column_means)




# ___________________________________________________________________________


df = pd.read_parquet('climax_processed_NA.parquet')
print(df.head())
print(df.shape)  # This will print (number of rows, number of columns)

# print("Column Titles:")
# print(df.columns.tolist())  # Prints all column names as a list

# def summarize_parquet_columns(parquet_file):
#     # Load the DataFrame
#     df = pd.read_parquet(parquet_file)

#     # Drop non-numeric columns (optional, depending on your dataset)
#     df_numeric = df.select_dtypes(include='number')

#     # Calculate mean and variance
#     summary = pd.DataFrame({
#         'Mean': df_numeric.mean(),
#         'Variance': df_numeric.var()
#     })

#     # Reset index to get column names as a column
#     summary.reset_index(inplace=True)
#     summary.rename(columns={'index': 'Feature'}, inplace=True)

#     print(summary.to_string(index=False))

#     return summary


# # Example usage
# summary_df = summarize_parquet_columns('climax_processed_NA.parquet')


# ________________________________________________________________________



# import numpy as np

# # Replace with the path to your .npz file
# file_path = '/Volumes/TOSHIBA EXT/5.625deg_npzmid/train/2015_4.npz'

# # Load the .npz file
# data = np.load(file_path)

# # Print the keys and the size of each value
# print("Keys and sizes in the .npz file:")
# for key in data.files:
#     value = data[key]
#     print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")


# _____________________________________________________________________

# data = np.load('climatology.npy')

# if data.dtype.names:  # Structured array
#     print("Field names (like columns):", data.dtype.names)
#     print("First row:\n", data[0])
# else:
#     print("Not a structured array. Shape:", data.shape)
#     print("First few entries:\n", data[:5])



# data = np.load('X_data_final.npy')
# print("Array shape:", data.shape)
# print("Array dtype:", data.dtype)
# print("First few entries:\n", data[:5])
