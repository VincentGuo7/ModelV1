import os
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_parquet('climax_processed_NA.parquet')

# Ensure 'date' is datetime for sorting
df['date'] = pd.to_datetime(df['date'])

print(f"Columns in dataset: {df.columns.tolist()}")

non_feature_cols = ['latitude']
feature_cols = [col for col in df.columns if col not in non_feature_cols]

x_path = "X_data.npy"
y_path = "y_data.npy"

# Clean up previous runs
for file in ["X_data_final.npy", "y_data_final.npy"]:
    if os.path.exists(file):
        os.remove(file)

x_shape = None
y_shape = None
sample_index = 0
max_samples = 30000000  # Estimated upper limit (adjust if needed)

with open("output_log_parquet.txt", "w") as f:
    for (lat, lon), group in df.groupby(['lat', 'lon']):
        # Sort by datetime first
        group = group.sort_values(by='date').reset_index(drop=True)

        # Convert 'date' to month *after* sorting, in-place
        group['date'] = group['date'].dt.month

        i = 0
        while i + 3 < len(group):
            input_block = group.iloc[i:i+3][feature_cols].copy()
            output_block = group.iloc[i+3][feature_cols].copy()

            input_block['lat'] = lat
            input_block['lon'] = lon
            output_block['lat'] = lat
            output_block['lon'] = lon

            if input_block.isnull().values.any() or output_block.isnull().values.any():
                i += 1
                continue

            f.write(f"\noutput_block values:\n{output_block}\n")

            input_arr = input_block.values.astype(np.float32)
            output_arr = output_block.values.astype(np.float32)

            if x_shape is None:
                x_shape = input_arr.shape
                y_shape = output_arr.shape

                # Create memory-mapped output files
                X_memmap = np.lib.format.open_memmap(
                    x_path, mode='w+', dtype=np.float32, shape=(max_samples, *x_shape)
                )
                Y_memmap = np.lib.format.open_memmap(
                    y_path, mode='w+', dtype=np.float32, shape=(max_samples, *y_shape)
                )

            if sample_index >= max_samples:
                raise RuntimeError(f"Exceeded preallocated sample limit ({max_samples}). Increase it.")

            X_memmap[sample_index] = input_arr
            Y_memmap[sample_index] = output_arr
            sample_index += 1
            i += 3

    # Flush changes to disk
    X_memmap.flush()
    Y_memmap.flush()

    # Truncate final saved arrays
    print(f"\n✅ Processed {sample_index} samples. Truncating output arrays...\n")
    X_final_path = x_path.replace(".npy", "_final.npy")
    Y_final_path = y_path.replace(".npy", "_final.npy")

    np.save(X_final_path, X_memmap[:sample_index])
    np.save(Y_final_path, Y_memmap[:sample_index])

    os.remove(x_path)
    os.remove(y_path)

print("\n✅ Dataset has been processed and saved to disk.\n")


# # Identify the 45 climate variable columns (exclude meta columns)
# meta_cols = ['lat', 'lon', 'date', 'land_sea_mask', 'orography', 'latitude']
# climate_vars = [col for col in df.columns if col not in meta_cols]

# print(f"\nCalculating climatology for {len(climate_vars)} climate variables...")

# # Group by lat/lon and compute mean
# climatology_df = df.groupby(['lat', 'lon'])[climate_vars].mean().reset_index()

# # Save as .npy
# climatology_arr = climatology_df[['lat', 'lon'] + climate_vars].values.astype(np.float32)
# np.save("climatology.npy", climatology_arr)

# print("✅ Climatology saved to climatology.npy\n")