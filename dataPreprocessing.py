import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

DEFAULT_PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925]

# Define lat-lon bounding box
LAT_MIN, LAT_MAX = 15, 65
LON_MIN, LON_MAX = 220, 300

def preprocess_to_daily_parquet():

    root_dir = "/Volumes/TOSHIBA EXT/5.625Deg"
    output_path = "/Users/vincentguo/Desktop/Weather-Prediction/climax_processed_NA.parquet"
    years = range(1979, 2019)
    variables = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity"
    ]

    # Load constants and subset to region
    constants_ds = xr.open_mfdataset(os.path.join(root_dir, "constants.nc"), combine="by_coords", parallel=True)
    constants_ds = constants_ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    constant_fields = ["land_sea_mask", "orography", "lattitude"]
    constant_vars = {}
    for f in constant_fields:
        var_name = NAME_TO_VAR[f]
        constant_vars[f] = constants_ds[var_name].to_numpy()  # shape (lat, lon)

    all_dfs = []

    for year in tqdm(years, desc="Processing years"):

        var_data_arrays = []

        # Get lat/lon/time from an example variable
        example_var = variables[0]
        example_files = glob.glob(os.path.join(root_dir, example_var, f"*{year}*.nc"))
        example_ds = xr.open_mfdataset(example_files, combine="by_coords", parallel=True)
        example_ds = example_ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        lat = example_ds["lat"].to_numpy()
        lon = example_ds["lon"].to_numpy()
        time = example_ds["time"]

        time = time.sel(time=(time.dt.year == year))  # filter by year exactly

        # Broadcast constants to (lat, lon), and convert to DataArrays
        for f in constant_fields:
            const_arr = constant_vars[f]
            const_da = xr.DataArray(const_arr, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name=f)
            var_data_arrays.append(const_da)

        # Process each variable
        for var in variables:
            var_code = NAME_TO_VAR[var]
            var_files = glob.glob(os.path.join(root_dir, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(var_files, combine="by_coords", parallel=True)
            ds = ds.sel(time=(ds.time.dt.year == year), lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))  # <-- MODIFIED

            if len(ds[var_code].shape) == 3:
                # Surface variable: (time, lat, lon)
                daily_mean = ds[var_code].resample(time="1D").mean(dim="time")
                daily_mean = daily_mean.rename(var)
                var_data_arrays.append(daily_mean)

            elif len(ds[var_code].shape) == 4:
                available_levels = ds["level"].to_numpy()
                levels_to_use = np.intersect1d(available_levels, DEFAULT_PRESSURE_LEVELS)
                for level in levels_to_use:
                    ds_level = ds.sel(level=level)
                    daily_mean = ds_level[var_code].resample(time="1D").mean(dim="time")
                    daily_mean = daily_mean.squeeze()
                    daily_mean.name = f"{var}_{int(level)}"

                    # Drop the 'level' coordinate if it's still there
                    if "level" in daily_mean.coords:
                        daily_mean = daily_mean.drop_vars("level")

                    var_data_arrays.append(daily_mean)
            else:
                raise RuntimeError(f"Unexpected shape for variable {var}: {ds[var_code].shape}")

        # Merge all variables
        combined_ds = xr.merge(var_data_arrays)

        # Add 'date' as separate coordinate
        combined_ds = combined_ds.assign_coords(date=combined_ds["time"].dt.date)

        # Convert to dataframe and reset index
        df = combined_ds.to_dataframe().reset_index()

        # Keep only relevant columns: lat, lon, date, [constants], [variables]
        df = df.drop(columns=["time"])

        # Rename 'lattitude' to 'latitude' if needed
        if "lattitude" in df.columns:
            df = df.rename(columns={"lattitude": "latitude"})

        df = df.sort_values(by=["lat", "lon", "date"]).reset_index(drop=True)
        all_dfs.append(df)

    # Combine all years
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Column ordering
    constants_lower = ["latitude" if f == "lattitude" else f for f in constant_fields]
    constants_lower = [c for c in constants_lower if c in full_df.columns]
    feature_columns = [c for c in full_df.columns if c not in ["lat", "lon", "date"] + constants_lower]
    full_df = full_df[["lat", "lon", "date"] + constants_lower + feature_columns]

    # Save
    full_df.to_parquet(output_path)
    print(f"Saved daily-averaged dataset to {output_path}")


if __name__ == "__main__":
    preprocess_to_daily_parquet()
