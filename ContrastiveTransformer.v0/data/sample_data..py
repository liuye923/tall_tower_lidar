import numpy as np
import xarray as xr
import os

def load_and_preprocess_data(data_dir, variables, time_range=None):
    # Load and preprocess data from NetCDF files
    data_list = []
    for var in variables:
        file_path = os.path.join(data_dir, f"{var}.nc")
        ds = xr.open_dataset(file_path)
        da = ds[var]
        if time_range:
            da = da.sel(time=slice(*time_range))
        data_list.append(da.values)
    data = np.stack(data_list, axis=1)
    return data

def save_preprocessed_data(data, save_path):
    np.save(save_path, data)

if __name__ == "__main__":
    data_dir = "data/raw"
    save_path = "data/processed/preprocessed_data.npy"
    variables = ["z500", "z850", "t500", "t850", "t2", "psfc"]
    time_range = ("2016-06-26", "2016-06-30")
    
    data = load_and_preprocess_data(data_dir, variables, time_range)
    save_preprocessed_data(data, save_path)
