# In preprocess_data.py and utils/data_loader.py
import xarray as xr
import numpy as np
import logging

def load_and_norm_single_variable_netcdf_data(data_dir, var, vname, time=None):
    fname = f'{data_dir}/{var}.2001-2020.anomaly.nc'
    with xr.open_dataset(fname) as ds:
        da = ds[vname]
        if time is not None:
            da = da.sel(time=slice(*time))
        da_max = da.max(['time', 'latitude', 'longitude']).data
        da_min = da.min(['time', 'latitude', 'longitude']).data
        my_vmax = np.maximum(da_max, -1 * da_min)
        vmax, vmin = my_vmax, -1 * my_vmax
        logging.debug(f'{var}: {vmin}, {vmax}')
        out = (da - vmin) / (vmax - vmin + 1e-8)  # Adding epsilon for numerical stability
    return out

def load_and_norm_netcdf_data(data_dir, time=None):
    t500_full = load_and_norm_single_variable_netcdf_data(data_dir, 't500', 'T', time=time)
    t850_full = load_and_norm_single_variable_netcdf_data(data_dir, 't850', 'T', time=time)
    z500_full = load_and_norm_single_variable_netcdf_data(data_dir, 'z500', 'Z', time=time)
    z850_full = load_and_norm_single_variable_netcdf_data(data_dir, 'z850', 'Z', time=time)
    t2_full = load_and_norm_single_variable_netcdf_data(data_dir, '2t', 'VAR_2T', time=time)
    sp_full = load_and_norm_single_variable_netcdf_data(data_dir, 'sp', 'SP', time=time)

    data_list = [t500_full, t850_full, z500_full, z850_full, t2_full, sp_full]
    time = data_list[0].time
    for v in data_list:
        time = np.intersect1d(time, v.time)

    for v in data_list:
        v = v.sel(time=time)

    # Stack along the variable axis
    normalized_data = np.stack(data_list, axis=1)  # Shape: (time, num_variables, height, width)
    logging.info(f"Data shape after stacking and normalization: {normalized_data.shape}")
    return normalized_data

def save_preprocessed_data(data, save_path):
    np.save(save_path, data)
