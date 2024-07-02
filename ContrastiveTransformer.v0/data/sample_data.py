import numpy as np
import xarray as xr
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_norm_single_variable_netcdf_data(data_dir, var, vname, time=None):
    fname = f'{data_dir}/{var}.2001-2020.anomaly.nc'
    
    with xr.open_dataset(fname) as ds:
        da = ds[vname]
        if time is not None:
            da = da.sel(time=slice(*time))
        da_max = da.max(['time','latitude','longitude']).data
        da_min = da.max(['time','latitude','longitude']).data
        
        my_vmax = np.maximum(da_max, -1*da_min)
        vmax, vmin = my_vmax, -1*my_vmax
        logging.debug(f'{var}: {vmin}, {vmax}')
   
        out = (da - vmin) / (vmax - vmin)
        
    return out

def load_and_norm_netcdf_data(data_dir, time=None):
    t500_full = load_and_norm_single_variable_netcdf_data(data_dir, 't500', 'T', time=time)
    t850_full = load_and_norm_single_variable_netcdf_data(data_dir, 't850', 'T', time=time)
    z500_full = load_and_norm_single_variable_netcdf_data(data_dir, 'z500', 'Z', time=time)
    z850_full = load_and_norm_single_variable_netcdf_data(data_dir, 'z850', 'Z', time=time)
    t2_full   = load_and_norm_single_variable_netcdf_data(data_dir, '2t', 'VAR_2T', time=time)
    sp_full   = load_and_norm_single_variable_netcdf_data(data_dir, 'sp', 'SP', time=time)

    data_list = [t500_full, t850_full, z500_full, z850_full, t2_full, sp_full]

    time = data_list[0].time
    for v in data_list:
        time = np.intersect1d(time, v.time)

    for v in data_list:
        v = v.sel(time=time)
    
    normalized_data = np.stack(data_list, axis=1)

    logging.debug(f"Data shape after stacking and normalization: {normalized_data.shape}")

    return normalized_data

def save_preprocessed_data(data, save_path):
    np.save(save_path, data)

if __name__ == "__main__":
    data_dir = '../ERA5_reduced/'
    save_path = "data/processed/preprocessed_data.2010-2020.npy"
    time_range = ("2010-01-01 00:00:00", "2020-12-31 23:00:00")
    
    data = load_and_norm_netcdf_data(data_dir, time_range)
    save_preprocessed_data(data, save_path)
