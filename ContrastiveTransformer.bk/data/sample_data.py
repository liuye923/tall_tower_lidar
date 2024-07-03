import numpy as np
import xarray as xr
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_norm_single_variable(data_dir, var, vname, time=None, da_min=None, da_max=None):
    fname = f'{data_dir}/{var}.2001-2024.anomaly.nc'
    
    with xr.open_dataset(fname) as ds:
        da = ds[vname]
        if time is not None:
            da = da.sel(time=slice(*time))

        if (da_min is None) or (da_max is None):
            da_max = da.max(['time','latitude','longitude']).data
            da_min = da.max(['time','latitude','longitude']).data
            logging.debug(f'computing variable range for {var}')

        my_vmax = np.maximum(da_max, -1*da_min)
        vmax, vmin = my_vmax, -1*my_vmax
        
        logging.debug(f'{var}: {vmin}, {vmax}')
   
        out = (da - vmin) / (vmax - vmin)
        print(out.shape)
    return out

def load_and_norm_netcdf_data(data_dir, vrange, time=None):
    vars   = ['t500', 't850', 'z500', 'z850', '2t', 'sp']
    vnames = ['T',    'T',    'Z',    'Z',    'VAR_2T', 'SP']
        
    data_list = []
    for var, vname in zip(vars, vnames):
        data_list.append(
            load_and_norm_single_variable(data_dir, var, vname, time, *vrange.loc[var])
        )

    time = data_list[0].time
    for v in data_list:
        time = np.intersect1d(time, v.time)

    data_list = [v.sel(time=time) for v in data_list]
    
    normalized_data = np.stack(data_list, axis=1)

    logging.debug(f"Data shape after stacking and normalization: {normalized_data.shape}")

    return normalized_data, time

def save_preprocessed_data(data, save_path):
    np.save(save_path, data)

if __name__ == "__main__":
    data_dir = '../ERA5_reduced/'
    save_path = "data/processed/preprocessed_data.2001-2020.npy"
    save_idx_path = "data/processed/preprocessed_data.time.2001-2020.csv"
    range_path = f"{data_dir}/global_range.2001-2020.csv"
    time_range = ("2001-01-01 00:00:00", "2020-12-31 23:00:00")
    
    vrange = pd.read_csv(range_path, index_col=0)
    print(vrange)

    data, time = load_and_norm_netcdf_data(
        data_dir, 
        vrange,
        time_range, 
    )
    save_preprocessed_data(data, save_path)
    pd.Series(time).to_csv(save_idx_path)
