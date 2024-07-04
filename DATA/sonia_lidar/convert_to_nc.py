import xarray as xr
import numpy as np
import pandas as pd
import os


data_path = './'
level_index = [10,20,30,38,50,60,70,90,110,130,150]
level       = np.array(level_index) + 74.
sitename = 'WREF'

def read_single_day(date):
    file_name = date.strftime('Wind_1381@Y%Y_M%m_D%d.ZPH.csv')
    file_name = os.path.join(data_path, file_name)
    if os.path.exists(file_name):
        print(file_name)
        data = pd.read_csv(file_name, skiprows=1)
        data.drop(list(data.filter(regex='^Checksum*')), axis=1, inplace=True)
        data.index = data['Time and Date']
        data.index.name = 'time'
        data.index = pd.to_datetime(data.index, format='%m/%d/%Y %I:%M:%S %p')
    
        ds = data.to_xarray()
        uv   = ds[[f'Horizontal Wind Speed (m/s) at {lev}m' for lev in level_index]].to_array(dim='level', name='uv')
        uv.coords['level'] = level
        wdir = ds[[f'Wind Direction (deg) at {lev}m' for lev in level_index]].to_array(dim='level', name='wdir')
        wdir.coords['level'] = level
        w    = ds[[f'Vertical Wind Speed (m/s) at {lev}m' for lev in level_index]].to_array(dim='level', name='w')
        w.coords['level'] = level
        out = xr.merge([uv, wdir, w])
        print(out)
        return out

#date = pd.to_datetime('2023-08-23')
#read_single_day(date)

dates = pd.date_range('2023-11-01', '2023-11-30', freq='1D')
ds = [read_single_day(date) for date in dates]
ds = xr.concat(ds, dim='time')
print(ds)

ds.to_netcdf(f'lidar_2023_11.nc')


