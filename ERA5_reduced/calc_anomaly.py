import xarray as xr
import pandas as pd
import numpy as np

era5_reduced_root = './'
var_list = ['t500', 'z500', 't850', 'z850','sp','2t']
v_list   = ['T','Z','T','Z','SP','VAR_2T']
#var_list = ['100u', '100v']
#v_list   = ['VAR_100U', 'VAR_100V']
climate_period = ('2001-01-01 00:00','2020-12-31 23:00')
climate_years = pd.to_datetime(climate_period).year
print(climate_years)

def calc_anomaly(ds):
    return (
        ds.mean('time'),
        ds.mean(['time', 'latitude', 'longitude']),
        ds.max('time'),
        ds.max(['time', 'latitude', 'longitude']),
        ds.min('time'),
        ds.min(['time', 'latitude', 'longitude'])
    )

def remove_monthly_cycle(ds):
    return ds - ds.rolling(time=30, min_periods=2, center=True).mean() 

def remove_regional_mean(ds):
    return ds - ds.mean(['latitude', 'longitude'])

def remove_all_mean(ds):
    return ds - ds.mean(['time', 'latitude', 'longitude'])

def remove_regional_running_mean(ds, period=30*24):
    fldmean = ds.mean(['latitude', 'longitude'])
    return ds - fldmean.rolling(time=period, min_periods=2, center=True).mean() 

min_full, max_full = [], []
vrange = []
for var, vname in zip(var_list, v_list):
    print(f'processing: {var}')
    ds = xr.open_dataset(f'{var}.2001-2024.nc')[vname].squeeze()
    if 'level' in ds.coords: del ds.coords['level']
    ds_clim = ds.sel(time=slice(*climate_period))

#    ds_ano = remove_monthly_cycle(ds)
#    ds_clim_ano = remove_monthly_cycle(ds_clim)
#    ds_ano.to_netcdf(f'{var}.2001-2024.anomaly.nc')

#    ds_ano = remove_regional_mean(ds)
#    ds_clim_ano = remove_regional_mean(ds_clim)
#    ds_ano.to_netcdf(f'{var}.2001-2024.anomaly_fldmean.nc')

    # ds_ano = remove_all_mean(ds)
    # ds_clim_ano = remove_all_mean(ds_clim)
    # ds_ano.to_netcdf(f'{var}.2001-2024.anomaly_allmean.nc')

    ds_ano = remove_regional_running_mean(ds)
    ds_clim_ano = remove_regional_running_mean(ds_clim)
    ds_ano.to_netcdf(f'{var}.2001-2024.anomaly_fld30d.nc')

    vrange.append([
        ds_clim_ano.min(['time', 'latitude', 'longitude']).data,
        ds_clim_ano.max(['time', 'latitude', 'longitude']).data
    ])
    print(vrange)

vrange = pd.DataFrame(vrange, index=var_list, columns=['vmin','vmax'])
print(vrange)

vrange.to_csv('global_range.2001-2024.fld30d.csv')
    


