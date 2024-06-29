import xarray as xr
import numpy as xp
import pandas as pd


def _read_wfip2_radar_lidar(datadir, site, time=None, height=None, freq=None):
    if site[0]=="L":
        prefix="lidar"
        vname='wind_speed'
    if site[0]=="S":
        prefix="sodar"
        vname='speed'
    if site in ["S03", "S04", "S07", "S10"]:
        prefix="sodar"
        vname='wind_speed'
    ds = xr.open_dataset(f"{datadir}/{prefix}_z{site[1:]}.nc")[vname].compute()
    if ds.size==0: return None
    if height is not None: ds = ds.sel(height=height)
    if freq is not None: ds = ds.resample(time=freq).nearest()
    if time is not None:
        time1, time2 = time
        ds = ds.sel(time=slice(time1, time2))
    return ds

def read_wfip2_radar_lidar(datadir, sitename, time=None, height=None, freq=None):
    if not isinstance(sitename, (tuple, list)): sitename = (sitename,)
    sitename_valid = []
    ds = []
    for site in sitename:
        _ds = _read_wfip2_radar_lidar(datadir, site, time=time, height=height, freq=freq)
        if _ds is not None: 
            sitename_valid.append(site)
            ds.append(_ds)
    ds = xr.concat(ds, dim='site')
    ds.coords['site'] = sitename_valid
    return ds

def get_wfip2_site_location(sitename):
    df_site = pd.read_t
