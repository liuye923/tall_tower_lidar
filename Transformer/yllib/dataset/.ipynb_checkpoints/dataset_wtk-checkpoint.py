from netCDF4 import Dataset
import pandas as pd
import xarray as xr
import os



def read_postprocess_daily_ts(datadir, vname, time=None, site=None, height=None, freq=None):
    sitename = [
    "L01", "L02", "L03", "L05", "L07", "L08", 
    "S01", "S02", "S03", "S04", "S05", "S06", "S07",
    "S10", "S12", "S18", "S14", "S17", "S20",
    ]
    vname = f'{vname}_{height}m'
    daytime = pd.date_range(time[0], time[1], freq="1D")
    mod = []
    for t in daytime:
        file = f"{datadir}/{t:%Y%m%d}.nc"
        if os.path.exists(file):
            data = xr.open_dataset(file)[vname]
            data.coords['sites'] = ('sites', sitename) 
            # print(data)
            mod.append(data)#.sel(Time=slice(t+pd.Timedelta("12H"),t+pd.Timedelta("35H")+pd.Timedelta("55min"))))
    mod = xr.concat(mod, dim="time")
    # print(mod)

    if site is not None: mod = mod.sel(sites=site)
    # del mod["XLONG"]; del mod["XLAT"]; del mod["lat"]; del mod["lon"]
    return mod



def read_wtk_ts_2km_pnnl(datadir, vname, time=None, site=None, height=None, freq=None):
    sitename = [
    "L01", "L02", "L03", "L05", "L07", "L08", 
    "S01", "S02", "S03", "S04", "S05", "S06", "S07",
    "S10", "S12", "S18", "S14", "S17", "S20",
    ]
    vname = f'{vname}_{height}m'
    mod = xr.open_dataset(f'{datadir}/wtk_ts.nc')
    mod.coords['sites'] = ('sites', sitename)
    mod = mod[vname]

    if site is not None: mod = mod.sel(sites=site)
    if time is not None: mod = mod.sel(time=slice(*time))
    return mod

def read_wtk_ts_4km_wtk(datadir, vname, time=None, site=None, height=None, freq=None):
    sitename = [
    "L01", "L02", "L03", "L05", "L07", "L08", 
    "S01", "S02", "S03", "S04", "S05", "S06", "S07",
    "S10", "S12", "S18", "S14", "S17", "S20",
    ]
    vname = f'{vname}_{height}m'
    mod = xr.open_dataset(f'{datadir}/wtk_ts.nc')
    mod.coords['sites'] = ('sites', sitename)
    mod = mod[vname]

    print(mod)
    if site is not None: mod = mod.sel(sites=site)
    if time is not None: mod = mod.sel(time=slice(*time))
    return mod