from netCDF4 import Dataset
import wrf
import pandas as pd
import xarray as xr
import os

def read_domain(domain_file):
    cart_proj, hgt, lats, lons = get_plot_element(domain_file, "HGT_M")
    return dict(projection=cart_proj, hgt=hgt, lat2d=lats, lon2d=lons)

def get_plot_element(infile, vname='HGT'):
    rootgroup = Dataset(infile, 'r')
    p = wrf.getvar(rootgroup, vname).squeeze()
    lats, lons = wrf.latlon_coords(p)
    cart_proj = wrf.get_cartopy(p)
    rootgroup.close()
    return cart_proj, p, lats, lons

def read_postprocess_daily_ts(datadir, vname, time=None, site=None, height=None, freq=None):
    xvname="uv" if vname in ("u", "v") else vname
    daytime = pd.date_range(time[0], time[1], freq="1D")
    mod = []
    for t in daytime:
        file = f"{datadir}/{xvname}_{t:%Y%m%d}.nc"
        if os.path.exists(file):
            data = xr.open_dataset(file)
            mod.append(data.sel(Time=slice(t+pd.Timedelta("12H"),t+pd.Timedelta("35H")+pd.Timedelta("50min"))))
    mod = xr.concat(mod, dim="Time")[vname]
    mod = mod.rename({"Time":"time"})
    if site is not None: mod = mod.sel(sites=site)
    if height is not None: mod = mod.sel(level=height)
    del mod["XLONG"]; del mod["XLAT"]; del mod["lat"]; del mod["lon"]
    return mod
