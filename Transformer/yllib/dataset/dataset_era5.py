from copy import deepcopy
from glob import glob
import os
import xarray as xr
import numpy as np
import pandas as pd

#========================================================================
# LOAD DATA
#========================================================================
class myERA5:
    def __init__(self):
        pass
    def __call__(self):
        pass

    def read_hgt(self, year1, year2, season=None, res=None, level=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/hgt_daily_globe/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/hgt_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["Z"]
        if level is not None: ds = ds.sel(level=level)
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_wind(self, year1, year2, season=None):
        fname = []
        for yr in np.arange(year1, year2+1):
#            fname += glob(f"/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/100uv_daily/100uv_{yr}??_daily.nc")
            fname += glob(f"/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/power_curve/100uv_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_psfc(self, year1, year2, season=None, res=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/psfc_daily/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/psfc_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["SP"]
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_sst(self, year1, year2, season, res=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/sst_daily/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/sstk_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["SSTK"]
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_basin_mean_daily(self):
#        fname = f"./data/era_basin_mean_hourly.nc"
        fname = f"./data/era_basin_mean_daily.nc"
        ds = xr.open_dataset(fname)
        return ds["uv"]

    def read_t2m(self, year1, year2, season=None, res=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/t2m_daily/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/t2m_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["t2m"]
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_sh(self, year1, year2, season=None, res=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/sh_daily/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/sh_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["ISHF"]
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_lh(self, year1, year2, season=None, res=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/lh_daily/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/lh_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["IE"]
        ds = ds * 2.25 * 1e6 * -1
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_sc(self, year1, year2, season=None, res=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/snow_daily/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/snow_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["SC"]
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

    def read_ci(self, year1, year2, season=None, res=None):
        fname = []
        datadir = "/qfs/projects/windpower_wfip2uq/liuy351/S2S/era5/ci_daily/"
        if res is not None: datadir = f"{datadir}/{res}"
        for yr in np.arange(year1, year2+1):
            fname += glob(f"{datadir}/ci_{yr}??_daily.nc")
        ds = xr.open_mfdataset(fname)["CI"]
        if season is not None: ds = ds.sel(time=ds["time.season"]==season)
        return ds

myera5 = myERA5()
