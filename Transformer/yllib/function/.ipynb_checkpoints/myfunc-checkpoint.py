import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import netCDF4 as nc
import xarray as xr
import pandas as pd
#import wrf
from shapely.geometry.polygon import Polygon
#from cartopy import crs as ccrs
#from cartopy.feature import NaturalEarthFeature, OCEAN, LAND, LAKES
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#import cmaps
import datetime
from copy import deepcopy
import glob
import os
from .commonfunc import CommonFunc

#========================================================================
# LOAD DATA
#========================================================================
#--- MODEL level 1 ---
class CurrentFunc(CommonFunc):
    def __init__(self):
        pass
    
    #...weighted average
    def wgt_area_ave(self, da, dim=['south_north','west_east'], wdim="lat"):
        lat  = da.coords[wdim]
        clat = np.cos(np.deg2rad(da.coords[wdim]))
        wda  = da * clat
        wave = wda.mean(dim=dim) / clat.mean()
        return wave
    
    def calc_seasonal_mean(self, data, time_coord="forecast_time"):
        season = data.groupby(f"{time_coord}.season").mean()
        season.coords["season"] = season.coords["season"].astype("str")
        ann    = data.mean(time_coord)
        ann    = ann.assign_coords({"season":"ANN"}).expand_dims("season")
        mean = xr.concat([ann, season], dim="season")
        return mean

    def calc_seasonal_std(self, data, time_coord="forecast_time"):
        season = data.groupby(f"{time_coord}.season").std()
        season.coords["season"] = season.coords["season"].astype("str")
        ann    = data.std(time_coord)
        ann    = ann.assign_coords({"season":"ANN"}).expand_dims("season")
        std = xr.concat([ann, season], dim="season")
        return std

    def calc_bias_correction_monthly(self, obs_hindcast, fct_hindcast, fct_forecast):
        obs_hindcast_mean = obs_hindcast.groupby("forecast_time.month").mean()
        fct_hindcast_mean = fct_hindcast.mean("realization").groupby("forecast_time.month").mean()
        fct_forecast_mean = fct_forecast.mean("realization").groupby("forecast_time.month").std()
        obs_hindcast_std  = obs_hindcast.groupby("forecast_time.month").std()
        fct_hindcast_std  = fct_hindcast.mean("realization").groupby("forecast_time.month").std()
#        fct_hindcast_std  = fct_hindcast.groupby("forecast_time.month").std()
        fct_forecast_std  = fct_forecast.mean("realization").groupby("forecast_time.month").std()

        fct_forecast_correct = xr.apply_ufunc(lambda x, x_mean, o_std, x_std, o_mean: (x-x_mean) * o_std/x_std + o_mean,
           fct_forecast.groupby("forecast_time.month"),
           fct_hindcast_mean,
           obs_hindcast_std,
#           fct_forecast_std,
           fct_hindcast_std,
           obs_hindcast_mean,
        )
        del fct_forecast_correct.coords["month"]
        print(fct_forecast_correct)
        return fct_forecast_correct

    def read_CTENSO(self, year1, year2, rolling=None):
        nino3 = self.read_NINO3(year1, year2)
        nino4 = self.read_NINO4(year1, year2)

        alpha = np.where(nino3*nino4>0, 0.4, 0)
        CT = nino3 - nino4 * alpha
        WP = nino4 - nino3 * alpha
        return CT, WP
    def read_NINO3(self, year1, year2, rolling=None):
        df = pd.read_csv("../climate_index/NINO3.txt", index_col=0, skiprows=1, skipfooter=3,
             delim_whitespace=True, header=None)
        da = xr.DataArray(df.values.flatten(), coords={"time": pd.date_range("1948-01-01", "2022-12-01", freq="1MS")})
        da = da.where(da!=-99)
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def read_NINO4(self, year1, year2, rolling=None):
        df = pd.read_csv("../climate_index/NINO4.txt", index_col=0, skiprows=1, skipfooter=3,
             delim_whitespace=True, header=None)
        da = xr.DataArray(df.values.flatten(), coords={"time": pd.date_range("1948-01-01", "2022-12-01", freq="1MS")})
        da = da.where(da!=-99)
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def read_NAO(self, year1, year2, rolling=None):
        df = pd.read_csv("../climate_index/NAO.txt", index_col=0, skiprows=1, skipfooter=3,
             delim_whitespace=True, header=None)
        da = xr.DataArray(df.values.flatten(), coords={"time": pd.date_range("1948-01-01", "2022-12-01", freq="1MS")})
        da = da.where(da!=-99)
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def read_AO(self, year1, year2, rolling=None):
        df = pd.read_csv("../climate_index/AO.txt", index_col=0, skiprows=1, skipfooter=3,
             delim_whitespace=True, header=None)
        da = xr.DataArray(df.values.flatten(), coords={"time": pd.date_range("1950-01-01", "2022-12-01", freq="1MS")})
        da = da.where(da!=-99)
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def read_PDO(self, year1, year2, rolling=None):
        df = pd.read_csv("../climate_index/PDO.txt", index_col=0, skiprows=1, skipfooter=12,
             delim_whitespace=True, header=None)
        da = xr.DataArray(df.values.flatten(), coords={"time": pd.date_range("1948-01-01", "2021-12-01", freq="1MS")})
        da = da.where(da!=-99)
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def read_PNA(self, year1, year2, rolling=None):
        df = pd.read_csv("../climate_index/PNA.txt", index_col=0, skiprows=1, skipfooter=3,
             delim_whitespace=True, header=None)
        da = xr.DataArray(df.values.flatten(), coords={"time": pd.date_range("1948-01-01", "2022-12-01", freq="1MS")})
        da = da.where(da!=-99)
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def read_MJO(self, year1, year2, rolling=None):
        df = pd.read_csv("../climate_index/MJO_BOM.txt", index_col=0, skiprows=2, skipfooter=0,
             delim_whitespace=True, header=None)
        da = xr.DataArray(df.values[:,-2:], 
             coords={"time": pd.date_range("1974-06-01", "2022-03-09", freq="1D"), 
                     "var": ["phase", "amplitude"]}, 
             dims=["time", "var"])
        da = da.where(da!=-99)
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def read_t2m_basin(self, year1, year2, rolling=None):
        da = xr.open_dataset("data/t2m_basin_mean_monthly.nc")["t2m"]
        da = da.sel(time=slice(f"{year1}-01-01", f"{year2}-12-31"))
        if rolling is not None: da = da.rolling(time=rolling, center=True, min_periods=2).mean()
        return da

    def mask_keep_us(self, data):
        mask = xr.open_dataset("mask_us.nc")["mask"]
        mask = mask.sel(latitude=data.latitude, longitude=data.longitude)
        data.coords["mask"] = mask
        data = data.where(mask==1)
        del data.coords["mask"]
        return data

myfunc = CurrentFunc()
