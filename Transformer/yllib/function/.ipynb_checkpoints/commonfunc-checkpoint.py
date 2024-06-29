import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import netCDF4 as nc
import xarray as xr
import pandas as pd
#import wrf
from shapely.geometry.polygon import Polygon
from cartopy import crs as ccrs
from cartopy.feature import NaturalEarthFeature, OCEAN, LAND, LAKES
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#import cmaps
import datetime
from copy import deepcopy
import glob
import os

class CommonFunc(object):
#========================================================================
#  wrf getvar
#========================================================================
    def get_plot_element(self, infile, vname='HGT'):
        rootgroup = nc.Dataset(infile, 'r')
        p = wrf.getvar(rootgroup, vname).squeeze()
        lats, lons = wrf.latlon_coords(p)
        cart_proj = wrf.get_cartopy(p)
        rootgroup.close()
        return cart_proj, p, lats, lons
#========================================================================
# plot - add polygon
#========================================================================
    def create_polygon(self, lats, lons):
        pgon = []
        for ilat, ilon in zip(lats[:,0],     lons[:,0]    ): pgon.append([ilon.data, ilat.data])
        for ilat, ilon in zip(lats[-1,:],    lons[-1,:]   ): pgon.append([ilon.data, ilat.data])
        for ilat, ilon in zip(lats[::-1,-1], lons[::-1,-1]): pgon.append([ilon.data, ilat.data])
        for ilat, ilon in zip(lats[0,::-1],  lons[0,::-1] ): pgon.append([ilon.data, ilat.data])
        return(Polygon(pgon))
#========================================================================
# DATA
#========================================================================
    def read_etopo(self, res=0.05, opt=None, extent=None):
        base = "/pic/projects/windpower_wfip2uq/liuy351/data/ETOPO/ETOPO1_Ice_g_gmt4"
        if opt is not None:
            fname = '{}.{}.{}.nc'.format(base, res, opt)
        else:
            fname = '{}.{}.nc'.format(base, res)
        topo = xr.open_dataset(fname)['z']
        if extent is not None:  topo = topo.sel(lon=slice(*extent[:2]), lat=slice(*extent[2:]))
        return topo
#========================================================================
# CALCULATION
#========================================================================
    def bootstrap(self, dat, sample_size=5, population=100):
        data = np.arange(50)
        dat = [resample(data, replace=True, n_samples=sample_size, random_state=i) for i in range(population)]
        book = np.vstack([a.reshape(sample_size,-1).T for a in book])
        return book
    
    #...weighted average
    def wgt_area_ave(self, da, dim=['south_north','west_east'], wdim="lat"):
        lat  = da.coords[wdim]
        clat = np.cos(np.deg2rad(da.coords[wdim]))
        wda  = da * clat
        wave = wda.mean(dim=dim) / clat.mean()
        return wave
    
    #...two sample difference with p-value
    def myTwoSampleDiff(self, xda1, xda2, dim='time', **kwargs):
        from scipy.stats import ttest_ind
        if isinstance(xda1, xr.Dataset): xda1 = xda1.to_array()
        if isinstance(xda2, xr.Dataset): xda2 = xda2.to_array()
    #    da1, da2 = xr.broadcast(xda1, xda2)
        da1, da2 = (xda1, xda2)
        dim_idx = da1.dims.index(dim)
        t, p = ttest_ind(da1.data, da2.data, axis=dim_idx)
        dims   = [k for k in da1.dims if k != dim]
        coords = {k: v for k, v in da1.coords.items() if k != dim}
        p    = xr.DataArray(p, dims=dims, coords=coords)
        diff = da1.mean(dim) - da2.mean(dim)
        diff.name='diff'
        p.name='pval'
        return diff, p
    
    #...two sample difference with p-value
    def myTwoSampleDiff_fast(self, xda1, xda2, dim='time', **kwargs):
        from scipy.stats import ttest_ind
        if isinstance(xda1, xr.Dataset): xda1 = xda1.to_array()
        if isinstance(xda2, xr.Dataset): xda2 = xda2.to_array()
        da1, da2 = (xda1, xda2)
        dim_idx = da1.dims.index(dim)
        t, p = ttest_ind(da1.data, da2.data, axis=dim_idx)
        dims   = [k for k in da1.dims if k != dim]
        coords = {k: v for k, v in da1.coords.items() if k != dim}
        p    = xr.DataArray(p, dims=dims, coords=coords)
        diff = da1.mean(dim) - da2.mean(dim)
        return diff, p

    #...linear regression 
    def sci_linregress(self, x, y, dim='time'):
        print(x.coords)
        print(y.coords)
        return xr.apply_ufunc(_sci_linregress, x, y,
            input_core_dims=[[dim], [dim]],
            vectorize=True,# !Important!
            output_core_dims=[[],[],[],[],[]],
            )

    def _sci_linregress(self, xx, yy):
        from scipy import stats
        x = xx[~np.isnan(xx) & ~np.isnan(yy)]
        y = yy[~np.isnan(xx) & ~np.isnan(yy)]
        if len(x)<4 or len(y)<4:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return stats.linregress(x, y)

    def sci_pearsonr(self, x, y, dim='time'):
        return xr.apply_ufunc(self._sci_pearsonr, x, y,
            input_core_dims=[[dim], [dim]],
            vectorize=True,# !Important!
            output_core_dims=[[],[]],
            )

    def _sci_pearsonr(self, xx, yy):
        from scipy import stats
        x = xx[~np.isnan(xx) & ~np.isnan(yy)]
        y = yy[~np.isnan(xx) & ~np.isnan(yy)]
        if len(x)<4 or len(y)<4:
            return np.nan, np.nan
        else:
            return stats.pearsonr(x, y)
        
    def sci_spearmanr(self, x, y, dim='time'):
        return xr.apply_ufunc(self._sci_spearmanr, x, y,
            input_core_dims=[[dim], [dim]],
            vectorize=True,# !Important!
            output_core_dims=[[],[]],
            )

    def _sci_spearmanr(self, xx, yy):
        from scipy import stats
        x = xx[~np.isnan(xx) & ~np.isnan(yy)]
        y = yy[~np.isnan(xx) & ~np.isnan(yy)]
        if len(x)<4 or len(y)<4:
            return np.nan, np.nan
        else:
            return stats.spearmanr(x, y)
    
    def pattern_corr(self, da_x, da_y, dim=["latitude", "longitude"], clat=None):
        if clat is not None:
#            clat = xr.ufuncs.cos(xr.ufuncs.deg2rad(da_x.coords[clat]))
            clat = np.cos(np.deg2rad(da_x.coords[clat]))
            da_x = da_x * clat
            da_y = da_y * clat
        return xr.corr(da_x, da_y, dim=dim)
 
    
#========================================================================
# write netcdf
#========================================================================
    def write_xarray_to_netcdf(self, xarray_array, output_path,mode='w', format='NETCDF4', group=None, engine=None,
                               encoding=None):
        """writes and xarray in a netcdf format outputfile
        Uses the xarray typical for wrf-python. The projection objects are transformed into strings
        to be able to use them as netcdf attributes
        :param xarray_array: xarray.DataArray
        :param output_path: str
        :param format: 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT' or 'NETCDF3_CLASSIC'
                        default: 'NETCDF4'
        :param group: str, default None
        :param engine: 'netcdf4', 'scipy' or 'h5netcdf'
        :param encoding: dict, default: None
        """
        xarray_array_out = xarray_array.copy(deep=True)
        # coordinates are extracted from variable
        del xarray_array_out.attrs['coordinates']
        # wrf-python projection object cannot be processed
        xarray_array_out.attrs['projection'] = str(xarray_array_out.attrs['projection'])
    
        xarray_array_out.to_netcdf(path=output_path, mode=mode, format=format, group=group,
                                   engine=engine,
                                   encoding=encoding)

#========================================================================
# Time record
#========================================================================
import time
class time_record(object):
    def __init__(self, s='Starting...'):
        self.s  = s
        self.t0 = time.time() 
        self._print(0)
    def proc(self, s='Processing...'):
        self.s  = s
        self._print(1)
    def end(self, s='Total'):
        self.s  = s
        self._print(2)

    def _print(self, opt):
        if opt==0: print(self.s); self.t1=self.t0
        if opt==1: 
           self.t2 = time.time()
           dt = self.t2 - self.t1
           print('{} - Time:{:.2f} sec'.format(self.s, dt))
           self.t1 = self.t2
        if opt==2: 
           self.t2 = time.time()
           dt = self.t2 - self.t0
           print('{} - Time:{:.2f} sec'.format(self.s, dt))


if __name__ == "__main__":
#    sites = CC_site_location()
#    print(sites)
#   TR = time_record()
#   TR.proc('abc')
#   TR.end()
 
   dat = READ_OBS(time=['201608120000','201608300000'])
   dat = xr.Dataset.from_dataframe(dat)
   dat.to_netcdf('obs.nc')
   dat = dat.to_array(dim='site')
   ave = dat.mean(dim='site')
   ave.to_netcdf('obs_ave.nc')
   print(dat)
   print(ave)
