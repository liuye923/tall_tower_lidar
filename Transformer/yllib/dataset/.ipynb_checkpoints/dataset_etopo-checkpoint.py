import xarray as xr
import numpy as xp
import pandas as pd


def read(res=0.05, opt=None):
    base='yllib/ETOPO/ETOPO1_Ice_g_gmt4'
    if opt is not None:
        fname = f'{base}.{res}.{opt}.nc4'
    else:
        fname = f'{base}.{res}.nc4'

    topo = xr.open_dataset(fname)['z']
    return topo

def crop(raw, lats, lons, buffer=10):
    if not isinstance(buffer, (tuple, list)): 
        buffer = [buffer, buffer, buffer, buffer]
    else:
        if len(buffer)<4: 
            raise NameError('buffer should be a scaler or list with [lon1, lon2, lat1, lat2]')
    cropped = raw.sel(lat=slice(lats[0,0]-buffer[2], lats[-1,-1]+buffer[3]),
                      lon=slice(lons.min()-buffer[0], lons.max()+buffer[1]))
    etopo_cropped = cropped
    return cropped
