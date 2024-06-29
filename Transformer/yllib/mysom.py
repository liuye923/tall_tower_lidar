import sompy
import xarray as xr
import numpy as np
import pandas as pd
### SOM ###
class MySOM(object):
    def __init__(self):
        '''initilization & load global variables'''
        pass
    def __call__(self):
        pass

    def calc_anomaly(self, data):
        return data - data.mean("time")

    def calc_anomaly_monthly(self, data):
        return data.groupby("time.month") - data.groupby("time.month").mean("time")

    def convert_to_1D(self, *args):
        dat = []
        for v in args:
#            v = v - v.mean(dim=["longitude", "latitude"])
#            v = (v-v.mean(dim='time'))/v.std(dim='time')
#            v = v/v.std(dim='time')
            dat.append(self._convert_to_1D(v))
        dat = xr.concat(dat, 'z').reset_index('z')
        dat.name = 'preSOM'
        dim = dat.shape
        dat.coords['z'] = np.arange(dim[1])
        return dat

    def _convert_to_1D(self, dat):
        lat = dat.latitude
        weight = np.sqrt(np.cos(1./180*3.14159*lat))
        datw = dat * weight
        dim  = datw.shape
        datw = datw.stack(z=['longitude','latitude'])
        datw = datw.dropna('z')
        return datw

    def read_bmu(self, fname_bmu):
        #...load bmu
        bmu = xr.open_dataset(fname_bmu)["bmu"]
        return bmu

    def switch_bmu_season_4x4(self, bmu, season):
        print("swaping bmu....")
        switch_dict = {
            "cold"  : [0,1,14,15,4,5,10,11,7,6,9,8,3,2,13,12],
        }
        switch = switch_dict[season] 
        print(switch)
        for i in range(16):
            bmu = xr.where(bmu==switch[i], i+1000, bmu)
        bmu = bmu - 1000
        return bmu

    def switch_bmu_season_2x2(self, bmu, season):
        print("swaping bmu....")
        switch_dict = {
            "cold"  : [0,3,1,2],
            "warm"  : [2,1,0,3],
        }
        switch = switch_dict[season] 
        print(switch)
        for i in range(4):
            bmu = xr.where(bmu==switch[i], i+1000, bmu)
        bmu = bmu - 1000
        return bmu

    def switch_bmu_month(self, bmu, month):
        print("swaping bmu....")
        switch_dict = {
            "1"  : [3,0,2,1],
            "2"  : [1,2,0,3],
            "3"  : [0,3,2,1],
            "4"  : [1,2,0,3],
            "5"  : [2,1,0,3],
            "6"  : [2,1,0,3],
            "7"  : [3,0,2,1],
            "8"  : [0,3,2,1],
            "9"  : [2,1,3,0],
            "10" : [1,2,3,0],
            "11" : [2,1,3,0],
            "12" : [2,1,0,3],
        }
        switch = switch_dict[month] 
        print(switch)
        for i in range(4):
            bmu = xr.where(bmu==switch[i], i+1000, bmu)
        bmu = bmu - 1000
        return bmu

    def calc_cluster_mapsize_fast(self, data, bmu, mapsize=None):
#        data = data.assign_coords(bmu=("time", bmu.data))#["bmu"] = bmu.astype(np.float)
        data.coords["time"] = bmu.data
        cluster_mean = []
        pct = []
        print(data)
        for ibmu in np.arange(mapsize[0] * mapsize[1]):
            cluster = data.sel(time=ibmu)
            if cluster.size==0:
                cluster_mean.append(None)
                pct.append(0.)
            else:
                cluster_mean.append(cluster.mean("time"))
                pct.append(len(cluster.coords["time"]) / len(data.coords["time"]) * 100.)
        return cluster_mean, pct

    def calc_cluster_mapsize(self, data, bmu, mapsize=None):
        data = data.assign_coords(bmu=("time", bmu.data))#["bmu"] = bmu.astype(np.float)
        clusters = []
        cluster_mean = []
        pct = []
        for ibmu in np.arange(mapsize[0] * mapsize[1]):
            cluster = data.where(data["bmu"]==ibmu, drop=True)
            if cluster.size==0:
                clusters.append(None)
                cluster_mean.append(None)
                pct.append(0.)
            else:
                clusters.append(cluster)
                cluster_mean.append(cluster.mean("time"))
                pct.append(len(cluster.coords["bmu"]) / len(data.coords["bmu"]) * 100.)
        self.ncluster = len(cluster_mean)
        return clusters, cluster_mean, pct

    def calc_bmu_to_cluster(self, bmu):
        cluster = bmu.attrs["cluster"]
        nbmu    = bmu.max().astype(int).item() + 1
        for i in range(nbmu):
            bmu = xr.where(bmu==i, cluster[i]+1000, bmu)
        bmu = bmu - 1000
        return bmu

    def bmu_filename(self, mapsize, factors, month, opt=None):
        fname = f"som/bmu_map{mapsize[0]}x{mapsize[1]}_fac{'+'.join(factors)}"
        if isinstance(month, str):
            print(month)
            fname += f"{month}"
        else:
            if isinstance(month, (tuple, list)):
                fname += f"_{month[0]}-{month[1]}"
            else:
                fname += f"{month}"
        return fname+".nc"

    def calc_distance(self, da_x, da_y, dim=["latitude", "longitude"], clat=None):
        if clat is not None:
#            clat = xr.ufuncs.cos(xr.ufuncs.deg2rad(da_x.coords[clat]))
            clat = np.cos(np.deg2rad(da_x.coords[clat]))
            da_x = da_x * clat
            da_y = da_y * clat
            da_x = da_x.stack(z=dim)
            da_y = da_y.stack(z=dim)
            da_x, da_y = xr.broadcast(da_x, da_y)
            dis = np.sqrt(((da_x-da_y)**2).sum("z"))
        return dis

mysom   = MySOM()
