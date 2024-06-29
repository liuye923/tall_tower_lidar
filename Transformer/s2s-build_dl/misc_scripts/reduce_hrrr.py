import xarray
import numpy as np
import glob
import tqdm
import multiprocessing as mp
import sys


def main(files=list()):

    mask_defined = False
    maskX = None
    maskY = None

    
    for file in tqdm.tqdm(files[:]):
        ds = xarray.open_dataset(file, engine="netcdf4")
        rel_filename = file.split('HRRR_wind_80m')[1][1:]
        print(rel_filename)

        if not mask_defined:
        
            X = ds.x
            Y = ds.y
        
            m1a = Y > 330
            m1b = Y < 1000 + 1
            
            maskX = np.logical_and(m1a,m1b)
            
            m2a = X > 30
            m2b = X < 290 + 1
        
            maskY = np.logical_and(m2a,m2b)

        if '.u.nc' in rel_filename:
            ds_small = ds['u'][:,maskX,maskY]

        elif '.v.nc' in rel_filename:
            ds_small = ds['v'][:,maskX,maskY]

        rows = ds['u'].values.shape[1]

        lat = np.zeros_like(ds_small)
        lon = np.zeros_like(ds_small)
        j=0
        for i in range(0,rows):
            if maskX[i]:
                lat[0,j] = ds.latitude.values[i,maskY]
                lon[0,j] = ds.longitude.values[i,maskY]
                j+=1
        print(j)
        lat = lat[0]
        lon = lon[0]
        
        print(lat.shape)
        print(lon.shape)

        np.save("hrrr_lat.npy", lat)
        np.save("hrrr_lon.npy", lon)

        sys.exit(0)
    
        #ds_small.to_netcdf(path="/pscratch/sd/j/jderm/HRRR_reduced/"+rel_filename)

    return True

if __name__ == '__main__':

    path = "/pscratch/sd/y/yeliu/HRRR_wind_80m/"
    files = glob.glob(path + "*.nc")[:1]
    files.sort()

    pool = mp.Pool(processes=1)
    results = pool.map(main, [files[:]])

    main(files_subset)
