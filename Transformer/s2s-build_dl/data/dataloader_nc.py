import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import netCDF4
import xarray as xr
import csv
import einops

mean_var = np.array([ \
[16.977392196655273,  15.04861831665039],
[7.633033275604248,  10.829400062561035],
[3.6484787464141846,  8.245396614074707],
[0.060167841613292694,  13.433121681213379],
[0.1114046573638916,  9.471024513244629],
[0.17080026865005493,  6.910264492034912],
[117325.4765625,  3694.645263671875],
[55407.9921875,  2176.040771484375],
[29832.50390625,  1323.74267578125],
[219.16659545898438,  4.6844377517700195],
[256.70770263671875,  9.16031265258789],
[272.0539245605469,  9.735929489135742]])

hrrr_mean_var = np.array([ \
[1.2301502227783203,  3.315296173095703],
[-2.704164743423462,  4.796664237976074]])

# mask is (670, 260)

#hrrr_mask = np.load("./hrrr_mask_reduced.npy")


"""
'(h w) (p1 p2 c) -> c (h p1) (w p2)
"""
class dataloader_superres(Dataset):

    def __init__(self,  ERA5_root_dir="/qfs/projects/windpower_wfip2uq/derm950/ERA5_S2S/",
                        HRRR_root_dir="/qfs/projects/windpower_wfip2uq/derm950/HRRR/",
                        correlation_file=None,
                        pretraining=True):
        """ arguments
        ERA5_root_dir (string): directory with ERA5 files
        correlation_file (string): path to correlation file
        """

        self.ERA5_root_dir = ERA5_root_dir
        
        self.correlation_file = correlation_file
        self.pretraining = pretraining

        if self.pretraining:
            with open(self.correlation_file, newline='') as f:
                reader = csv.reader(f)
                self.observation_pairs = [(pair[0],pair[1],pair[2]) for pair in reader]
        else:
            with open(self.correlation_file, newline='') as f:
                reader = csv.reader(f)
                self.observation_pairs = [(pair[0],pair[1],pair[2],pair[3]) for pair in reader]
                
    def __len__(self):
        return len(self.observation_pairs)


    def __getitem__(self, idx):

        if self.pretraining:

            ERA5_sample = self.read_era5(self.ERA5_root_dir, self.observation_pairs[idx][0])
            ERA5_sample = torch.as_tensor(ERA5_sample).float()

            return ERA5_sample, torch.tensor(0)

        else:

            ERA5_sample, HRRR_sample = self.read_era5_hrrr(self.ERA5_root_dir, self.HRRR_root_dir, self.observation_pairs[idx])

            ERA5_sample = torch.as_tensor(ERA5_sample).float()
            HRRR_sample = torch.as_tensor(HRRR_sample).float()

            """ The HRRR sample is served in patches. We will not need to un-patch the HRRR target and 
                model output to perform a loss.
            """

            HRRR_sample = einops.rearrange(HRRR_sample,'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 10, p2 = 10)
            HRRR_sample = HRRR_sample[self.patch_mask]

            return ERA5_sample, HRRR_sample, torch.tensor(idx)

    def read_era5(self,path,basestring):

        paramlist = ['u','v','z','t']
        heightlist = ['200','500','700']
        hour = np.random.choice(range(0,23))

        lat_min, lat_max = 10, 70
        lon_min, lon_max = -150,-40

        p_index = 0

        uvzt_stack = np.zeros(shape=(12,240,440))
        
        for param in paramlist:
            for height in heightlist:
                pfile = basestring.replace("z200","".join([param,height]))
                pfile = pfile.replace("z.200","".join([param,".",height]))

                ds = xr.open_dataset(path+pfile, engine="netcdf4")

                if self.lat_indices is None:
                    lat = ds.latitude
                    lon = ds.longitude

                    self.lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                    self.lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

                arr  = ds[param.upper()][hour][0][self.lat_indices[0]:self.lat_indices[-1],self.lon_indices[0]:self.lon_indices[-1]].values

                """
                    We normalize each channel to be mean zero with std 1. Early tests
                    suggested that this is very important.
                """
                arr  = arr - mean_var[p_index,0]
                arr /= mean_var[p_index,1]

                uvzt_stack[p_index] = arr

                p_index += 1
        
        return uvzt_stack

    def read_era5_hrrr(self, ERA5_path, HRRR_path, observation):

        era5_file = observation[0]
        era5_hour = int(observation[1])
        hrrr_file = observation[2]
        hrrr_index = int(observation[3])

        paramlist = ['u','v','z','t']
        heightlist = ['200','500','700']


        lat_min, lat_max = 10, 70
        lon_min, lon_max = -150,-40


        p_index = 0

        uvzt_stack = np.zeros(shape=(12,240,440))

        """ era5 """
        
        for param in paramlist:
            for height in heightlist:
                pfile = era5_file.replace("z200","".join([param,height]))
                pfile = pfile.replace("z.200","".join([param,".",height]))

                ds = xr.open_dataset(ERA5_path+pfile, engine="netcdf4")

                if self.lat_indices is None:
                    lat = ds.latitude
                    lon = ds.longitude

                    self.lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                    self.lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

                arr  = ds[param.upper()][era5_hour][0][self.lat_indices[0]:self.lat_indices[-1],self.lon_indices[0]:self.lon_indices[-1]].values
                arr  = arr - mean_var[p_index,0]
                arr /= mean_var[p_index,1]
                uvzt_stack[p_index] = arr
                p_index += 1


        """ hrrr """

        hrrr_uvzt_stack = np.zeros(shape=(2,670,260))
        p_index = 0
        for param in ["u","v"]:
            pfile = hrrr_file.replace("v",param)

            ds = xr.open_dataset(HRRR_path+pfile, engine="netcdf4")

            arr  = ds[param][hrrr_index].values
            arr  = arr - hrrr_mean_var[p_index,0]
            arr /= hrrr_mean_var[p_index,1]
            hrrr_uvzt_stack[p_index] = arr
            p_index += 1


        return uvzt_stack, hrrr_uvzt_stack


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dl = dataloader_superres("/pscratch/sd/j/jderm/ERA5_reduced/",
                             "/pscratch/sd/j/jderm/HRRR_reduced/",
                             "../era_hrrr_train.txt",
                             False)
    #print(f"len of dataset is {len(dl)}")
    
    dataloader = DataLoader(dl,batch_size=10, shuffle=True, num_workers=32)

    for batch, sample in enumerate(dataloader):
        print(batch)
        print(sample[0].shape)

