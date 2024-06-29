import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from shapely.geometry.polygon import Polygon
import datetime
from copy import deepcopy
import glob
import os

#========================================================================
# LOAD DATA
#========================================================================
#--- MODEL level 1 ---
class CurrentFunc():
    def __init__(self):
        pass

    def calculate_geographic_rectangle(self, midpoint1, midpoint2, width_km):
        # Constants
        EARTH_RADIUS_KM = 6371.0  # Approximate radius of the earth in kilometers
        DEG_TO_KM = np.pi * EARTH_RADIUS_KM / 180  # Factor to convert degrees to kilometers
        
        # Convert midpoints to numpy arrays
        mp1 = np.array(midpoint1)
        mp2 = np.array(midpoint2)
        
        # Vector from the first to the second midpoint
        vector = mp2 - mp1
        
        # Calculate half-width in degrees
        half_width_deg = (width_km / 2) / DEG_TO_KM
        
        # Normalize the vector
        vector_norm = np.linalg.norm(vector)
        unit_vector = vector / vector_norm
        
        # Rotate 90 degrees to find perpendicular
        perp_vector = np.array([-unit_vector[1], unit_vector[0]]) * half_width_deg
        
        # Calculate vertices assuming square (midpoints are parallel to sides)
        vertex1 = mp1 + perp_vector
        vertex2 = mp1 - perp_vector
        vertex3 = mp2 - perp_vector
        vertex4 = mp2 + perp_vector
        
        # Calculate angle from East
        angle_from_east = np.degrees(np.arctan2(vector[1], vector[0]))
        
        return (vertex1[::-1], vertex2[::-1], vertex3[::-1], vertex4[::-1]), angle_from_east

    def calculate_grid(self, vertices, num_lines, num_points):
        lat_2_3 = np.linspace(vertices[2][1], vertices[3][1], num_points)
        lat_1_0 = np.linspace(vertices[1][1], vertices[0][1], num_points)
        lon_2_3 = np.linspace(vertices[2][0], vertices[3][0], num_points)
        lon_1_0 = np.linspace(vertices[1][0], vertices[0][0], num_points)

        grid_lat = []
        grid_lon = []
        for ilat23, ilon23, ilat10, ilon10 in zip(lat_2_3, lon_2_3, lat_1_0, lon_1_0):
            grid_lat.append(np.linspace(ilat10, ilat23, num_lines))
            grid_lon.append(np.linspace(ilon10, ilon23, num_lines))
        grid_lat = np.array(grid_lat).T
        grid_lon = np.array(grid_lon).T
        lat = xr.DataArray(grid_lat, 
            coords={'lat2d':(('south_north', 'west_east'), grid_lat), 
                    'lon2d':(('south_north', 'west_east'), grid_lon)}, 
            dims=('south_north', 'west_east'), 
            name='lat', attrs={'units':'degrees_north'})
        lon = xr.DataArray(grid_lon, 
            coords={'lat2d':(('south_north', 'west_east'), grid_lat), 
                    'lon2d':(('south_north', 'west_east'), grid_lon)}, 
            dims=('south_north', 'west_east'), 
            name='lon', attrs={'units':'degrees_north'})
        coords = xr.merge([lat, lon])
        # print(ds_out)
        return coords
        
mycross_section = CurrentFunc()

if __name__ == '__main__':
    import xesmf as xe
    ncfile = Dataset('/pscratch/sd/y/yeliu/WRF/control_2020/wrfout_d01_2020-06-01_00:00:00')
    midpoint1 = (29.374, -95.30)  
    midpoint2 = (30.853, -95.30)  
    width_km = 45  # Example width in kilometers
    
    vertices, angle = mycross_section.calculate_geographic_rectangle(midpoint1, midpoint2, width_km)
    print("Vertices of the rectangle are:", vertices)
    print('angle from east:', angle)
    da = xr.open_dataset('../../test_data.nc')
    grid =  mycross_section.calculate_grid(vertices, 20, 10)

    da.coords['lat'] = da.coords['XLAT']
    da.coords['lon'] = da.coords['XLONG']
    regridder = xe.Regridder(da, grid, "bilinear")
    print(regridder)
    da_regrid = regridder(da)
    print(da_regrid)
    da_regrid.to_netcdf('regrid.nc')
