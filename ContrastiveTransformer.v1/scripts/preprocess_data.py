import numpy as np
import xarray as xr
import os
import logging
from utils.data_loader import load_and_norm_netcdf_data, save_preprocessed_data

def main():
    data_dir = '../ERA5_reduced/'
    save_path = "./data/processed/preprocessed_data.npy"
    time_range = ("2016-06-01", "2016-06-30")
    
    # Load and normalize the data
    logging.info("Loading and normalizing data...")
    data = load_and_norm_netcdf_data(data_dir, time_range)
    
    # Save the preprocessed data
    logging.info(f"Saving preprocessed data to {save_path}...")
    save_preprocessed_data(data, save_path)
    logging.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
