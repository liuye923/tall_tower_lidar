import torch

import numpy as np
import scipy.io as sio
import pandas as pd
import datetime as dt
import xarray as xr
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


root_dir = '/global/cfs/projectdirs/m1657/liuy351/TallTower/ContrastiveTransformer'

data_option = 'allmean'
model_option = '4patch'
cluster_option = 'ANN.r30d24h' # r30d - remove 30 day running mean, 24h - 24 hour running mean
#cluster_option = 'SON.r30d24h' # r30d - remove 30 day running mean, 24h - 24 hour running mean
feature_path = f'../runs/transformer_embedding.s.e10b128.2001-2024.{data_option}.{model_option}.pt'
time_idx_path = f'../data/processed/preprocessed_data_time.2001-2024.{data_option}.csv'
linkage_path = f'../runs/linkage_metrix.s.e10b128.2014-2024.{data_option}.{model_option}.{cluster_option}.npy'
clustering_label_path = f'../runs/clustering_linkage.s.e10b128.2014-2024.{data_option}.{model_option}.{cluster_option}.nc'
#linkage_path = f'../runs/linkage_metrix.s.e10b128.2001-2024.{data_option}.{model_option}.{cluster_option}.npy'
#clustering_label_path = f'../runs/clustering_linkage.s.e10b128.2001-2024.{data_option}.{model_option}.{cluster_option}.nc'

# read features
feature = torch.load(feature_path, map_location=torch.device('cpu')).numpy()

time = pd.read_csv(time_idx_path, index_col=0).squeeze()
time = pd.to_datetime(time[:feature.shape[0]])

feature = xr.DataArray(feature, dims=['time', 'feature'], coords=[time, np.arange(feature.shape[1])])

feature = feature - feature.rolling(time=24*30, min_periods=2, center=True).mean()
feature = feature.rolling(time=24, min_periods=2, center=True).mean()
# feature = feature.rolling(time=12, min_periods=2, center=True).mean()

# xfeature = feature.sel(time=slice('2016-06-01', '2016-09-01'))
#xfeature = feature.sel(time=feature['time.month'].isin([6,7,8]))
#xfeature = feature.sel(time=feature['time.month'].isin([9,10,11]))
xfeature = feature.sel(time=slice('2014-01-01', '2024-12-31'))

Z = linkage(xfeature, 'ward')
np.save(linkage_path, Z)

# Z = np.load(linkage_path)

