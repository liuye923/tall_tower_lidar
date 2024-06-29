import numpy as np
from data.dataloader_nc import dataloader_superres
import matplotlib.pyplot as plt
import tqdm

train_dataset = dataloader_superres( \
            "/pscatch/sd/j/jderm/ERA5_reduced/",
            "/dev/shm/derm950/HRRR_wind80m_reduced/",
            "./era_full_train.txt",
            pretraining=True)

mean = [list() for z in range(0,12)]
min = [list() for z in range(0,12)]
max = [list() for z in range(0,12)]
std = [list() for z in range(0,12)]

for i in tqdm.tqdm(range(5000)):
    _, x = train_dataset[i]
    
    for q in range(0,12):
        mean[q].append(np.mean(x[q].numpy()))
        min[q].append(np.min(x[q].numpy()))
        max[q].append(np.max(x[q].numpy()))
        std[q].append(np.std(x[q].numpy()))

for z in range(0,12):
    print(f"[{np.mean(mean[z])},  {np.mean(std[z])}],")

    
