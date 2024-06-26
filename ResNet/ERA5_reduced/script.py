import glob, sys,os,tqdm
from pathlib import Path
import multiprocessing as mp


def convert_file_batch(file):
    file_rp = file.split("/ERA5_processed/")[1]
    folder = '/'.join(file_rp.split('/')[:2])
    target_path = f"./{file_rp}" 
#
    myfile = Path(target_path)
    if not myfile.is_file():
        os.system(f'mkdir -p {folder}')
        os.system(f"cdo -sellonlatbox,-127,-116,41,50 {file} {target_path}>/dev/null")

    return True


files = glob.glob("/pscratch/sd/y/yeliu/MetOcean/ERA5_processed/[uvzt]*/**/*.nc")
#files = glob.glob("/pscratch/sd/y/yeliu/MetOcean/ERA5/e5.oper.an.sfc/**/*.nc")

print(len(files))
cks = 2
sublists =  [files[i:i+cks] for i in range(0, len(files), cks)]
#sublists =  [files[i:i+cks] for i in range(0, 10, cks)]

pool = mp.Pool(processes=cks)

for files in tqdm.tqdm(sublists):

    results = pool.map(convert_file_batch,files)
