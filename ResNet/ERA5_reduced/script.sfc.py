import glob, sys,os,tqdm
from pathlib import Path
import multiprocessing as mp


def convert_file_batch(file):
    file_rp = file.split("/e5.oper.an.sfc/")[1]
    folder = 'sfc/' + '/'.join(file_rp.split('/')[:1])
    target_path = f"./sfc/{file_rp}" 
#
    myfile = Path(target_path)
    if myfile.is_file():
        if myfile.stat().st_size < 1 * 1024 * 1024: os.system(f'rm -f {myfile}')
       
    if not myfile.is_file():
        os.system(f'mkdir -p {folder}')
#        print(f"cdo -sellonlatbox,-127,-116,41,50 {file} {target_path}")
        os.system(f"cdo -sellonlatbox,-127,-116,41,50 {file} {target_path}>/dev/null")

    return True


#files = glob.glob("/pscratch/sd/y/yeliu/MetOcean/ERA5_processed/[uvzt]*/**/*.nc")
files = glob.glob("/pscratch/sd/y/yeliu/MetOcean/ERA5/e5.oper.an.sfc/**/*.nc")

print(len(files))
cks = 128
sublists =  [files[i:i+cks] for i in range(0, len(files), cks)]
#sublists =  [files[i:i+cks] for i in range(0, 10, cks)]

pool = mp.Pool(processes=cks)

for files in tqdm.tqdm(sublists):

    results = pool.map(convert_file_batch,files)
