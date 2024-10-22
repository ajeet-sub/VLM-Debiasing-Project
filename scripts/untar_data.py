import os
from tqdm import tqdm
import argparse

# To use the script
# python ./scripts/untar_data.py --root-dir ./home/hice1/mbibars3/scratch/vlm-debiasing/e-daic_orig_data --dest-dir /home/hice1/mbibars3/scratch/vlm-debiasing/e-daic_untar_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root-dir", type=str, default="./vlm-debiasing/e-daic_orig_data")
    parser.add_argument("--dest-dir", type=str, default="./vlm-debiasing/e-daic_untar_data")
    args = parser.parse_args()
    
    tar_files = sorted( os.listdir(args.root_dir) )
    
    for tar_file in tqdm(tar_files):
       tar_path = os.path.join(args.root_dir, tar_file)
       os.system(f"tar -xf {tar_path} -C {args.dest_dir}")