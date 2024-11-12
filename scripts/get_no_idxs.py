import os
import numpy as np
import sys
sys.path.append('/home/hice1/awagh31/scratch/miniconda3/envs/vlm-debiasing/lib/python3.12/site-packages')
from tqdm import tqdm

def get_no_idxs(data_dir, dest_dir, modalities=["voice", "face"]):
    
    sessionIDs = sorted([x.split(".")[0] for x in os.listdir(data_dir)])

    for modality in modalities:
        os.makedirs(os.path.join(dest_dir, f"no_{modality}_idxs"), exist_ok=True)

    for sessionID in tqdm(sessionIDs, desc="Creating missing modality indices"):
        for modality in modalities:
            no_modality_path = os.path.join(dest_dir, f"no_{modality}_idxs", f"{sessionID}.npz")

            seq = np.array([], dtype=np.float32)
            np.savez_compressed(no_modality_path, data=seq)


data_dir = "./data/E-DAIC/original_data/"
dest_dir = "./data/E-DAIC/no-chunked/"
modalities = ["face"]
