import os
import sys
sys.path.append('/home/hice1/awagh31/scratch/miniconda3/envs/vlm-debiasing/lib/python3.12/site-packages')
import scipy.io
import numpy as np
from tqdm import tqdm

def prepare_cnn_resnet(src_root, dest_root, modality_id="video_cnn_resnet", featureID="CNN_ResNet", feature_dir="features"):
    dest_dir = os.path.join(dest_root, modality_id)
    os.makedirs(dest_dir, exist_ok=True)

    sessionIDs = sorted(os.listdir(src_root))
    for sessionID in tqdm(sessionIDs, desc="Processing sessions"):
        feature_path = os.path.join(src_root, sessionID, feature_dir, f"{sessionID}_{featureID}.mat")
        if not os.path.exists(feature_path):
            print(f"Feature file not found for session {sessionID}: {feature_path}")
            continue

        data = scipy.io.loadmat(feature_path)

        seq = data["feature"]

        dest_path = os.path.join(dest_dir, f"{sessionID}.npz")
        np.savez_compressed(dest_path, data=seq)


src_root = "./data/E-DAIC/data/"
dest_root = "./data/E-DAIC/no-chunked/"
modality_id = "video_cnn_resnet"
featureID = "CNN_ResNet"

#prepare_cnn_resnet(src_root, dest_root, modality_id, featureID)
