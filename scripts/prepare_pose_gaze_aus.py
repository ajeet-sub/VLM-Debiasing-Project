import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def prepare_pose_gaze_aus(src_root, dest_root, modality_id="video_pose_gaze_aus", featureID="OpenFace2.1.0_Pose_gaze_AUs", feature_dir="features"):
    dest_dir = os.path.join(dest_root, modality_id)
    os.makedirs(dest_dir, exist_ok=True)

    sessionIDs = sorted(os.listdir(src_root))
    for sessionID in tqdm(sessionIDs, desc="Processing sessions"):
        feature_path = os.path.join(src_root, sessionID, feature_dir, f"{sessionID}_{featureID}.csv")
        if not os.path.exists(feature_path):
            print(f"Feature file not found for session {sessionID}: {feature_path}")
            continue

        df = pd.read_csv(feature_path)

        seq = df.iloc[:, 4:].to_numpy()

        dest_path = os.path.join(dest_dir, f"{sessionID}.npz")
        np.savez_compressed(dest_path, data=seq)

src_root = "./data/E-DAIC/data/"
dest_root = "./data/E-DAIC/no-chunked/"
modality_id = "video_pose_gaze_aus"
featureID = "OpenFace2.1.0_Pose_gaze_AUs"

#prepare_pose_gaze_aus(src_root, dest_root, modality_id, featureID)