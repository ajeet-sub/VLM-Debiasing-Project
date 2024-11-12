import os
import joblib
import numpy as np
from tqdm import tqdm

def process_video(videoID, modality, modality_dir, dest_dir, frame_step):
    dest_modality_dir = os.path.join(dest_dir, videoID.replace(".npz", ""), modality)
    os.makedirs(dest_modality_dir, exist_ok=True)

    seq_path = os.path.join(modality_dir, videoID)
    seq = np.load(seq_path)["data"]

    for start in range(0, len(seq), frame_step):
        end = min(start + frame_step, len(seq))
        chunk = seq[start:end]

        dest_path = os.path.join(dest_modality_dir, videoID.replace(".npz", f"_{str(start).zfill(6)}_{str(end).zfill(6)}.npz"))
        np.savez_compressed(dest_path, data=chunk)

def split_into_chunks(source_dir, dest_dir, nseconds, modalities, no_idxs_modalities, frame_rates):
    for modality, fps in zip(modalities, frame_rates):
        frame_step = fps * nseconds
        modality_dir = os.path.join(source_dir, modality)

        print(f"Processing modality: {modality}")
        videoIDs = sorted(os.listdir(modality_dir))

        joblib.Parallel(n_jobs=8)(
            joblib.delayed(process_video)(videoID, modality, modality_dir, dest_dir, frame_step) for videoID in videoIDs
        )

    for no_idxs_folder in no_idxs_modalities:
        no_idxs_folder_path = os.path.join(source_dir, no_idxs_folder)
        no_idxs_videoIDs = sorted(os.listdir(no_idxs_folder_path))

        for videoID in tqdm(no_idxs_videoIDs, desc=f"Processing no-idxs folder: {no_idxs_folder}", leave=False):
            dest_modality_dir = os.path.join(dest_dir, videoID.replace(".npz", ""), no_idxs_folder)
            os.makedirs(dest_modality_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_modality_dir, videoID)  
            
            np.savez_compressed(dest_path, data=np.load(os.path.join(no_idxs_folder_path, videoID))["data"])

source_dir = "./data/E-DAIC/no-chunked/"
dest_dir = "./data/E-DAIC/data/"
nseconds = 5
modalities = ["video_cnn_resnet", "video_cnn_vgg", "video_pose_gaze_aus"]
no_idxs_modalities = ["no_face_idxs", "no_voice_idxs"]
frame_rates = [30, 30, 30]  # Assuming each modality has a frame rate of 30 fps

#split_into_chunks(source_dir, dest_dir, nseconds, modalities, no_idxs