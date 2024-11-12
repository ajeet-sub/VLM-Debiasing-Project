import os
from tqdm import tqdm
import get_no_idxs, prepare_cnn_resnet, prepare_pose_gaze_aus, split_into_chunks

backup_dir = "./data/E-DAIC/backup/"
original_data_dir = "./data/E-DAIC/original_data/"
no_chunked_dir = "./data/E-DAIC/no-chunked/"
data_dir = "./data/E-DAIC/data/"

print("Generating no-modality indices...")
get_no_idxs(original_data_dir, no_chunked_dir)

print("Preparing CNN ResNet features...")
prepare_cnn_resnet(
    src_root=original_data_dir,
    dest_root=no_chunked_dir,
    modality_id="video_cnn_resnet",
    featureID="CNN_ResNet"
)

print("Splitting CNN ResNet features into chunks...")
split_into_chunks(
    source_dir=no_chunked_dir,
    dest_dir=data_dir,
    nseconds=5,
    modalities=["video_cnn_resnet"],
    no_idxs_modalities=["no_face_idxs"],
    frame_rates=[30] 

print("Preparing Pose, Gaze, and AU features...")
prepare_pose_gaze_aus(
    src_root=original_data_dir,
    dest_root=no_chunked_dir,
    modality_id="video_pose_gaze_aus",
    featureID="OpenFace2.1.0_Pose_gaze_AUs"
)

print("Splitting Pose, Gaze, and AU features into chunks...")
split_into_chunks(
    source_dir=no_chunked_dir,
    dest_dir=data_dir,
    nseconds=5,
    modalities=["video_pose_gaze_aus"],
    no_idxs_modalities=["no_face_idxs"],
    frame_rates=[30] 
)

print("Feature extraction and chunking process completed.")
