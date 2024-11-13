import sys
#sys.path.append('miniconda3/envs/vlm-debiasing/lib/python3.12/site-packages')
import os
import pandas as pd
import numpy as np
import re
import scipy.io
from tqdm import tqdm

def list_visual_files_paths(main_dir, labels_csv, cols):
    """
    Lists paths for visual files (e.g., .mat or .csv files) in the main_dir, and adds metadata from labels_csv.
    """
    # Prepare a dictionary with columns for metadata and file paths
    _cols = cols + ["file_path"]
    data = {_col: [] for _col in _cols}

    patient_folders = os.listdir(main_dir)
    #print(patient_folders)
    
    for folder in tqdm(patient_folders, desc="Listing visual files"):
        patient_num = int(re.search(r"(\d+)_P", folder).group(1))
        patient_path = os.path.join(main_dir, folder, "features")
        #print(patient_path)
        
        # Process each file in the patient directory
        for file in os.listdir(patient_path):
            file_path = os.path.join(patient_path, file)
            #print(file_path)
            if os.path.isfile(file_path) and file_path.endswith("BoVW_openFace_2.1.0_Pose_Gaze_AUs.csv"):
            #if os.path.isfile(file_path) and (file_path.endswith(".mat") or file_path.endswith(".csv")):
                data['file_path'].append(file_path)
                row = labels_csv[labels_csv["Participant"] == patient_num]
                
                for col in cols:
                    data[col].append(row[col].values[0])
            
    return pd.DataFrame(data)

def extract_and_save_visual_embeddings(file_paths, target_path, chunked=False):
    """
    Extracts visual embeddings from .mat/.csv files, optionally splits them into chunks, and saves as .npy files.
    """
    for file_path in tqdm(file_paths, desc="Extracting visual embeddings"):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        #print(file_name)
        last_directory = os.path.basename(os.path.dirname(file_path))
        path_to_save = os.path.join(target_path, last_directory, file_name)
        #print(path_to_save)
        os.makedirs(path_to_save, exist_ok=True)
        try:
            if file_path.endswith(".mat"):
                data = scipy.io.loadmat(file_path)
                features = data["feature"] 
                
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                features = df.to_numpy()  # Convert CSV data to NumPy array for consistency
            
            if chunked:
                # Split features into chunks
                chunk_size = 100  # Adjust
                for i in range(0, features.shape[0], chunk_size):
                    chunk = features[i:i + chunk_size]
                    if chunk.size == 0:
                        continue
                    chunk_file_name = f"{file_name}_chunk_{i // chunk_size}.npy"
                    np.save(os.path.join(path_to_save, chunk_file_name), chunk)
            else:
                # Save the entire feature array without chunking
                np.save(os.path.join(path_to_save, f"{file_name}.npy"), features)
                print(f"saved {path_to_save}, {file_name}.npy")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

