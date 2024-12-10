import sys
#sys.path.append('miniconda3/envs/vlm-debiasing/lib/python3.12/site-packages')
import os
import pandas as pd
import numpy as np
import re
import scipy.io
from tqdm import tqdm
import re
#from scipy.interpolate import interp1d

def resize_to_fixed_length(array, target_length=700):
    # Truncate if array is too long
    if array.size > target_length:
        return array[:target_length]
    # Pad with zeros if array is too short
    elif array.size < target_length:
        padding = np.zeros(target_length - array.size)
        return np.concatenate([array, padding])
    # Return as-is if already the target length
    else:
        return array


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
            if os.path.isfile(file_path) and file_path.endswith("densenet201.csv"):
            #if os.path.isfile(file_path) and (file_path.endswith(".mat") or file_path.endswith(".csv")):
                data['file_path'].append(file_path)
                row = labels_csv[labels_csv["Participant"] == patient_num]
                
                for col in cols:
                    data[col].append(row[col].values[0])
            
    return pd.DataFrame(data)

def extract_and_save_visual_embeddings(
    file_paths, target_path, labels_csv_path, chunked=False
):
    """
    Extracts visual embeddings from .mat/.csv files, saves as .npy files, 
    and updates the given CSV file with the paths to the saved .npy files under the 'video' column.

    Parameters:
        file_paths (list): List of paths to the input .mat/.csv files.
        target_path (str): Directory to save the .npy files.
        labels_csv_path (str): Path to the existing CSV file to update.
        chunked (bool): Whether to split features into chunks when saving.
    """
    # Load the existing CSV file
    labels_csv = pd.read_csv(labels_csv_path)

    # Ensure "video" column exists in the DataFrame
    if "video" not in labels_csv.columns:
        labels_csv["video"] = None

    for file_path in tqdm(file_paths, desc="Extracting visual embeddings"):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        patient_number = re.match(r"^\d+", file_name).group()  # Extract patient number
        last_directory = os.path.basename(os.path.dirname(file_path))
        path_to_save = os.path.join(target_path, last_directory)
        os.makedirs(path_to_save, exist_ok=True)

        try:
            # Load the data
            if file_path.endswith(".mat"):
                data = scipy.io.loadmat(file_path)
                features = data["feature"]
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                features = df.to_numpy()  # Convert CSV data to NumPy array

            # Filter and resize features
            features = features[np.vectorize(lambda x: isinstance(x, (int, float, np.number)))(features)]
            #features = resize_to_fixed_length(features)
            #features = features.reshape(700, 1)

            if chunked:
                # Split features into chunks
                chunk_size = 100  # Adjust chunk size as needed
                for i in range(0, features.shape[0], chunk_size):
                    chunk = features[i:i + chunk_size]
                    if chunk.size == 0:
                        continue
                    chunk_file_name = f"{file_name}_chunk_{i // chunk_size}.npy"
                    np.save(os.path.join(path_to_save, chunk_file_name), chunk)
            else:
                # Save the entire feature array without chunking
                npy_file_name = f"{patient_number}_vis.npy"
                npy_file_path = os.path.join(path_to_save, npy_file_name)
                np.save(npy_file_path, features)
                print(f"Saved {npy_file_path}")

                # Update the "video" column in the CSV
                labels_csv.loc[labels_csv["file_path"] == file_path, "video"] = npy_file_path

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save the updated CSV back to the same path
    labels_csv.to_csv(labels_csv_path, index=False)
    print(f"Updated labels CSV saved to {labels_csv_path}")
    