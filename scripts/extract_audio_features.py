from tqdm import tqdm
import librosa ###
import torch
import os
import pandas as pd
import numpy as np 
import re

def save_csv(csv, path_to_save, file_name):
    """
    save a dataframe (csv) as a filename file inside the path path_to_save
    if directories dont exist in the path to save the function will create them
    filename should be "name.csv"
    """
    # Check if the directory exists, and if not, create it
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    to_save = os.path.join(path_to_save, file_name)
    # Save the dataframe as a CSV file
    csv.to_csv(to_save, index=False)
    
def save_npy(path_to_save, npy_file):
    """
    npy_file should be name.npy
    path_to_save is self-explanatory
    """
    # Save the features as a .npy file
    np.save(path_to_save, npy_file)

def extract_features_AST(audio_paths, target_path, processor, model, pooling = False):
    """
    target_path : str
        The base directory where the processed audio files (with extracted features) will be saved.
        The function will create subdirectories based on the last directory in each input CSV path and save the corresponding files there.
    The folder heirarchy will be:
    - target_path
        - patient1 ==> last directory
            - patient1_..-AST.npy ==> name of original audio file
            - patient1_..-AST.npy
        - patient2
            - patient2_..-AST.npy
            - patient2_..-AST.npy
        - patient3

    Returns:
    --------
    None

    """
    for wav_file_path in tqdm(audio_paths):
        y, sr = librosa.load(wav_file_path, sr=16000)
        inputs = processor(raw_speech=y, sampling_rate=sr, return_tensors="pt")
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the 9th layer hidden states and flatten to give 1D array
        layer_9_features = outputs.hidden_states[9].squeeze().numpy() 
        #print(layer_9_features.shape)
        # Get the last directory in the path
        last_directory = os.path.basename(os.path.dirname(wav_file_path))
        # Get the CSV file name
        file_name = os.path.splitext(os.path.basename(wav_file_path))[0]
        path_to_save = os.path.join(target_path, last_directory)
        if pooling:
            layer_9_features = layer_9_features.mean(0)[:,np.newaxis]  # Reduces to shape (1, 768)
            _saving_name = file_name+".npy"
        else:
            _saving_name = file_name+".npy"
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        path_to_save = os.path.join(path_to_save,_saving_name)
        # Save to numpy
        save_npy(path_to_save,layer_9_features)

def save_csv(csv, path_to_save, file_name):
    """
    save a dataframe (csv) as a filename file inside the path path_to_save
    if directories dont exist in the path to save the function will create them
    filename should be "name.csv"
    """
    # Check if the directory exists, and if not, create it
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    to_save = os.path.join(path_to_save, file_name)
    # Save the dataframe as a CSV file
    csv.to_csv(to_save, index=False)

def list_audio_files_paths(main_dir, labels_csv, cols):
    """
    This function creates a dataframe that contains the paths to files with the patient name and class label
    for each file, given a path to a main_folder.
    Input:
    - The labels_csv should be a pd.DataFrame that has rows for each patient and the patient's label: 
        patient, label, ptsd_severity,...
    - cols should be:
        list of columns to look at from the labels_csv
        ex: ["Participant","PHQ8_1_NoInterest","PHQ8_2_Depressed"]
    - main folder path

    The folder heirarchy should be:
    - main_folder
        - patient1_P
            - patient1_..-.wav
            - patient1_..-.wav
        - patient2_P
            - patient2_..-.wav
        - patient3_P
    Output:
    - a dataframe that contains the paths to files with the label, age, gender
    {'subject_name': [patient1, patient1, patient2,], 
     'class': [0,5,0,], 'file_path': "__"}
    """
    ## Create columns dynamically
    _cols = cols + ["file_path"]
    data = {_col:[] for _col in _cols}

    patient_folders = os.listdir(main_dir)
    # Loop over each patient directory in the main directory
    for _patient_folder in tqdm(patient_folders):
        ## extract the number from folder name 300_P ==> 300
        patient_num = int(re.search(r"(\d+)_P", _patient_folder).group(1))
        patient_path = os.path.join(main_dir, _patient_folder)
        
        # Check if it's a directory
        if os.path.isdir(patient_path):
            # Loop over each file in the patient directory
            for patient_sample in os.listdir(patient_path):
                _sample_path = os.path.join(patient_path, patient_sample)
                # Check if it's a file
                if os.path.isfile(_sample_path) and _sample_path.lower().endswith(".wav"):
                    # Check its a wav file too!
                    data['file_path'].append(_sample_path)
                    row = labels_csv[labels_csv["Participant"] == patient_num]
                    for col in cols:
                        data[col].append(row[col].values[0])
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    return df
    
        
            

