import sys
#sys.path.append('miniconda3/envs/vlm-debiasing/lib/python3.12/site-packages')
import os
import pandas as pd
import numpy as np
import re
import scipy.io
from tqdm import tqdm

from visual_embeddings import list_visual_files_paths, extract_and_save_visual_embeddings

# path to the detailed_labels.csv file from the edaic labels folder
#csv_path = "/home/hice1/awagh31/scratch/original/labels/labels/detailed_lables.csv" 
csv_path = "/home/hice1/mbibars3/scratch/vlm-debiasing/data/e-daic/original/labels/detailed_lables.csv" 
labels_df = pd.read_csv(csv_path)

# folder which contains each patient's data
#main_dir = "/home/hice1/awagh31/scratch/original/data_untarred" 
main_dir = "/home/hice1/mbibars3/scratch/vlm-debiasing/data/e-daic/untarred" 
cols = ["gender", "split", "PTSD_label", "age", "PTSD_severity"]

# List visual file paths with metadata
visual_paths_df = list_visual_files_paths(main_dir, labels_df, cols)
#print(visual_paths_df.head)

# the directory where we want to save paths and metadata to CSV for reference
"""
save_dir = '/home/hice1/awagh31/scratch/scripts'
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "labels_visual_files_dense.csv")
file_name = "labels_visual_files_dense.csv"
visual_paths_df.to_csv(os.path.join(csv_path), index=False)
#print(csv_path)
"""

# path to directory for final embeddings
#target_path = "/home/hice1/awagh31/scratch/final_embeddings_vis"
target_path = "/home/hice1/mbibars3/scratch/vlm-debiasing/data/e-daic/visual_2d"
visual_file_paths = visual_paths_df["file_path"]

# Extract and save embeddings, with optional chunking
extract_and_save_visual_embeddings(visual_file_paths, target_path, csv_path, chunked=False)
