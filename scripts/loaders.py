import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class MultiModalityDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load each modality as a numpy array and convert it to a torch tensor
        modality1 = torch.from_numpy(np.load(row['modality1_path'])).float()
        modality2 = torch.from_numpy(np.load(row['modality2_path'])).float()
        modality3 = torch.from_numpy(np.load(row['modality3_path'])).float()
        label = torch.tensor(row['label']).float()
        
        return [modality1, modality2, modality3], label

def collate_fn(batch):
    # Extract lists of modalities and labels from the batch
    modalities, labels = zip(*batch)
    
    # Stack each modality across batch dimension
    modality1 = torch.stack([m[0] for m in modalities], dim=0)
    modality2 = torch.stack([m[1] for m in modalities], dim=0)
    modality3 = torch.stack([m[2] for m in modalities], dim=0)
    labels = torch.stack(labels)
    
    # Return in the desired format: [modality1, modality2, modality3], labels
    return [modality1, modality2, modality3], labels