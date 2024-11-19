import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class MultiModalityDataset(Dataset):
    def __init__(self, df, modalities, label, normalize=False):
        self.df = df
        self.modalities = modalities
        self.label = label
        self.normalize = normalize
        self.min_max_values = self._compute_min_max() if normalize else None

    def _compute_min_max(self):
        # Compute min and max for each modality across the entire dataset
        min_max = {modality: {"min": float("inf"), "max": float("-inf")} for modality in self.modalities}

        for idx, row in self.df.iterrows():
            for modality in self.modalities:
                data = np.load(row[modality], allow_pickle=True)
                min_max[modality]["min"] = min(min_max[modality]["min"], data.min())
                min_max[modality]["max"] = max(min_max[modality]["max"], data.max())

        return min_max

    def _normalize(self, data, modality):
        min_val = self.min_max_values[modality]["min"]
        max_val = self.min_max_values[modality]["max"]
        # Avoid division by zero
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return data  # Return unnormalized if all values are the same


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load each modality as a numpy array and convert it to a torch tensor
        np_rows = []
        for modality in self.modalities:
            data = np.load(row[modality], allow_pickle=True)
            if self.normalize:
                data = self._normalize(data, modality)
            _modality = torch.from_numpy(data).float()
            np_rows.append(_modality)
        label = torch.tensor(row[self.label]).float()
        return np_rows, label

def collate_fn(batch):
    # Extract lists of modalities and labels from the batch
    batch_data, labels = zip(*batch)
    num_modalities = len(batch_data[0])
    #print(modalities[0])
    data = []

    # Stack each modality across batch dimension
    for i in range(num_modalities):
        #print(f"============modality_{i}================")
        samples = [sample[i] for sample in batch_data]

        #print(np.unique([sample[i] for sample in batch_data]))
        _modality = torch.stack(samples, dim=0)
        data.append(_modality)

    labels = torch.stack(labels)
    
    # Return in the desired format: data = [modality1, modality2, modality3], labels
    return data, labels