import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class MultiModalityDataset(Dataset):
    def __init__(self, df, modalities, label):
        self.df = df
        self.modalities = modalities
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load each modality as a numpy array and convert it to a torch tensor
        np_rows = []
        for modality in self.modalities:
            try:
                _modality = torch.from_numpy(np.load(row[modality], allow_pickle=True)).float()
            except Exception as e:
                _modality = torch.as_tensor(np.array(np.load(row[modality], allow_pickle=True)).astype('float'))
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