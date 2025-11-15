import torch
from torch.utils.data import Dataset

class ClinicalDataset(Dataset):
    """
    Dataset personnalisé pour les données cliniques.
    """
    def __init__(self, dataframe, feature_cols, target_col, transform=None):
        self.data = dataframe
        self.X = dataframe[feature_cols].values
        self.y = dataframe[target_col].values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.transform:
            X = self.transform(X)
        return X, y
