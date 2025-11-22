import torch
import torch.nn as nn
from .model import BaseModel

class AttentionBlock(nn.Module):
    """
    Bloc d’attention feature-wise pour données tabulaires :
    - calcule un score pour chaque feature
    - pondère les features avant le passage dans le MLP
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()  # scores entre 0 et 1
        )

    def forward(self, x):
        attn_scores = self.attention(x)
        return x * attn_scores  # pondération élément par élément

class MedicalTabularModel(BaseModel):
    """
    Modèle tabulaire médical basé sur MLP + attention feature-wise.
    Hérite de BaseModel, donc compatible avec predict(), save(), load().
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.att_block = AttentionBlock(input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)  # sortie binaire
        )

    def forward(self, x):
        x = self.att_block(x)
        x = self.mlp(x)
        return x
