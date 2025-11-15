# base_model.py
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """À surcharger dans les modèles enfants"""
        raise NotImplementedError

    def predict(self, x):
        """Méthode de prédiction commune"""
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.sigmoid(outputs)

    def save(self, path: str):
        """Sauvegarde du modèle"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Chargement du modèle"""
        self.load_state_dict(torch.load(path, map_location="cpu"))
