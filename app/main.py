import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
import numpy as np

# Imports depuis les packages du projet
from core import get_model, get_optimizer
from core.losses import get_loss
from pipeline import train, evaluate
from utils import load_and_preprocess

# -------------------------------
# CONFIGURATION
# -------------------------------
CSV_PATH = "data/patients_fake.csv"
FEATURE_COLS = ["age", "cholesterol"]
TARGET_COL = "label_infected"
BATCH_SIZE = 8
EPOCHS = 20
LR = 0.001
MODEL_TYPE = "logistic"  # "logistic" ou "nn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_TYPE = "bce"  # type de loss modulaire

# -------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------
(X_train, y_train), (X_test, y_test), scaler = load_and_preprocess(
    CSV_PATH, FEATURE_COLS, TARGET_COL
)

# -------------------------------
# DATASET & DATALOADER
# -------------------------------
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NumpyDataset(X_train, y_train)
test_dataset = NumpyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# MODEL & OPTIMIZER
# -------------------------------
input_dim = len(FEATURE_COLS)
model = get_model(MODEL_TYPE, input_dim)
optimizer = get_optimizer(model, lr=LR, optimizer_type="adam")
criterion = get_loss(LOSS_TYPE)  # <- utilisation de la loss modulaire

# -------------------------------
# TRAINING
# -------------------------------
print("Début de l'entraînement...\n")
train(model, train_loader, optimizer, epochs=EPOCHS, device=DEVICE, loss_type=LOSS_TYPE)

# -------------------------------
# EVALUATION
# -------------------------------
print("\nÉvaluation sur le jeu de test :")
evaluate(model, test_loader, device=DEVICE)

# -------------------------------
# EXEMPLE DE DIAGNOSTIC
# -------------------------------
patient_example = np.array([45, 6.2], dtype=np.float32)  # exemple patient
model.eval()
with torch.no_grad():
    x_tensor = torch.tensor(patient_example).unsqueeze(0).to(DEVICE)
    proba = model(x_tensor).item()
    diagnosis = "Infecté" if proba > 0.5 else "Sain"

print("\nExemple patient :", patient_example)
print(f"Probabilité d'infection : {proba:.3f}")
print(f"Diagnostic : {diagnosis}")
