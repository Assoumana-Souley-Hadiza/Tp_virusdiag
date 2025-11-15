# pipeline/evaluator.py

import torch
from utils.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model, test_loader, device="cpu"):
        self.model = model
        self.test_loader = test_loader
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self):
        y_true, y_pred = [], []

        for X_batch, y_batch in self.test_loader:
            X_batch = X_batch.to(self.device)

            # Forward
            outputs = self.model(X_batch).squeeze()

            # Sigmoid si nécessaire (selon modèle)
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            # Transformer logits -> probas si non fait dans le modèle
            if outputs.max() > 1 or outputs.min() < 0:
                outputs = torch.sigmoid(outputs)

            preds = (outputs > 0.5).float().cpu().numpy()

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(
            f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | "
            f"Recall: {rec:.3f} | F1-score: {f1:.3f}"
        )

        return acc, prec, rec, f1
