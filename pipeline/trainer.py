import torch
from core.losses import get_loss

class Trainer:
    def __init__(self, model, train_loader, optimizer, loss_type="bce", device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device

        # Initialize loss function
        self.criterion = get_loss(loss_type)

        # Move model to device
        self.model.to(self.device)

    def train(self, epochs=20):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)

                # Backward
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

        return self.model
