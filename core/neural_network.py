import torch.nn as nn

class ClinicalNeuralNet(nn.Module):
    """
    Petit réseau de neurones fully-connected pour données cliniques.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super(ClinicalNeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
