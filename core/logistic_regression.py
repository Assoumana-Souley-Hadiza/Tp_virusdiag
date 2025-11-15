# logistic_regression.py
import torch.nn as nn
from .model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
