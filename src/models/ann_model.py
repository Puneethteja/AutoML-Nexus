import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, hidden_units: int, output_dim: int):
        super(ANN, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def get_ann_model(input_dim: int, n_layers: int, hidden_units: int, output_dim: int) -> ANN:
    return ANN(input_dim, n_layers, hidden_units, output_dim)