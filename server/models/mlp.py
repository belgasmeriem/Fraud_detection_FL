import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    """
    MLP — Multi-Layer Perceptron (baseline)
    =========================================
    Architecture améliorée avec BatchNorm et Dropout.
    Utilisé comme référence de comparaison avec CNN1D.

    IMPORTANT : pas de Sigmoid() final — on utilise BCEWithLogitsLoss
    qui inclut le Sigmoid en interne pour plus de stabilité numérique.
    """
    def __init__(self, input_dim: int = 37):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            # Pas de Sigmoid() — BCEWithLogitsLoss l'inclut
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
