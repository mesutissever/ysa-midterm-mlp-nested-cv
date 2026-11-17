"""PyTorch MLP: gizli katmanlar [Linear -> aktivasyon -> Dropout], çıkış tek logit."""
import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(self, n_inputs, hidden=(32,), activation="relu", dropout=0.0):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Bilinmeyen aktivasyon: {activation}")
        layers, prev = [], n_inputs
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))          # tek logit; sigmoid kayıp/tahminde uygulanır
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(model):
    """Öğrenilebilir parametre sayısı: Σ (girdi × çıktı + bias)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_params(n_inputs, hidden):
    """Formülle parametre sayısı (rapor/kontrol için)."""
    total, prev = 0, n_inputs
    for h in list(hidden) + [1]:
        total += prev * h + h
        prev = h
    return total
