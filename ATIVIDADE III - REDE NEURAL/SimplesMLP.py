import torch.nn as nn
from torchvision.ops import MLP

class SimplesMLP(nn.Module):
    def __init__(self, inputChannels, numClasses, hiddenChannels=[128, 64], dropout=0.2):
        super(SimplesMLP, self).__init__()
        
        self.mlp = nn.Sequential(
            MLP(
                in_channels=inputChannels,         # Dimensão da entrada
                hidden_channels=hiddenChannels,    # Camadas ocultas
                dropout=dropout
            ),
            nn.Linear(hiddenChannels[-1], numClasses)  # Camada de saída
        )

    def forward(self, x):
        return self.mlp(x)
