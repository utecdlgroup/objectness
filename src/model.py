import torch.nn as nn

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 10, 3, padding=1),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        y_hat = self.model(x).reshape(-1, 10)
        return y_hat