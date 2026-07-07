import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        return z


if __name__ == "__main__":
    model = Encoder()
    x = torch.randn(2, 1, 32, 32)
    z = model(x)
    print(z.shape)