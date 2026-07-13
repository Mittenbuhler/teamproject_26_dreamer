"""
Decoder.py — Sprint 4
CNN-Decoder: (B, 32) Latent  →  (B, 1, 32, 32) Bild

Hinweis: output_padding=1 auf der letzten Schicht korrigiert die
  asymmetrische Stride-2-Ausdehnung (31→32 statt 31→31).
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (16, 7, 7)),                                    # (B, 16, 7, 7)
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2),             # (B, 8, 15, 15)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, output_padding=1),  # (B, 1, 32, 32)
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        Rückgabe: (B, 1, 32, 32) in [0, 1]
        """
        return self.net(z)


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = Decoder(latent_dim=32)
    z = torch.randn(2, 32)
    img = model(z)
    print(f"Input:  {z.shape}")
    print(f"Output: {img.shape}")