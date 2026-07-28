from __future__ import annotations

import torch
from torch import nn


class MiniGlonetCNN(nn.Module):
    """Small residual CNN mapping a sequence of fields to the next field."""

    def __init__(self, history: int = 4, hidden_channels: int = 32) -> None:
        super().__init__()
        if history < 1:
            raise ValueError("history must be positive")
        if hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")

        self.history = history
        self.network = nn.Sequential(
            nn.Conv2d(history, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, history, height, width]")
        if x.shape[1] != self.history:
            raise ValueError(
                f"expected {self.history} history channels, got {x.shape[1]}"
            )

        persistence = x[:, -1:, :, :]
        correction = self.network(x)
        return persistence + correction
