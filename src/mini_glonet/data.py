from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_synthetic_ocean(
    num_steps: int,
    height: int,
    width: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate smooth, moving 2D fields with ocean-like spatial structure."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if height < 8 or width < 8:
        raise ValueError("height and width must both be at least 8")

    rng = np.random.default_rng(seed)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    fields = []
    for t in range(num_steps):
        phase = 2.0 * np.pi * t / 80.0

        cx1 = 0.45 * np.sin(phase)
        cy1 = 0.35 * np.cos(0.8 * phase)
        vortex1 = np.exp(
            -((xx - cx1) ** 2 / 0.16 + (yy - cy1) ** 2 / 0.10)
        )

        cx2 = -0.50 * np.cos(0.6 * phase)
        cy2 = 0.40 * np.sin(0.9 * phase)
        vortex2 = -0.75 * np.exp(
            -((xx - cx2) ** 2 / 0.12 + (yy - cy2) ** 2 / 0.18)
        )

        wave = 0.25 * np.sin(3.0 * xx + 2.0 * yy - phase)
        seasonal = 0.15 * np.cos(yy * np.pi + 0.4 * phase)
        noise = rng.normal(0.0, 0.015, size=(height, width))

        field = vortex1 + vortex2 + wave + seasonal + noise
        fields.append(field.astype(np.float32))

    return np.stack(fields, axis=0)


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float


class OceanSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create history-to-next-step forecasting samples."""

    def __init__(
        self,
        fields: np.ndarray,
        history: int,
        stats: NormalizationStats | None = None,
    ) -> None:
        if fields.ndim != 3:
            raise ValueError("fields must have shape [time, height, width]")
        if history < 1:
            raise ValueError("history must be positive")
        if len(fields) <= history:
            raise ValueError("not enough time steps for the requested history")

        self.history = history

        if stats is None:
            mean = float(fields.mean())
            std = float(fields.std())
            if std < 1e-8:
                raise ValueError("field standard deviation is too small")
            stats = NormalizationStats(mean=mean, std=std)

        self.stats = stats
        normalized = (fields - stats.mean) / stats.std
        self.fields = torch.from_numpy(normalized.astype(np.float32))

    def __len__(self) -> int:
        return len(self.fields) - self.history

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fields[index : index + self.history]
        y = self.fields[index + self.history].unsqueeze(0)
        return x, y


def split_fields(
    fields: np.ndarray,
    train_fraction: float,
    history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronologically split a time series while retaining validation context."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")

    split = int(len(fields) * train_fraction)
    if split <= history or len(fields) - split < 1:
        raise ValueError("split leaves too few samples")

    train = fields[:split]
    validation = fields[split - history :]
    return train, validation
