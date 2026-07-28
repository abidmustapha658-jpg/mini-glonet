from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_forecast_figure(
    truth: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Save truth, prediction and error as separate Matplotlib figures."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error = prediction - truth
    stem = output_path.stem
    suffix = output_path.suffix or ".png"

    figures = [
        ("truth", truth, f"{title} — truth"),
        ("prediction", prediction, f"{title} — prediction"),
        ("error", error, f"{title} — prediction error"),
    ]

    for name, field, figure_title in figures:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        image = ax.imshow(field, origin="lower", aspect="auto")
        ax.set_title(figure_title)
        ax.set_xlabel("Longitude index")
        ax.set_ylabel("Latitude index")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output_path.with_name(f"{stem}_{name}{suffix}"), dpi=160)
        plt.close(fig)


def save_training_curve(
    train_losses: list[float],
    validation_losses: list[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label="Train MSE")
    ax.plot(epochs, validation_losses, label="Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Mini-GLONET training curve")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
