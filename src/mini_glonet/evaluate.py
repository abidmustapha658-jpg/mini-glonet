from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from mini_glonet.data import (
    NormalizationStats,
    OceanSequenceDataset,
    generate_synthetic_ocean,
    split_fields,
)
from mini_glonet.model import MiniGlonetCNN
from mini_glonet.visualization import save_forecast_figure


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    seed = int(config["seed"])
    data_cfg = config["data"]
    history = int(data_cfg["history"])

    fields = generate_synthetic_ocean(
        num_steps=int(data_cfg["num_steps"]),
        height=int(data_cfg["height"]),
        width=int(data_cfg["width"]),
        seed=seed,
    )
    _, validation_fields = split_fields(
        fields,
        train_fraction=float(data_cfg["train_fraction"]),
        history=history,
    )

    normalization = checkpoint["normalization"]
    stats = NormalizationStats(
        mean=float(normalization["mean"]),
        std=float(normalization["std"]),
    )
    dataset = OceanSequenceDataset(
        validation_fields,
        history=history,
        stats=stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
    )

    model = MiniGlonetCNN(
        history=history,
        hidden_channels=int(config["model"]["hidden_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    targets = []
    persistence_predictions = []

    with torch.no_grad():
        for inputs, target in loader:
            prediction = model(inputs)
            predictions.append(prediction)
            targets.append(target)
            persistence_predictions.append(inputs[:, -1:, :, :])

    predictions_t = torch.cat(predictions)
    targets_t = torch.cat(targets)
    persistence_t = torch.cat(persistence_predictions)

    model_rmse_normalized = rmse(predictions_t, targets_t)
    persistence_rmse_normalized = rmse(persistence_t, targets_t)

    model_rmse = model_rmse_normalized * stats.std
    persistence_rmse = persistence_rmse_normalized * stats.std

    print(f"Persistence RMSE: {persistence_rmse:.6f}")
    print(f"Mini-GLONET RMSE: {model_rmse:.6f}")

    figure_dir = Path(config["training"]["output_dir"]) / "figures"
    num_examples = min(
        int(config["evaluation"]["num_examples"]),
        len(predictions_t),
    )

    for index in range(num_examples):
        truth = targets_t[index, 0].numpy() * stats.std + stats.mean
        prediction = (
            predictions_t[index, 0].numpy() * stats.std + stats.mean
        )
        save_forecast_figure(
            truth=truth,
            prediction=prediction,
            output_path=figure_dir / f"forecast_{index:02d}.png",
            title=f"Forecast example {index}",
        )


if __name__ == "__main__":
    main()
