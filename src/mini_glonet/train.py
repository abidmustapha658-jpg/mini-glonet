from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from mini_glonet.data import (
    OceanSequenceDataset,
    generate_synthetic_ocean,
    split_fields,
)
from mini_glonet.model import MiniGlonetCNN
from mini_glonet.visualization import save_training_curve


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(inputs)
            loss = criterion(predictions, targets)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = inputs.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["seed"])
    set_seed(seed)

    data_cfg = config["data"]
    fields = generate_synthetic_ocean(
        num_steps=int(data_cfg["num_steps"]),
        height=int(data_cfg["height"]),
        width=int(data_cfg["width"]),
        seed=seed,
    )

    history = int(data_cfg["history"])
    train_fields, validation_fields = split_fields(
        fields,
        train_fraction=float(data_cfg["train_fraction"]),
        history=history,
    )

    train_dataset = OceanSequenceDataset(train_fields, history=history)
    validation_dataset = OceanSequenceDataset(
        validation_fields,
        history=history,
        stats=train_dataset.stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
    )

    model = MiniGlonetCNN(
        history=history,
        hidden_channels=int(config["model"]["hidden_channels"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = nn.MSELoss()

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation = float("inf")

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_loss = run_epoch(
            model, validation_loader, criterion, device
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization": {
                        "mean": train_dataset.stats.mean,
                        "std": train_dataset.stats.std,
                    },
                    "config": config,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"train MSE={train_loss:.6f} | "
            f"validation MSE={validation_loss:.6f}"
        )

    save_training_curve(
        train_losses,
        validation_losses,
        output_dir / "training_curve.png",
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
