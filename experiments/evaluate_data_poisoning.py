from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
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
from mini_glonet.poisoning import poison_training_fields
from mini_glonet.train import run_epoch, set_seed


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_and_evaluate(
    config: dict,
    clean_train_fields,
    validation_fields,
    poison_rate: float,
    poison_strength: float,
) -> dict:
    seed = int(config["seed"])
    data_cfg = config["data"]
    training_cfg = config["training"]

    history = int(data_cfg["history"])

    # IMPORTANT:
    # Compute normalization ONCE from CLEAN training data.
    # Every poison rate reuses exactly the same normalization.
    clean_reference_dataset = OceanSequenceDataset(
        clean_train_fields,
        history=history,
    )
    clean_stats = clean_reference_dataset.stats

    poisoned_train_fields, poisoned_indices = poison_training_fields(
        clean_train_fields,
        poison_rate=poison_rate,
        noise_scale=poison_strength,
        seed=seed,
    )

    train_dataset = OceanSequenceDataset(
        poisoned_train_fields,
        history=history,
        stats=clean_stats,
    )

    # Validation is always completely clean.
    validation_dataset = OceanSequenceDataset(
        validation_fields,
        history=history,
        stats=clean_stats,
    )

    # Reset random state before every training run so that the only
    # intentional difference between experiments is the poisoned data.
    set_seed(seed)

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
        hidden_channels=int(
            config["model"]["hidden_channels"]
        ),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    criterion = nn.MSELoss()

    best_validation_mse = float("inf")
    best_epoch = -1
    final_train_mse = float("nan")

    print("")
    print(
        f"Poison rate = {poison_rate * 100:.1f}% "
        f"| poisoned maps = {len(poisoned_indices)}"
    )

    for epoch in range(
        1,
        int(training_cfg["epochs"]) + 1,
    ):
        train_mse = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        validation_mse = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        final_train_mse = train_mse

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch

        print(
            f"  Epoch {epoch:02d} "
            f"| train MSE={train_mse:.6f} "
            f"| clean val MSE={validation_mse:.6f}"
        )

    validation_rmse = (
        best_validation_mse ** 0.5
    ) * clean_stats.std

    return {
        "poison_rate": poison_rate,
        "poison_percent": poison_rate * 100.0,
        "num_poisoned_maps": len(poisoned_indices),
        "poison_strength_std": poison_strength,
        "best_epoch": best_epoch,
        "final_train_mse_normalized": final_train_mse,
        "best_validation_mse_normalized": best_validation_mse,
        "validation_rmse": validation_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--poison-rates",
        nargs="+",
        type=float,
        default=[
            0.0,
            0.01,
            0.05,
            0.10,
            0.20,
        ],
    )

    parser.add_argument(
        "--poison-strength",
        type=float,
        default=0.5,
        help=(
            "Gaussian poison noise std as a fraction "
            "of clean training-field std."
        ),
    )

    parser.add_argument(
        "--report-dir",
        default="reports/data-poisoning",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config["seed"])
    data_cfg = config["data"]
    history = int(data_cfg["history"])

    fields = generate_synthetic_ocean(
        num_steps=int(data_cfg["num_steps"]),
        height=int(data_cfg["height"]),
        width=int(data_cfg["width"]),
        seed=seed,
    )

    clean_train_fields, validation_fields = split_fields(
        fields,
        train_fraction=float(
            data_cfg["train_fraction"]
        ),
        history=history,
    )

    # Keep untouched copies so we can verify that the experiment never
    # changes the original clean arrays in place.
    clean_train_reference = clean_train_fields.copy()
    clean_validation_reference = validation_fields.copy()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("")
    print("==========================================")
    print(" DATA POISONING ROBUSTNESS EVALUATION")
    print("==========================================")
    print(
        f"Poison strength = "
        f"{args.poison_strength:.2f} x clean training std"
    )

    for poison_rate in args.poison_rates:
        result = train_and_evaluate(
            config=config,
            clean_train_fields=clean_train_fields,
            validation_fields=validation_fields,
            poison_rate=poison_rate,
            poison_strength=args.poison_strength,
        )

        results.append(result)

    if not torch.equal(
        torch.from_numpy(clean_train_fields),
        torch.from_numpy(clean_train_reference),
    ):
        raise RuntimeError(
            "Clean training data were modified in place."
        )

    if not torch.equal(
        torch.from_numpy(validation_fields),
        torch.from_numpy(clean_validation_reference),
    ):
        raise RuntimeError(
            "Validation data were modified."
        )

    clean_rmse = results[0]["validation_rmse"]

    for row in results:
        row["degradation_percent"] = (
            (
                row["validation_rmse"]
                - clean_rmse
            )
            / clean_rmse
            * 100.0
        )

    csv_path = report_dir / "results.csv"

    fieldnames = [
        "poison_rate",
        "poison_percent",
        "num_poisoned_maps",
        "poison_strength_std",
        "best_epoch",
        "final_train_mse_normalized",
        "best_validation_mse_normalized",
        "validation_rmse",
        "degradation_percent",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    poison_percents = [
        row["poison_percent"]
        for row in results
    ]

    validation_rmses = [
        row["validation_rmse"]
        for row in results
    ]

    figure, axis = plt.subplots(
        figsize=(7, 4.5)
    )

    axis.plot(
        poison_percents,
        validation_rmses,
        marker="o",
    )

    axis.set_xlabel(
        "Poisoned training maps (%)"
    )

    axis.set_ylabel(
        "Clean validation RMSE"
    )

    axis.set_title(
        "Mini-GLONET under data poisoning"
    )

    axis.grid(True)
    figure.tight_layout()

    figure.savefig(
        report_dir
        / "rmse_vs_poison_rate.png",
        dpi=160,
    )

    plt.close(figure)

    table_lines = [
        "| Poison rate | Poisoned maps | Validation RMSE | Degradation |",
        "|---:|---:|---:|---:|",
    ]

    for row in results:
        table_lines.append(
            f"| {row['poison_percent']:.1f}% "
            f"| {row['num_poisoned_maps']} "
            f"| {row['validation_rmse']:.6f} "
            f"| {row['degradation_percent']:.1f}% |"
        )

    report = f"""# Data Poisoning robustness experiment

## Objective

Evaluate the sensitivity of Mini-GLONET to controlled corruption of its
training data.

Unlike FGSM, which attacks the input at inference time, this experiment
modifies only a fraction of the training time maps and then trains a new model
from scratch.

## Experimental protocol

- Dataset: synthetic ocean fields
- Model: MiniGlonetCNN
- Training epochs per poison rate: {int(config['training']['epochs'])}
- Validation set: always clean
- Poison rates: {', '.join(f'{r * 100:.1f}%' for r in args.poison_rates)}
- Poison type: zero-mean Gaussian corruption
- Poison strength: {args.poison_strength:.2f} x clean training-field standard deviation
- Normalization statistics: computed from clean training data and reused for every experiment
- Random seed: {seed}

## Results

{chr(10).join(table_lines)}

![RMSE versus poison rate](rmse_vs_poison_rate.png)

## Interpretation

The 0% experiment is the clean reference. For higher poison rates, a
controlled fraction of training maps is corrupted before training while the
validation data remain unchanged.

An increase in clean-validation RMSE indicates that training-data poisoning
has degraded the learned forecasting model.

## Important limitation

This is a controlled academic experiment on synthetic ocean-like fields.
The current poisoning mechanism models generic corrupted observations; it is
not yet tied to a specific real ocean sensor or operational data pipeline.

## Reproduction

```powershell
python experiments/evaluate_data_poisoning.py `
    --config configs/base.yaml `
    --poison-rates 0 0.01 0.05 0.10 0.20 `
    --poison-strength 0.5
```
"""

    with open(
        report_dir / "README.md",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(report)

    print("")
    print("==========================================")
    print(" FINAL RESULTS")
    print("==========================================")

    for row in results:
        print(
            f"{row['poison_percent']:5.1f}% poison "
            f"| RMSE={row['validation_rmse']:.6f} "
            f"| degradation="
            f"{row['degradation_percent']:+.1f}%"
        )

    print("")
    print(f"CSV    : {csv_path}")
    print(
        "Figure : "
        f"{report_dir / 'rmse_vs_poison_rate.png'}"
    )
    print(
        "Report : "
        f"{report_dir / 'README.md'}"
    )


if __name__ == "__main__":
    main()
