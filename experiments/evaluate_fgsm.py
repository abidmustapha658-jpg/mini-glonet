from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from mini_glonet.attacks import fgsm_attack
from mini_glonet.data import (
    NormalizationStats,
    OceanSequenceDataset,
    generate_synthetic_ocean,
    split_fields,
)
from mini_glonet.model import MiniGlonetCNN


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)))


def evaluate_fgsm(
    model: MiniGlonetCNN,
    loader: DataLoader,
    epsilon: float,
    std: float,
    clip_min: float,
    clip_max: float,
) -> float:
    predictions = []
    targets = []

    for inputs, target in loader:
        adversarial_inputs = fgsm_attack(
            model=model,
            inputs=inputs,
            target=target,
            epsilon=epsilon,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        with torch.no_grad():
            prediction = model(adversarial_inputs)

        predictions.append(prediction)
        targets.append(target)

    predictions_t = torch.cat(predictions)
    targets_t = torch.cat(targets)

    return rmse(predictions_t, targets_t) * std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument("--report-dir", default="reports/fgsm")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    data_cfg = config["data"]
    history = int(data_cfg["history"])

    fields = generate_synthetic_ocean(
        num_steps=int(data_cfg["num_steps"]),
        height=int(data_cfg["height"]),
        width=int(data_cfg["width"]),
        seed=int(config["seed"]),
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

    clip_min = float(dataset.fields.min())
    clip_max = float(dataset.fields.max())

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("")
    print("FGSM robustness evaluation")
    print("==========================")

    for epsilon in args.epsilons:
        attacked_rmse = evaluate_fgsm(
            model=model,
            loader=loader,
            epsilon=epsilon,
            std=stats.std,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        results.append(
            {
                "epsilon": epsilon,
                "rmse": attacked_rmse,
            }
        )

        print(f"epsilon={epsilon:.3f} | RMSE={attacked_rmse:.6f}")

    clean_rmse = results[0]["rmse"]

    for row in results:
        row["degradation_percent"] = (
            (row["rmse"] - clean_rmse)
            / clean_rmse
            * 100.0
        )

    with open(
        report_dir / "results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epsilon",
                "rmse",
                "degradation_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    epsilons = [row["epsilon"] for row in results]
    rmses = [row["rmse"] for row in results]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epsilons, rmses, marker="o")
    axis.set_xlabel("FGSM epsilon (normalized space)")
    axis.set_ylabel("RMSE")
    axis.set_title("Mini-GLONET robustness under FGSM")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(report_dir / "rmse_vs_epsilon.png", dpi=160)
    plt.close(figure)

    table_lines = [
        "| epsilon | RMSE | degradation vs clean |",
        "|---:|---:|---:|",
    ]

    for row in results:
        table_lines.append(
            f"| {row['epsilon']:.3f} | "
            f"{row['rmse']:.6f} | "
            f"{row['degradation_percent']:.1f}% |"
        )

    report_text = f"""# FGSM robustness experiment

## Objective

Evaluate Mini-GLONET against a controlled white-box FGSM input-manipulation attack.

The attack is:

`x_adv = x + epsilon * sign(dL/dx)`

The model weights are never updated during the attack.

## Setup

- Model: MiniGlonetCNN
- Dataset: synthetic ocean validation split
- History: {history} time steps
- Attack loss: MSE
- Attack: white-box FGSM
- Input clipping: normalized validation min/max
- Epsilon is measured in normalized input space

## Results

{chr(10).join(table_lines)}

![RMSE versus epsilon](rmse_vs_epsilon.png)

## Interpretation

Epsilon = 0 is the clean baseline. Higher epsilon values create input
perturbations chosen to increase the model loss. The RMSE increase measures
the sensitivity of Mini-GLONET to controlled adversarial input manipulation.

## Limitation

The current experiment uses synthetic normalized ocean fields. Epsilon is not
directly a temperature, salinity, or other physical ocean unit.
"""

    with open(
        report_dir / "README.md",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(report_text)

    print("")
    print(f"Results saved in: {report_dir}")


if __name__ == "__main__":
    main()
