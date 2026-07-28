# Mini-GLONET

A small, reproducible ocean forecasting project inspired by the regionalisation workflow of GLONET.

## Goal

Predict the next 2D ocean field from the four previous fields:

\[
X_{t-3}, X_{t-2}, X_{t-1}, X_t \rightarrow \hat{X}_{t+1}
\]

The first version uses synthetic ocean-like data so the full PyTorch pipeline can be tested before downloading real NetCDF data.

## Current baseline

- Synthetic moving ocean fields
- PyTorch `Dataset` and `DataLoader`
- Persistence baseline: \(\hat{X}_{t+1}=X_t\)
- Small convolutional neural network
- MSE training loss
- RMSE evaluation
- Matplotlib visualisation of truth, prediction, and error

## Project structure

```text
mini-glonet/
├── configs/
│   └── base.yaml
├── src/
│   └── mini_glonet/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       └── visualization.py
├── tests/
│   └── test_shapes.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Train

```bash
python -m mini_glonet.train --config configs/base.yaml
```

## Evaluate

```bash
python -m mini_glonet.evaluate \
  --config configs/base.yaml \
  --checkpoint outputs/best_model.pt
```

The evaluation script writes figures into `outputs/figures/`.

## Roadmap

1. Validate the synthetic forecasting pipeline.
2. Replace synthetic fields with real NetCDF data using `xarray`.
3. Add land/sea masks and missing-value handling.
4. Compare persistence, training from scratch, and fine-tuning.
5. Pre-train on a large coarse region.
6. Fine-tune on an IBI-like regional domain.
7. Add multi-variable forecasting and simple physical regularisation.
