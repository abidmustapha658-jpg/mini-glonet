import numpy as np
import torch

from mini_glonet.data import OceanSequenceDataset, generate_synthetic_ocean
from mini_glonet.model import MiniGlonetCNN


def test_dataset_shapes() -> None:
    fields = generate_synthetic_ocean(
        num_steps=20,
        height=16,
        width=24,
        seed=0,
    )
    dataset = OceanSequenceDataset(fields, history=4)
    x, y = dataset[0]

    assert x.shape == (4, 16, 24)
    assert y.shape == (1, 16, 24)
    assert len(dataset) == 16


def test_model_output_shape() -> None:
    model = MiniGlonetCNN(history=4, hidden_channels=8)
    x = torch.randn(3, 4, 16, 24)
    y = model(x)

    assert y.shape == (3, 1, 16, 24)
    assert np.isfinite(y.detach().numpy()).all()
