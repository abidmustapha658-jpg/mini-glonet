from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> torch.Tensor:
    """Create a white-box FGSM adversarial input."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    if (clip_min is None) != (clip_max is None):
        raise ValueError("clip_min and clip_max must be provided together")

    x = inputs.detach().clone()
    x.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    prediction = model(x)
    loss = F.mse_loss(prediction, target)
    loss.backward()

    if x.grad is None:
        raise RuntimeError("input gradient was not computed")

    adversarial = x + epsilon * x.grad.sign()

    if clip_min is not None and clip_max is not None:
        adversarial = torch.clamp(
            adversarial,
            min=clip_min,
            max=clip_max,
        )

    return adversarial.detach()
