from __future__ import annotations

import numpy as np


def poison_training_fields(
    fields: np.ndarray,
    poison_rate: float,
    noise_scale: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Poison a controlled fraction of training time maps.

    Selected maps receive zero-mean Gaussian corruption with standard
    deviation = noise_scale * std(clean training fields).

    The input array is never modified in place.

    Returns
    -------
    poisoned_fields:
        Copy of the fields after poisoning.
    poisoned_indices:
        Sorted time indices that were poisoned.
    """
    if fields.ndim != 3:
        raise ValueError("fields must have shape [time, height, width]")

    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must be in [0, 1]")

    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")

    poisoned = fields.copy()

    if poison_rate == 0.0 or noise_scale == 0.0:
        return poisoned, np.empty(0, dtype=np.int64)

    num_steps = len(fields)
    num_poisoned = int(round(num_steps * poison_rate))

    if num_poisoned == 0:
        num_poisoned = 1

    clean_std = float(fields.std())
    if clean_std < 1e-8:
        raise ValueError("field standard deviation is too small")

    rng = np.random.default_rng(seed)

    # Using one seeded permutation makes poison sets nested when the same
    # seed is reused with increasing poison rates.
    permutation = rng.permutation(num_steps)
    selected = permutation[:num_poisoned]

    noise_std = noise_scale * clean_std
    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=poisoned[selected].shape,
    ).astype(np.float32)

    poisoned[selected] = poisoned[selected] + noise

    return poisoned.astype(np.float32, copy=False), np.sort(selected)
