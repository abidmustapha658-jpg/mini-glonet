import numpy as np
import pytest

from mini_glonet.poisoning import poison_training_fields


def make_fields() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.normal(
        size=(100, 16, 20)
    ).astype(np.float32)


def test_zero_poison_rate_keeps_data_unchanged() -> None:
    fields = make_fields()

    poisoned, indices = poison_training_fields(
        fields,
        poison_rate=0.0,
        noise_scale=0.5,
        seed=42,
    )

    assert np.array_equal(poisoned, fields)
    assert len(indices) == 0


def test_poisoning_keeps_shape() -> None:
    fields = make_fields()

    poisoned, _ = poison_training_fields(
        fields,
        poison_rate=0.10,
        noise_scale=0.5,
        seed=42,
    )

    assert poisoned.shape == fields.shape


def test_poisoning_does_not_modify_original_array() -> None:
    fields = make_fields()
    original = fields.copy()

    _poisoned, _ = poison_training_fields(
        fields,
        poison_rate=0.10,
        noise_scale=0.5,
        seed=42,
    )

    assert np.array_equal(fields, original)


def test_poisoned_count_is_correct() -> None:
    fields = make_fields()

    _poisoned, indices = poison_training_fields(
        fields,
        poison_rate=0.10,
        noise_scale=0.5,
        seed=42,
    )

    assert len(indices) == 10


def test_unselected_maps_remain_unchanged() -> None:
    fields = make_fields()

    poisoned, indices = poison_training_fields(
        fields,
        poison_rate=0.10,
        noise_scale=0.5,
        seed=42,
    )

    clean_indices = np.setdiff1d(
        np.arange(len(fields)),
        indices,
    )

    assert np.array_equal(
        poisoned[clean_indices],
        fields[clean_indices],
    )


def test_poisoning_is_reproducible() -> None:
    fields = make_fields()

    poisoned_a, indices_a = poison_training_fields(
        fields,
        poison_rate=0.10,
        noise_scale=0.5,
        seed=42,
    )

    poisoned_b, indices_b = poison_training_fields(
        fields,
        poison_rate=0.10,
        noise_scale=0.5,
        seed=42,
    )

    assert np.array_equal(indices_a, indices_b)
    assert np.array_equal(poisoned_a, poisoned_b)


def test_larger_rate_contains_smaller_rate_indices() -> None:
    fields = make_fields()

    _, indices_small = poison_training_fields(
        fields,
        poison_rate=0.05,
        noise_scale=0.5,
        seed=42,
    )

    _, indices_large = poison_training_fields(
        fields,
        poison_rate=0.20,
        noise_scale=0.5,
        seed=42,
    )

    assert set(indices_small).issubset(set(indices_large))


def test_invalid_poison_rate_is_rejected() -> None:
    fields = make_fields()

    with pytest.raises(ValueError):
        poison_training_fields(
            fields,
            poison_rate=1.1,
        )
