import pytest
import torch

from mini_glonet.attacks import fgsm_attack
from mini_glonet.model import MiniGlonetCNN


def make_example():
    torch.manual_seed(7)
    model = MiniGlonetCNN(history=4, hidden_channels=8)
    inputs = torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    return model, inputs, target


def test_fgsm_keeps_shape() -> None:
    model, inputs, target = make_example()
    adversarial = fgsm_attack(model, inputs, target, epsilon=0.01)
    assert adversarial.shape == inputs.shape


def test_fgsm_zero_epsilon_keeps_input_unchanged() -> None:
    model, inputs, target = make_example()
    adversarial = fgsm_attack(model, inputs, target, epsilon=0.0)
    assert torch.equal(adversarial, inputs)


def test_fgsm_does_not_modify_original_input() -> None:
    model, inputs, target = make_example()
    original = inputs.clone()
    _ = fgsm_attack(model, inputs, target, epsilon=0.02)
    assert torch.equal(inputs, original)


def test_fgsm_respects_epsilon_without_clipping() -> None:
    model, inputs, target = make_example()
    epsilon = 0.02
    adversarial = fgsm_attack(model, inputs, target, epsilon=epsilon)
    maximum_change = (adversarial - inputs).abs().max().item()
    assert maximum_change <= epsilon + 1e-6


def test_fgsm_rejects_negative_epsilon() -> None:
    model, inputs, target = make_example()
    with pytest.raises(ValueError):
        fgsm_attack(model, inputs, target, epsilon=-0.01)
