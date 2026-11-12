import torch
from pydataknot.scaler import FluidNormalize


def test_normalize():
    x = torch.randn(2, 2)
    scaler = FluidNormalize()
    scaler.fit(x)
    y = scaler.transform(x)
    assert y.min() == 0.0
    assert y.max() == 1.0

    # Now with constant data
    x = torch.ones(2, 2)
    scaler.fit(x)
    y = scaler.transform(x)
    assert torch.all(y == 0.0)
