import json
from pathlib import Path

import torch

from pydataknot.model import FluidMLP, regressor_from_dict

test_data = Path(__file__).parent.joinpath("data")


def test_init_from_dict():
    data_path = test_data.joinpath("snaretrainingcomplete.json")
    with open(data_path, "r") as fp:
        model_dict = json.load(fp)

    model_dict = model_dict["mlpclassifier"]["mlp"]
    mlp_1 = FluidMLP.from_dict(model_dict)
    mlp_2 = regressor_from_dict(model_dict)

    for layer_a, layer_b in zip(mlp_1.model.modules(), mlp_2.modules()):
        assert type(layer_a) is type(layer_b)
        if isinstance(layer_a, torch.nn.Linear):
            assert torch.allclose(layer_a.weight.data, layer_b.weight.data)
            assert torch.allclose(layer_a.bias.data, layer_b.bias.data)


def test_model_load():
    data_path = test_data.joinpath("snaretrainingcomplete.json")
    mlp = FluidMLP.load(data_path)

    l_a = [(89, 104), (74, 89), (59, 74), (44, 59), (29, 44), (10, 29)]
    l_b = [
        layer.weight.data.shape
        for layer in mlp.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    for a, b in zip(l_a, l_b):
        assert a == b

    data_path = test_data.joinpath("snaretrainingcomplete.json")
    with open(data_path, "r") as fp:
        model_dict = json.load(fp)

    model_dict = model_dict["mlpclassifier"]["mlp"]
    mlp_1 = FluidMLP.from_dict(model_dict)
    mlp_2 = regressor_from_dict(model_dict)

    x = torch.randn(1, 104)

    y_a = mlp(x)
    y_b = mlp_1(x)
    y_c = mlp_2(x)

    assert torch.allclose(y_a, y_b)
    assert torch.allclose(y_a, y_c)
