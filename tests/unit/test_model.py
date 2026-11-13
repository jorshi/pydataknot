import json
from pathlib import Path

import torch

from pydataknot.model import FluidMLP, regressor_from_dict
from pydataknot.utils import save_trained_model

test_data = Path(__file__).parents[1].joinpath("data")


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


def test_model_load_python_trained(tmp_path):
    # Round trip through save_trained_model, which writes the pythonclassifier
    # entry in the _pytrained.json / _optimized.json outputs
    torch.manual_seed(0)
    mlp = FluidMLP(
        input_size=8,
        hidden_layers=[16, 4],
        output_size=3,
        activation=3,
        output_activation=1,
    )
    output = {"meta": {"info": {}}}
    data = {"labels": ["a", "b", "c"], "scaler": None}
    output_path = tmp_path.joinpath("model_pytrained.json")
    save_trained_model(output_path, None, mlp.to_dict(), data, [], output)

    loaded = FluidMLP.load(output_path, python_trained=True)

    x = torch.randn(5, 8)
    assert torch.allclose(mlp(x), loaded(x))
