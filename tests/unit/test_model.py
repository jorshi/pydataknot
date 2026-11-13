import json
from pathlib import Path

import numpy as np
import torch

from pydataknot.model import FluidMLP
from pydataknot.utils import save_trained_model

test_data = Path(__file__).parents[1].joinpath("data")


def load_fixture_model_dict():
    with open(test_data.joinpath("snaretrainingcomplete.json"), "r") as fp:
        return json.load(fp)["mlpclassifier"]["mlp"]


NUMPY_ACTIVATIONS = {
    0: lambda x: x,
    1: lambda x: 1.0 / (1.0 + np.exp(-x)),
    2: lambda x: np.maximum(x, 0.0),
    3: np.tanh,
}


def reference_forward(model_dict, x):
    # Independent forward pass of a FluCoMa MLP computed directly from the
    # dictionary. FluCoMa stores weights as (rows=inputs, cols=outputs).
    for layer in model_dict["layers"]:
        x = x @ np.array(layer["weights"]) + np.array(layer["biases"])
        x = NUMPY_ACTIVATIONS[layer["activation"]](x)
    return x


def test_from_dict():
    model_dict = load_fixture_model_dict()
    mlp = FluidMLP.from_dict(model_dict)

    linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == len(model_dict["layers"])
    for linear, layer in zip(linears, model_dict["layers"]):
        assert linear.in_features == layer["rows"]
        assert linear.out_features == layer["cols"]
        weights = torch.tensor(layer["weights"]).T
        assert torch.allclose(linear.weight.data, weights)
        assert torch.allclose(linear.bias.data, torch.tensor(layer["biases"]))


def test_to_dict_round_trip():
    model_dict = load_fixture_model_dict()
    round_trip = FluidMLP.from_dict(model_dict).to_dict()

    assert len(round_trip["layers"]) == len(model_dict["layers"])
    for a, b in zip(round_trip["layers"], model_dict["layers"]):
        assert a.keys() == b.keys()
        for key in ["activation", "cols", "rows"]:
            assert a[key] == b[key]
        for key in ["weights", "biases"]:
            assert np.allclose(a[key], b[key], atol=1e-6)


def test_model_load():
    model_dict = load_fixture_model_dict()
    mlp = FluidMLP.load(test_data.joinpath("snaretrainingcomplete.json"))

    shapes = [
        m.weight.data.shape for m in mlp.modules() if isinstance(m, torch.nn.Linear)
    ]
    assert shapes == [(89, 104), (74, 89), (59, 74), (44, 59), (29, 44), (10, 29)]

    torch.manual_seed(0)
    x = torch.randn(4, 104)
    expected = reference_forward(model_dict, x.double().numpy())
    assert np.allclose(mlp(x).detach().numpy(), expected, atol=1e-5)


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
