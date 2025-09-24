import json
import os
from pathlib import Path

from hydra import initialize_config_module, compose
import numpy as np
import pytest

from pydataknot import train_classifier

test_data = Path(__file__).parent.joinpath("data")


@pytest.fixture
def rundir(tmp_path):
    current_dir = os.getcwd()
    yield tmp_path
    os.chdir(current_dir)


def test_prepare_data(rundir):
    with initialize_config_module(version_base=None, config_module="pydataknot.config"):
        data_arg = f"data={test_data.joinpath('snare_headrim_dataset.json')}"
        cfg = compose(
            "classifier_config",
            overrides=[
                data_arg,
                "mlp.max_iter=2",
                "mlp.activation=1",
                "mlp.hidden_layers=[8,8]",
            ],
        )

        os.chdir(rundir)
        train_classifier.main(cfg)

        # Check output files
        model_path = "snare_headrim_dataset_pytrained.json"
        assert Path(model_path).exists()
        with open(model_path) as f:
            trained_model = json.load(f)

        assert "pythonclassifier" in trained_model
        classifier = trained_model["pythonclassifier"]
        assert len(classifier["labels"]["labels"]) == cfg.mlp.output_size
        assert set(classifier["labels"]["labels"]) == set(
            trained_model["meta"]["info"]["class_names"]
        )

        mlp = classifier["mlp"]
        layer1 = mlp["layers"][0]
        assert len(layer1["biases"]) == 8
        assert np.array(layer1["weights"]).shape == (104, 8)
        assert layer1["activation"] == 1
        assert layer1["rows"] == 104
        assert layer1["cols"] == 8

        layer2 = mlp["layers"][1]
        assert len(layer2["biases"]) == 8
        assert np.array(layer2["weights"]).shape == (8, 8)
        assert layer2["activation"] == 1
        assert layer2["rows"] == 8
        assert layer2["cols"] == 8

        layer3 = mlp["layers"][2]
        assert len(layer3["biases"]) == cfg.mlp.output_size
        assert np.array(layer3["weights"]).shape == (8, cfg.mlp.output_size)
        assert layer3["activation"] == 1
        assert layer3["rows"] == 8
        assert layer3["cols"] == cfg.mlp.output_size

        # Check normalizer output
        input_scaler = trained_model["input_scaler"]
        assert input_scaler["cols"] == cfg.mlp.input_size
        assert input_scaler["min"] == 0.0
        assert input_scaler["max"] == 1.0
        assert len(input_scaler["data_min"]) == cfg.mlp.input_size
        assert len(input_scaler["data_max"]) == cfg.mlp.input_size
