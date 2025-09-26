# Integration test for classifier optimization
import json
import os
from pathlib import Path

from hydra import initialize_config_module, compose
import pytest

from pydataknot import optimize_classifier

test_data = Path(__file__).parent.joinpath("data")


@pytest.fixture
def rundir(tmp_path):
    current_dir = os.getcwd()
    yield tmp_path
    os.chdir(current_dir)


def test_optimize_classifier(rundir):
    with initialize_config_module(version_base=None, config_module="pydataknot.config"):
        data_arg = f"data={test_data.joinpath('snare_headrim_dataset.json')}"
        cfg = compose(
            "optimize_classifier_config",
            overrides=[
                data_arg,
                "mlp.max_iter=2",
                "n_trials=2",
                "optimize_features=true",
            ],
        )

        os.chdir(rundir)
        optimize_classifier.main(cfg)

        # Check output files
        model_path = "snare_headrim_dataset_optimized.json"
        assert Path(model_path).exists()
        with open(model_path) as f:
            trained_model = json.load(f)

        assert "pythonclassifier" in trained_model
        classifier = trained_model["pythonclassifier"]
        assert len(classifier["labels"]["labels"]) == cfg.mlp.output_size
        assert set(classifier["labels"]["labels"]) == set(
            trained_model["meta"]["info"]["class_names"]
        )
