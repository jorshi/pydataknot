# Integration test for outlier detection
import json
import os
from pathlib import Path

from hydra import initialize_config_module, compose
import pytest

from pydataknot import outlier_detection

test_data = Path(__file__).parent.joinpath("data")


@pytest.fixture
def rundir(tmp_path):
    current_dir = os.getcwd()
    yield tmp_path
    os.chdir(current_dir)


def test_outlier_detection(rundir):
    with initialize_config_module(version_base=None, config_module="pydataknot.config"):
        data_arg = f"data={test_data.joinpath('snare_headrim_dataset.json')}"
        cfg = compose(
            "outlier_detection_config",
            overrides=[
                data_arg,
            ],
        )

        os.chdir(rundir)
        outlier_detection.main(cfg)

        # Check output files
        model_path = "snare_headrim_dataset_outliers.json"
        assert Path(model_path).exists()
        with open(model_path) as f:
            trained_model = json.load(f)

        assert "outliers" in trained_model
        assert len(trained_model["outliers"]) == 2
        assert set(trained_model["outliers"]) == set([55, 100])
        assert trained_model["meta"]["info"]["outliers"] == 1
