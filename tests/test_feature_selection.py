# Integration test for classifier optimization
import json
import os
from pathlib import Path

from hydra import initialize_config_module, compose
import pytest

from pydataknot import feature_selection

test_data = Path(__file__).parent.joinpath("data")


@pytest.fixture
def rundir(tmp_path):
    current_dir = os.getcwd()
    yield tmp_path
    os.chdir(current_dir)


def test_feature_select(rundir):
    with initialize_config_module(version_base=None, config_module="pydataknot.config"):
        data_arg = f"data={test_data.joinpath('snare_headrim_dataset.json')}"
        cfg = compose(
            "feature_select_config",
            overrides=[
                data_arg,
                "num_features=10",
            ],
        )

        os.chdir(rundir)
        feature_selection.main(cfg)

        # Check output files
        model_path = "snare_headrim_dataset_feature_select.json"
        assert Path(model_path).exists()
        with open(model_path) as f:
            trained_model = json.load(f)

        assert "feature_select" in trained_model
        assert len(trained_model["feature_select"]) == 10
        assert trained_model["meta"]["info"]["feature_select"] == 1
