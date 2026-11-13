# Integration test for classifier optimization
import json
import os
from pathlib import Path

from hydra import initialize_config_module, compose

from pydataknot import optimize_classifier

test_data = Path(__file__).parents[1].joinpath("data")


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


def test_optimize_non_contiguous_features(rundir, mocker):
    # Regression test: the deep run used to re-apply the selected feature indices
    # to an already reduced dataset, which crashed (or picked the wrong columns)
    # whenever the best trial selected a non-contiguous subset of features.
    selected = [0, 5, 17]

    def fixed_selection(num_features, dataset, labels, cfg):
        dataset_selected = {
            "cols": len(selected),
            "data": {k: [v[i] for i in selected] for k, v in dataset["data"].items()},
        }
        return dataset_selected, list(selected)

    mocker.patch.object(
        optimize_classifier, "select_features_mrmr", side_effect=fixed_selection
    )

    with initialize_config_module(version_base=None, config_module="pydataknot.config"):
        data_arg = f"data={test_data.joinpath('snare_headrim_dataset.json')}"
        cfg = compose(
            "optimize_classifier_config",
            overrides=[
                data_arg,
                "mlp.max_iter=2",
                "n_trials=1",
                "include_default=false",
                "optimize_features=true",
                "deep_run=true",
                "deep_run_max_iters=2",
            ],
        )

        os.chdir(rundir)
        optimize_classifier.main(cfg)

    with open("snare_headrim_dataset_optimized.json") as f:
        trained_model = json.load(f)

    assert trained_model["feature_select"] == selected
    layers = trained_model["pythonclassifier"]["mlp"]["layers"]
    assert layers[0]["rows"] == len(selected)


def test_optimize_without_feature_optimization(rundir):
    # Regression test: with optimize_features=false the best trial has no feature
    # selection, which used to be written out as an empty selection and reduce the
    # dataset to zero columns after the study.
    with initialize_config_module(version_base=None, config_module="pydataknot.config"):
        data_arg = f"data={test_data.joinpath('snare_headrim_dataset.json')}"
        cfg = compose(
            "optimize_classifier_config",
            overrides=[
                data_arg,
                "mlp.max_iter=2",
                "n_trials=1",
                "optimize_features=false",
                "deep_run=true",
                "deep_run_max_iters=2",
            ],
        )

        os.chdir(rundir)
        optimize_classifier.main(cfg)

    with open("snare_headrim_dataset_optimized.json") as f:
        trained_model = json.load(f)

    # All features are used and no feature selection is recorded
    num_features = trained_model["dataset"]["cols"]
    layers = trained_model["pythonclassifier"]["mlp"]["layers"]
    assert layers[0]["rows"] == num_features
    assert "feature_select" not in trained_model
    assert trained_model["meta"]["info"].get("feature_select", 0) == 0
