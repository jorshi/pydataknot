"""
Use optuna to optimize a classifier for a particular dataset.
"""

from functools import partial
import json
import os
from pathlib import Path
from typing import Dict, Optional

from flucoma_torch.data import (
    convert_fluid_dataset_to_tensor,
    convert_fluid_labelset_to_tensor,
)
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.integration import PyTorchLightningPruningCallback
import torch

from pydataknot.config import DKOptimizeClassifierConfig
from pydataknot.data import load_data
import pydataknot.mrmr as mrmr
from pydataknot.train_classifier import fit_model, select_features, setup_data
from pydataknot.utils import save_trained_model


def select_features_mrmr(
    num_features: int, dataset: Dict, labels: Dict, cfg: DictConfig
):
    x = convert_fluid_dataset_to_tensor(dataset)
    y, _ = convert_fluid_labelset_to_tensor(labels)
    y = torch.argmax(y, dim=-1)

    scaler = instantiate(cfg.scaler) if cfg.scaler else None
    if scaler is not None:
        logger.info(f"Scaling dataset with {str(scaler)}")
        scaler.fit(x)
        x = scaler.transform(x)

    # Apply mRMR feature selection
    relevancy, redundancy = mrmr.relevancy_redundancy_clssif(x, y)
    selected_features = mrmr.select_features(num_features, relevancy, redundancy)
    selected_features = sorted(selected_features)

    dataset_selected = {"cols": 0, "data": {}}
    for key, value in dataset["data"].items():
        dataset_selected["data"][key] = [value[i] for i in selected_features]
        dataset_selected["cols"] = len(selected_features)

    return dataset_selected, selected_features


def objective(
    trial, cfg: DictConfig, artifact_store: Optional[FileSystemArtifactStore] = None
):
    # Model hyperparameters -- override the cfg
    n_layers = trial.suggest_int("n_layers", 1, 8)
    layers = []

    layer_sizes = [2**i for i in range(0, 9)]
    for i in range(n_layers):
        layers.append(trial.suggest_categorical(f"n_units_l{i}", layer_sizes))

    cfg.mlp.hidden_layers = layers
    cfg.mlp.activation = trial.suggest_int("activation", 0, 3)
    cfg.mlp.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64])
    cfg.mlp.learn_rate = trial.suggest_float("lr", 1e-6, 1.0, log=True)
    cfg.mlp.momentum = trial.suggest_float("momentum", 0.0, 1.0)

    # Optimize on val loss if it is being used
    metric = "val_loss" if cfg.mlp.validation > 0.0 else "train_loss"

    # Callback integration to enable optuna to prune trials
    callbacks = [PyTorchLightningPruningCallback(trial, monitor=metric)]

    # Reload the data and potentially study num features for mRMR
    dataset, labels, output = load_data(cfg)
    if cfg.optimize_features:
        if cfg.features != "":
            raise ValueError(
                "Manual feature selection not available with feature optimization"
            )
        num_features = trial.suggest_int("num_features", 1, dataset["cols"])
        dataset, selected_features = select_features_mrmr(
            num_features, dataset, labels, cfg
        )
    else:
        dataset, selected_features = select_features(dataset, output, cfg)
    data = setup_data(dataset, labels, cfg)

    # Fit the model
    fit = fit_model(cfg, data, extra_callbacks=callbacks)

    # Save model artefacts
    if artifact_store is not None:
        # Save the model json and selected features
        model_dict = {
            "mlp": fit["mlp"].model.get_as_dict(),
            "selected_features": selected_features,
            "scaler": data["scaler"],
        }
        model_path = "model.json"
        with open("model.json", "w") as f:
            json.dump(model_dict, f)

        # Upload as optuna artefact
        artifact_id = optuna.artifacts.upload_artifact(
            artifact_store=artifact_store,
            file_path=model_path,
            study_or_trial=trial,
        )
        trial.set_user_attr("model_artifact_id", artifact_id)

    return fit["trainer"].callback_metrics[metric]


@hydra.main(version_base=None, config_name="optimize_classifier_config")
def main(cfg: DKOptimizeClassifierConfig) -> None:
    logger.info("Starting hyperparameter optimization with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    dataset, labels, output = load_data(cfg)
    dataset, selected_features = select_features(dataset, output, cfg)
    data = setup_data(dataset, labels, cfg)

    # Optuna pruner will cancel trials that aren't looking good after a specified
    # number of trials and model warmup steps.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.n_startup_trials, n_warmup_steps=cfg.n_warmup_steps
    )

    # Create the artifact store
    base_path = "./artifacts"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)

    # Optional save the trial to a sqlite database
    storage = None
    if cfg.sqlite:
        storage = Path(cfg.storage_name)
        storage = f"sqlite:///{storage}.sqlite3"

    study = optuna.create_study(
        direction="minimize", pruner=pruner, storage=storage, study_name=cfg.study_name
    )

    # Run the study
    objective_func = partial(objective, cfg=cfg, artifact_store=artifact_store)
    study.optimize(objective_func, n_trials=cfg.n_trials)

    # Report
    logger.info("Number of finished trials: {}".format(len(study.trials)))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Get the best trained model
    best_artifact_id = study.best_trial.user_attrs.get("model_artifact_id")
    download_path = Path("best_model.json")

    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        artifact_id=best_artifact_id,
        file_path=download_path,
    )

    # Load the best model dict
    with open("best_model.json", "r") as fp:
        best_model = json.load(fp)

    selected_features = best_model["selected_features"]
    data["scaler"] = best_model["scaler"]
    model_dict = best_model["mlp"]

    output_path = f"{Path(cfg.data).stem}_optimized.json"
    save_trained_model(output_path, cfg, model_dict, data, selected_features, output)

    # Remove temporary files
    os.remove("best_model.json")
    os.remove("model.json")
