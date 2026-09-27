"""
Use optuna to optimize a classifier for a particular dataset.
"""

from functools import partial
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import hydra
from hydra.utils import instantiate
import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.integration import PyTorchLightningPruningCallback
from rich.console import Console
from rich.table import Table
import torch

from pydataknot.config import DKOptimizeClassifierConfig
import pydataknot.data as dkdata
from pydataknot.model import regressor_from_dict
import pydataknot.mrmr as mrmr
from pydataknot.train_classifier import fit_model, select_features, setup_data
from pydataknot.utils import save_trained_model, json_dump


def select_features_mrmr(
    num_features: int, dataset: Dict, labels: Dict, cfg: DictConfig
):
    x = dkdata.convert_fluid_dataset_to_tensor(dataset)
    y, _ = dkdata.convert_fluid_labelset_to_tensor(labels)
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
    cfg.mlp.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    cfg.mlp.learn_rate = trial.suggest_float("lr", 1e-6, 1.0, log=True)
    cfg.mlp.momentum = trial.suggest_float("momentum", 0.0, 1.0)

    # Optimize on val loss if it is being used
    metric = "val_acc" if cfg.mlp.validation > 0.0 else "train_loss"

    # Callback integration to enable optuna to prune trials
    callbacks = [PyTorchLightningPruningCallback(trial, monitor=metric)]

    # Reload the data and potentially study num features for mRMR
    dataset, labels, output = dkdata.load_data(cfg)
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


def perform_outlier_detection(dataset, outlier_detection, output, cfg):
    dataset = dkdata.convert_fluid_dataset_to_tensor(dataset)
    outliers, inliers = outlier_detection(dataset)
    outliers = outliers.numpy()

    logger.info(f"Found {len(outliers)} outliers.")
    if len(outliers) > 0:
        logger.info(f"Outlier data indices: {outliers}")

    # Add detected outliers to the incoming json file
    output["meta"]["info"]["outliers"] = 1
    output["outliers"] = [int(x) for x in outliers]

    output_name = Path(cfg.data).stem
    output_name = f"{output_name}_tmp.json"
    with open(output_name, "w") as f:
        f.write(json_dump(output, indent=4))

    return Path(output_name).absolute()


def print_hyperparameters(study: optuna.Study):
    table = Table(title="Star Wars Movies")

    table.add_column("Released", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Box Office", justify="right", style="green")

    table.add_row("Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$952,110,690")
    table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
    table.add_row("Dec 15, 2017", "Star Wars Ep. V111: The Last Jedi", "$1,332,539,889")
    table.add_row("Dec 16, 2016", "Rogue One: A Star Wars Story", "$1,332,439,889")

    console = Console()
    console.print(table)


@hydra.main(version_base=None, config_name="optimize_classifier_config")
def main(cfg: DKOptimizeClassifierConfig) -> None:
    logger.info("Starting hyperparameter optimization with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    dataset, labels, output = dkdata.load_data(cfg)

    prev_outliers = getattr(output["meta"]["info"], "outliers", 0)
    outlier_detection = instantiate(cfg.outlier)
    tmp_data_path = None
    original_data_path = cfg.data
    if prev_outliers == 0 and outlier_detection is not None:
        tmp_data_path = perform_outlier_detection(
            dataset, outlier_detection, output, cfg
        )
        cfg.data = tmp_data_path
        logger.info("Reloading data ...")
        dataset, labels, output = dkdata.load_data(cfg)

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

    # If validation is preset optimize on validation accuracy
    direction = "maximize" if cfg.mlp.validation > 0.0 else "minimize"
    study = optuna.create_study(
        direction=direction, pruner=pruner, storage=storage, study_name=cfg.study_name
    )

    # Add default parameters as our current best guess
    default_hyperparameters = {
        "activation": cfg.mlp.activation,
        "batch_size": 64,
        "lr": cfg.mlp.learn_rate,
        "momentum": cfg.mlp.momentum,
        "n_layers": len(cfg.mlp.hidden_layers),
        "num_features": data["train_dataset"][0][0].shape[0],
    }
    for i, layer in enumerate(cfg.mlp.hidden_layers):
        default_hyperparameters[f"n_units_l{i}"] = 2 ** math.ceil(math.log2(layer))

    study.enqueue_trial(default_hyperparameters)

    # Run the study
    objective_func = partial(objective, cfg=cfg, artifact_store=artifact_store)
    study.optimize(objective_func, n_trials=cfg.n_trials)

    # Report
    # TODO: use rich to print this nicer
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

    # ------
    # Starting deep run
    # ------

    # Reload data ...
    output["meta"]["info"]["feature_select"] = 1
    output["feature_select"] = selected_features
    dataset, selected_features = select_features(dataset, output, cfg)

    # cfg.mlp still holds the last trial's values -- restore the best trial's
    # before building the dataloaders, which use the batch size.
    params = trial.params
    cfg.mlp.max_iter = 1000
    cfg.mlp.learn_rate = params["lr"]
    cfg.mlp.momentum = params["momentum"]
    cfg.mlp.batch_size = params["batch_size"]
    cfg.mlp.activation = params["activation"]
    cfg.mlp.hidden_layers = [params[f"n_units_l{i}"] for i in range(params["n_layers"])]
    data = setup_data(dataset, labels, cfg)
    cfg.mlp.input_size = data["train_dataset"][0][0].shape[0]
    cfg.mlp.output_size = data["train_dataset"][0][1].shape[0]

    model = regressor_from_dict(model_dict)
    mlp = instantiate(cfg.mlp)
    mlp.model = model

    trainer = L.Trainer(max_epochs=cfg.mlp.max_iter, callbacks=data["callbacks"])

    # Sanity check -- the loaded model should reproduce the best trial's loss.
    # Without a validation split the trial optimized train_loss, so check on the
    # train set instead (reported by validate as val_loss).
    check_dataloader = data["val_dataloader"] or data["train_dataloader"]
    check_loss = trainer.validate(mlp, check_dataloader)[0]["val_loss"]
    logger.info(
        f"Loaded model loss: {check_loss:.6f} (best trial value: {trial.value:.6f})"
    )

    logger.info("Starting training...")
    trainer.fit(mlp, data["train_dataloader"], val_dataloaders=data["val_dataloader"])

    output_path = f"{Path(original_data_path).stem}_optimized.json"
    save_trained_model(output_path, cfg, model_dict, data, selected_features, output)

    # Remove temporary files
    os.remove("best_model.json")
    os.remove("model.json")
    if tmp_data_path is not None:
        os.remove(tmp_data_path)
