"""
CLI entry point for training the model.
"""

from pathlib import Path
from typing import Dict, List, Optional

import hydra
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch
from flucoma_torch.data import load_classifier_dateset, split_dataset_for_validation

from pydataknot.config import DKClassifierConfig
from pydataknot.data import load_data
from pydataknot.utils import save_trained_model


def select_features(
    features: Dict[str, List[float]], dataset: Dict, cfg: DKClassifierConfig
) -> Dict[str, List[float]]:
    # Check for existing feature selection in dataset input
    selected_features = []
    has_prior_selection = 0
    if "feature_select" in dataset["meta"]["info"]:
        has_prior_selection = dataset["meta"]["info"]["feature_select"]
        if has_prior_selection == 1:
            logger.info(f"Found prior feature selection {dataset['feature_select']}")
            selected_features = dataset["feature_select"]

    if cfg.features != "":
        if has_prior_selection:
            logger.warning("Prior feature selection in dataset will be overriddnen")

        if cfg.features == "all":
            logger.info("Using all features")
            return features, list()
        else:
            start, end = cfg.features.split("-")
            selected_features = list(range(int(start), int(end) + 1))
            logger.info(f"Selecting features from {start} to {end}")

    elif cfg.features == "" and not has_prior_selection:
        return features, list()

    dataset_selected = {"cols": 0, "data": {}}
    for key, value in features["data"].items():
        dataset_selected["data"][key] = [value[i] for i in selected_features]
        dataset_selected["cols"] = len(selected_features)

    return dataset_selected, selected_features


def setup_data(source: Dict, target: Dict, cfg: DKClassifierConfig) -> Dict:
    # Load the dataset
    # TODO: split dataset into validation as well.
    scaler = instantiate(cfg.scaler) if cfg.scaler else None
    train_dataset, source_scaler, labels = load_classifier_dateset(
        source_data=source,
        target_data=target,
        scaler=scaler,
    )
    logger.info(f"Loaded dataset with {len(train_dataset)} samples.")

    # Split dataset if using validation
    val_ratio = cfg.mlp.validation
    val_dataloader = None
    callbacks = []
    if val_ratio > 0.0:
        logger.info(f"Using a validation split ratio of {val_ratio}")
        # TODO: add a seed for valdiation
        train_dataset, val_dataset = split_dataset_for_validation(
            train_dataset, val_ratio
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.mlp.batch_size, shuffle=False
        )
        early_stopping = EarlyStopping("val_loss", patience=100)
        callbacks.append(early_stopping)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.mlp.batch_size, shuffle=True
    )

    data = {
        "train_dataloader": train_dataloader,
        "train_dataset": train_dataset,
        "val_dataloader": val_dataloader,
        "callbacks": callbacks,
        "scaler": source_scaler,
        "scaler_name": scaler.name if scaler is not None else "none",
        "labels": labels,
    }
    return data


def fit_model(
    cfg: DictConfig, data: Dict, extra_callbacks: Optional[List[L.Callback]] = None
):
    # Initialize the model
    cfg.mlp["input_size"] = data["train_dataset"][0][0].shape[0]
    cfg.mlp["output_size"] = data["train_dataset"][0][1].shape[0]
    mlp = instantiate(cfg.mlp)

    # Setup callbacks
    callbacks = []
    if data["callbacks"] is not None:
        callbacks.extend(data["callbacks"])
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    # Train the model
    trainer = L.Trainer(max_epochs=cfg.mlp.max_iter, callbacks=callbacks)
    logger.info("Starting training...")
    trainer.fit(mlp, data["train_dataloader"], val_dataloaders=data["val_dataloader"])

    return {"mlp": mlp, "trainer": trainer}


@hydra.main(version_base=None, config_name="classifier_config")
def main(cfg: DKClassifierConfig) -> None:
    logger.info("Starting training with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    dataset, labels, output = load_data(cfg)
    dataset, selected_features = select_features(dataset, output, cfg)
    data = setup_data(dataset, labels, cfg)

    # Create and fit the model
    fit = fit_model(cfg, data)

    # Save the model
    logger.info("Training complete. Saving model...")

    # MLPClassifier needs labels corresponding to the onehot
    # prediction along with the model weights.
    model_dict = fit["mlp"].model.get_as_dict()
    output_path = f"{Path(cfg.data).stem}_pytrained.json"
    save_trained_model(output_path, cfg, model_dict, data, selected_features, output)


if __name__ == "__main__":
    main()
