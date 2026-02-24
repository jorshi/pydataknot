import json

import hydra
from loguru import logger
from omegaconf import DictConfig


def load_data(cfg: DictConfig) -> None:
    """
    Load data and label files from the data json file.
    """
    logger.info(f"Preparing data with config:\n{cfg}")
    source = hydra.utils.to_absolute_path(cfg.data)
    with open(source, "r") as fp:
        data = json.load(fp)

    dataset = data["dataset"]
    labelset = data["labelset"]

    print(dataset["cols"])
    print(dataset["data"].keys())

    return dataset, labelset, data
