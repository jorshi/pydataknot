import json
from typing import Any, Dict, List, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig


def filter_outliers(
    data: Dict[str, Any], outliers: List[int]
) -> Dict[str, List[float]]:
    """
    Iterate through dataset and labelset and remove indices marked as outliers
    """

    def no_outliers(item: Tuple[str, Any]):
        return int(item[0]) not in outliers

    return dict(filter(no_outliers, data.items()))


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

    # Remove outliers if they have been marked in the dataset
    should_filter_outliers = getattr(cfg, "filter_outliers", False)
    if should_filter_outliers and data["meta"]["info"].get("outliers", 0) == 1:
        outliers = data["outliers"]
        logger.info(f"Filtering out {len(outliers)} outliers.")

        dataset["data"] = filter_outliers(dataset["data"], outliers)
        labelset["data"] = filter_outliers(labelset["data"], outliers)

        # Verify that filtering worked as expected
        assert len(dataset["data"].keys()) == len(labelset["data"].keys())
        for key in dataset["data"].keys():
            assert key in labelset["data"]
            assert int(key) not in outliers

    return dataset, labelset, data
