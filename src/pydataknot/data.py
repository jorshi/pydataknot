import json
from typing import Any, Dict, List, Optional, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch

from pydataknot.scaler import FluidBaseScaler


class FluidDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for Fluid data.
    """

    def __init__(self, source: torch.Tensor, target: torch.Tensor):
        """
        Initialize the dataset with data and targets.

        :param data: Input data as a tensor.
        :param targets: Target values as a tensor.
        """
        assert source.ndim == 2, "Source data should be a 2D tensor."
        assert target.ndim == 2, "Target data should be a 2D tensor."
        assert (
            source.shape[0] == target.shape[0]
        ), "Source and target must have the same number of samples."

        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


def convert_fluid_dataset_to_tensor(fluid_data: Dict):
    data = []

    # Sort the keys to ensure consistent order
    keys = sorted([int(i) for i in fluid_data["data"].keys()])
    for key in keys:
        data.append(fluid_data["data"][str(key)])

    if len(data) == 0:
        raise ValueError("No data found in the fluid dataset.")

    data = torch.tensor(data, dtype=torch.float32)
    assert data.ndim == 2, "Data should be a 2D tensor."
    assert (
        data.shape[1] == fluid_data["cols"]
    ), f"Data shape mismatch: expected {fluid_data['cols']} columns, got {data.shape[1]}."
    return data


def convert_fluid_labelset_to_tensor(fluid_data: Dict):
    """
    Create a one-hot encoded tensor from the labels.
    """
    # Assert that there is only one col in the data -- we assume that there is only
    # one label for each datapoint.
    assert fluid_data["cols"] == 1, "Expcted labelset to have one column only"

    # Sort the labeles -- this isn't exactly what FluCoMa does, but as long as the
    # order of the labels is correct in the classifier dict then we should be good.
    keys = sorted([int(i) for i in fluid_data["data"].keys()])
    labels = sorted(list(set(fluid_data["data"][str(k)][0] for k in keys)))
    assert len(labels) > 1, "Only a single label found!"

    data = []

    for key in keys:
        label_idx = labels.index(fluid_data["data"][str(key)][0])
        onehot = torch.zeros(len(labels))
        onehot[label_idx] = 1.0
        data.append(onehot)

    data = torch.vstack(data)
    assert data.ndim == 2, "Data should be a 2D tensor."
    assert data.shape[1] == len(
        labels
    ), f"Data shape mismatch: expected {len(labels)} columns, got {data.shape[1]}."
    return data, labels


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


def load_classifier_dateset(
    source_data: Dict,
    target_data: Dict,
    scaler: Optional[FluidBaseScaler] = None,
):
    """
    Load source and target datasets from dictionaries
    """
    source_data = convert_fluid_dataset_to_tensor(source_data)
    target_data, target_labels = convert_fluid_labelset_to_tensor(target_data)

    if source_data.shape[0] != target_data.shape[0]:
        raise ValueError(
            "Source and target datasets must have the same number of samples."
        )

    # Apply scaler to input if needed
    source_scaler_dict = None
    if scaler is not None:
        scaler.fit(source_data)
        source_data = scaler.transform(source_data)
        source_scaler_dict = scaler.get_as_dict()

    dataset = FluidDataset(source_data, target_data)
    return dataset, source_scaler_dict, target_labels


def split_dataset_for_validation(
    dataset: FluidDataset, val_ratio: float, seed: int = 42
):
    assert 0.0 < val_ratio < 1.0, "Expected val_ratio to be between 0.0 and 1.0"

    num_data = len(dataset)
    assert num_data > 1, "Expected a dataset with at least 2 items"

    generator = torch.Generator()
    generator.manual_seed(seed)

    num_val = int(num_data * val_ratio)
    idx = torch.randperm(num_data, generator=generator)
    val_idx = idx[:num_val]
    train_idx = idx[num_val:]
    assert len(train_idx) + len(val_idx) == num_data

    train_dataset = FluidDataset(
        source=dataset.source[train_idx], target=dataset.target[train_idx]
    )

    val_dataset = FluidDataset(
        source=dataset.source[val_idx], target=dataset.target[val_idx]
    )

    return train_dataset, val_dataset
