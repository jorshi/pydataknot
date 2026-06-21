"""
Outlier detection applied to a training dataset
"""

from pathlib import Path

from flucoma_torch.data import (
    convert_fluid_dataset_to_tensor,
    convert_fluid_labelset_to_tensor,
)
import hydra
from hydra.utils import instantiate
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import torch

from pydataknot.config import DKOutlierDetectionConfig
from pydataknot.data import load_data
from pydataknot.utils import json_dump


class IsolationForestOutlierDetection:
    """
    Apply outlier detection using isolation forest

    Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008, December).
    Isolation forest.
    In 2008 8th IEEE International Conference on Data Mining (pp. 413-422). IEEE.
    """

    def __init__(self, random_state: int, num_estimators: int):
        super().__init__()
        self.random_state = random_state
        self.num_estimators = num_estimators

    def __call__(self, dataset: torch.Tensor):
        # Apply outlier detection
        dataset = dataset.numpy()
        outlier_detection = IsolationForest(
            random_state=self.random_state, n_estimators=self.num_estimators
        )
        outlier_scores = outlier_detection.fit_predict(dataset)
        outliers = np.where(outlier_scores == -1)[0]
        inliers = np.where(outlier_scores == 1)[0]
        return torch.from_numpy(outliers), torch.from_numpy(inliers)


def save_outlier_plot(data: np.ndarray, inliers: np.ndarray, outliers: np.ndarray):
    assert data.ndim == 2, "Data must be (num_points, dimensionality)"
    if data.shape[-1] > 2:
        data = PCA().fit_transform(data)[:, :2]
    elif data.shape[-1] < 2:
        raise ValueError("Can't plot 1D dataset")

    inliers = data[inliers]
    outliers = data[outliers]

    plt.scatter(inliers[:, 0], inliers[:, 1], c="mediumblue", label="inliers")
    plt.scatter(outliers[:, 0], outliers[:, 1], c="orangered", label="outliers")
    plt.legend()
    plt.title("Outliers")
    plt.tight_layout()
    plt.savefig("outliers.png", dpi=100)


@hydra.main(version_base=None, config_name="outlier_detection_config")
def main(cfg: DKOutlierDetectionConfig):
    dataset, labels, output = load_data(cfg)
    dataset = convert_fluid_dataset_to_tensor(dataset)
    labels, _ = convert_fluid_labelset_to_tensor(labels)
    labels = torch.argmax(labels, dim=-1)

    scaler = instantiate(cfg.scaler) if cfg.scaler else None
    if scaler is not None:
        logger.info(f"Scaling dataset with {str(scaler)}")
        scaler.fit(dataset)
        dataset = scaler.transform(dataset)

    outlier_detection = instantiate(cfg.outlier)
    outliers, inliers = outlier_detection(dataset)
    dataset = dataset.numpy()
    outliers = outliers.numpy()
    inliers = inliers.numpy()

    logger.info(f"Found {len(outliers)} outliers.")
    if len(outliers) > 0:
        logger.info(f"Outlier data indices: {outliers}")

    if cfg.plot:
        save_outlier_plot(dataset, inliers, outliers)

    # Add detected outliers to the incoming json file
    output["meta"]["info"]["outliers"] = 1
    output["outliers"] = [int(x) for x in outliers]

    output_name = Path(cfg.data).stem
    with open(f"{output_name}_outliers.json", "w") as f:
        f.write(json_dump(output, indent=4))


if __name__ == "__main__":
    main()
