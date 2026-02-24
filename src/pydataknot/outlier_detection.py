"""
Outlier detection applied to a training dataset
"""

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

from pydataknot.config import DKFeatureSelectConfig
from pydataknot.data import load_data


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
def main(cfg: DKFeatureSelectConfig):
    dataset, labels, output = load_data(cfg)
    dataset = convert_fluid_dataset_to_tensor(dataset)
    labels, _ = convert_fluid_labelset_to_tensor(labels)
    labels = torch.argmax(labels, dim=-1)

    scaler = instantiate(cfg.scaler) if cfg.scaler else None
    if scaler is not None:
        logger.info(f"Scaling dataset with {str(scaler)}")
        scaler.fit(dataset)
        dataset = scaler.transform(dataset)

    # Apply outlier detection
    dataset = dataset.numpy()
    outlier_detection = IsolationForest(
        random_state=cfg.seed, n_estimators=cfg.num_estimators
    )
    outlier_scores = outlier_detection.fit_predict(dataset)
    outliers = np.where(outlier_scores == -1)
    inliers = np.where(outlier_scores == 1)

    if cfg.plot:
        save_outlier_plot(dataset, inliers, outliers)


if __name__ == "__main__":
    main()
