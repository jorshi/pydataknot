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


def save_feature_plots(
    relevancy: torch.Tensor, redundancy: torch.Tensor, prefix="", features=None
):
    prefix = f"{prefix}_" if prefix != "" else ""
    x = range(relevancy.shape[0]) if features is None else features

    # Save relevancy as a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(x, relevancy.numpy())
    plt.tight_layout()
    plt.savefig(f"{prefix}feature_relevancy.png", dpi=100)
    plt.close()

    # Save redundancy as heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(redundancy.numpy()), cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Correlation Coefficient")
    plt.title("Feature Redundancy (Correlation Matrix)")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    plt.savefig(f"{prefix}feature_redundancy.png", dpi=100)


def save_outlier_plot(data: torch.Tensor, outliers: np.ndarray):
    assert data.ndim == 2, "Data must be (num_points, dimensionality)"
    data = data.numpy()
    if data.shape[-1] > 2:
        data = PCA().fit_transform(data)[:, :2]
    elif data.shape[-1] < 2:
        raise ValueError("Can't plot 1D dataset")

    plt.scatter(data[:, 0], data[:, 1], c=outliers, cmap="viridis")
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
    outlier_detection = IsolationForest(
        random_state=cfg.seed, n_estimators=cfg.num_estimators
    )
    outliers = outlier_detection.fit_predict(dataset.numpy())
    print(np.where(outliers == -1))

    if cfg.plot:
        save_outlier_plot(dataset, outliers)


if __name__ == "__main__":
    main()
