"""
Suggest features using Maximum Relevancy Minimum Redundancy
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
import torch

from pydataknot.config import DKFeatureSelectConfig
from pydataknot.data import load_data
import pydataknot.mrmr as mrmr
from pydataknot.utils import json_dump


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


@hydra.main(version_base=None, config_name="feature_select_config")
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

    # Apply mRMR feature selection
    relevancy, redundancy = mrmr.relevancy_redundancy_clssif(dataset, labels)
    selected_features = mrmr.select_features(cfg.num_features, relevancy, redundancy)
    logger.info(f"Selected features: {selected_features}")

    selected_features = sorted(selected_features)
    if cfg.plot:
        save_feature_plots(relevancy, redundancy, prefix="pre")
        redundancy = redundancy[selected_features, :]
        redundancy = redundancy[:, selected_features]
        save_feature_plots(
            relevancy[selected_features],
            redundancy,
            prefix="post",
            features=selected_features,
        )

    # Add selected features to the incoming json file
    output["meta"]["info"]["feature_select"] = 1
    output["feature_select"] = selected_features

    output_name = Path(cfg.data).stem
    with open(f"{output_name}_feature_select.json", "w") as f:
        f.write(json_dump(output, indent=4))


if __name__ == "__main__":
    main()
