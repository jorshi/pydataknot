"""
Suggest features using Maximum Relevancy Minimum Redundancy
"""

from pathlib import Path

from flucoma_torch.data import (
    convert_fluid_dataset_to_tensor,
    convert_fluid_labelset_to_tensor,
)
from flucoma_torch.scaler import FluidNormalize
import hydra
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_classif
import torch

from pydataknot.config import DKFeatureSelectConfig
from pydataknot.data import load_data
from pydataknot.utils import json_dump

FLOOR = 0.001

# TODO: check whether Rodrigo zero indexes feature selection


def select_features_mrmr(
    num_featues: int, relevancy: torch.Tensor, redundancy: torch.Tensor
):
    features = np.arange(relevancy.shape[0])
    num_featues = min(num_featues, len(features))
    selected_features = []
    not_selected_features = features.copy()
    scores = []

    relevancy = relevancy.numpy()
    redundancy = redundancy.numpy()

    for i in range(num_featues):
        score_numerator = relevancy[not_selected_features]

        if i > 0:
            # The denominator is the average redundancy (correlation) of the not
            # selected features and the selected features
            score_denominator = redundancy[not_selected_features, :]
            score_denominator = score_denominator[:, selected_features]
            score_denominator = np.mean(
                np.abs(score_denominator), axis=-1, keepdims=False
            )
        else:
            score_denominator = np.ones_like(score_numerator)

        score = score_numerator / score_denominator

        best_feature = int(np.argmax(score))
        scores.append(score[best_feature])

        best_feature = int(not_selected_features[best_feature])
        selected_features.append(best_feature)
        not_selected_features = [x for x in not_selected_features if x != best_feature]

    return selected_features


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
    labels, classes = convert_fluid_labelset_to_tensor(labels)
    labels = torch.argmax(labels, dim=-1)

    # Normalize data
    normalizer = FluidNormalize()
    normalizer.fit(dataset)
    dataset = normalizer.transform(dataset)

    relevancy = torch.from_numpy(f_classif(dataset.numpy(), labels.numpy())[0])
    redundancy = torch.corrcoef(dataset)

    selected_features = select_features_mrmr(cfg.num_features, relevancy, redundancy)
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

    output["meta"]["info"]["feature_select"] = 1
    output["feature_select"] = selected_features

    output_name = Path(cfg.data).stem
    with open(f"{output_name}_feature_select.json", "w") as f:
        f.write(json_dump(output, indent=4))


if __name__ == "__main__":
    main()
