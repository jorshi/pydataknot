"""
Maximum Relevancy Minimum Redundancy Feature Selection
"""

from loguru import logger
import numpy as np
from sklearn.feature_selection import f_classif
import torch


def relevancy_redundancy_clssif(x: torch.Tensor, y: torch.Tensor):
    """
    Get relevancy and redundancy data for classification data.
    """
    relevancy = torch.from_numpy(f_classif(x.numpy(), y.numpy())[0])
    redundancy = torch.corrcoef(x.T)

    # Constant features have NaN relevancy (0/0 F-statistic) and NaN correlations.
    # NaN wins argmax in select_features, so treat these features as having no
    # relevancy and being fully redundant -- they will be selected last.
    constant = torch.isnan(relevancy).nonzero().flatten().tolist()
    if len(constant) > 0:
        logger.warning(f"Found constant features: {constant}")
    relevancy = torch.nan_to_num(relevancy, nan=0.0)
    redundancy = torch.nan_to_num(redundancy, nan=1.0)

    assert relevancy.shape == (x.shape[-1],)
    assert redundancy.shape == (x.shape[-1], x.shape[-1])

    return relevancy, redundancy


def select_features(
    num_featues: int, relevancy: torch.Tensor, redundancy: torch.Tensor
):
    """
    Iteratively select features, maximizing the relevancy and minimizing redundancy.
    """
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
