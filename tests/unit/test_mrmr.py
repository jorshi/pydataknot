import torch

from pydataknot import mrmr


def make_dataset():
    # Feature 1 separates the classes, feature 2 is constant
    torch.manual_seed(0)
    y = torch.tensor([0] * 10 + [1] * 10)
    x = torch.randn(20, 4)
    x[:, 1] += y * 3
    x[:, 2] = 0.0
    return x, y


def test_constant_feature_not_nan():
    x, y = make_dataset()
    relevancy, redundancy = mrmr.relevancy_redundancy_clssif(x, y)

    assert not torch.any(torch.isnan(relevancy))
    assert not torch.any(torch.isnan(redundancy))
    assert relevancy[2] == 0.0


def test_constant_feature_selected_last():
    x, y = make_dataset()
    relevancy, redundancy = mrmr.relevancy_redundancy_clssif(x, y)
    selected = mrmr.select_features(4, relevancy, redundancy)

    assert selected[0] == 1
    assert selected[-1] == 2

    # Selection order of the other features matches selection without the
    # constant feature present
    keep = [0, 1, 3]
    relevancy, redundancy = mrmr.relevancy_redundancy_clssif(x[:, keep], y)
    expected = [keep[i] for i in mrmr.select_features(3, relevancy, redundancy)]
    assert selected[:-1] == expected
