from omegaconf import OmegaConf
import pytest
import torch

from pydataknot.train_classifier import setup_data

NUM_ROWS = 40
NUM_COLS = 6


def make_fluid_data():
    # Small DataKnot-style dataset and labelset with two classes
    torch.manual_seed(0)
    x = torch.randn(NUM_ROWS, NUM_COLS)
    source = {
        "cols": NUM_COLS,
        "data": {str(i): x[i].tolist() for i in range(NUM_ROWS)},
    }
    target = {
        "cols": 1,
        "data": {str(i): ["a" if i % 2 == 0 else "b"] for i in range(NUM_ROWS)},
    }
    return x, source, target


def make_cfg(scaler=None, validation=0.25):
    return OmegaConf.create(
        {"scaler": scaler, "mlp": {"validation": validation, "batch_size": 8}}
    )


NORMALIZE = {"_target_": "pydataknot.scaler.FluidNormalize"}


def test_scaler_fit_on_train_split_only():
    x, source, target = make_fluid_data()
    data = setup_data(source, target, make_cfg(NORMALIZE))

    # Fit on the train split, the scaled train data spans exactly [0, 1]
    train = data["train_dataset"].source
    assert torch.all(train.min(dim=0).values == 0.0)
    assert torch.all(train.max(dim=0).values == 1.0)

    # The saved scaler must differ from one fit on all the data, otherwise
    # validation statistics have leaked into training
    data_min = torch.tensor(data["scaler"]["data_min"])
    data_max = torch.tensor(data["scaler"]["data_max"])
    leaked = torch.equal(data_min, x.min(dim=0).values) and torch.equal(
        data_max, x.max(dim=0).values
    )
    assert not leaked

    # Validation data is transformed with the train scaler, not refit
    val = data["val_dataloader"].dataset.source
    assert len(val) == NUM_ROWS - len(train)
    assert not torch.all(val.min(dim=0).values == 0.0)


def test_no_validation_split():
    x, source, target = make_fluid_data()
    data = setup_data(source, target, make_cfg(NORMALIZE, validation=0.0))

    assert data["val_dataloader"] is None
    assert data["callbacks"] == []
    assert len(data["train_dataset"]) == NUM_ROWS

    # Without a split the scaler is fit on all the data
    assert torch.allclose(torch.tensor(data["scaler"]["data_min"]), x.min(dim=0).values)


@pytest.mark.parametrize("validation", [0.0, 0.25])
def test_no_scaler(validation):
    x, source, target = make_fluid_data()
    data = setup_data(source, target, make_cfg(None, validation))

    assert data["scaler"] is None
    assert data["scaler_name"] == "none"

    # Data is passed through unscaled
    train = data["train_dataset"].source
    rows = {tuple(r) for r in x.tolist()}
    assert all(tuple(r) in rows for r in train.tolist())
