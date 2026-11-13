from omegaconf import OmegaConf
import pytest
import torch

from pydataknot.data import FluidDataset, split_dataset_for_validation
from pydataknot.train_classifier import setup_data


NUM_ROWS = 40
NUM_COLS = 6


def make_fluid_data(num_rows: int, num_cols: int):
    # Small DataKnot-style dataset and labelset with two classes
    torch.manual_seed(0)
    x = torch.randn(num_rows, num_cols)
    source = {
        "cols": num_cols,
        "data": {str(i): x[i].tolist() for i in range(num_rows)},
    }
    target = {
        "cols": 1,
        "data": {str(i): ["a" if i % 2 == 0 else "b"] for i in range(num_rows)},
    }
    return x, source, target


def make_cfg(scaler=None, validation=0.25, stratify=False):
    return OmegaConf.create(
        {
            "scaler": scaler,
            "stratify": stratify,
            "mlp": {"validation": validation, "batch_size": 8},
        }
    )


NORMALIZE = {"_target_": "pydataknot.scaler.FluidNormalize"}


def test_scaler_fit_on_train_split_only():
    x, source, target = make_fluid_data(NUM_ROWS, NUM_COLS)
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
    x, source, target = make_fluid_data(NUM_ROWS, NUM_COLS)
    data = setup_data(source, target, make_cfg(NORMALIZE, validation=0.0))

    assert data["val_dataloader"] is None
    assert data["callbacks"] == []
    assert len(data["train_dataset"]) == NUM_ROWS

    # Without a split the scaler is fit on all the data
    assert torch.allclose(torch.tensor(data["scaler"]["data_min"]), x.min(dim=0).values)


@pytest.mark.parametrize("validation", [0.0, 0.25])
def test_no_scaler(validation):
    x, source, target = make_fluid_data(NUM_ROWS, NUM_COLS)
    data = setup_data(source, target, make_cfg(None, validation))

    assert data["scaler"] is None
    assert data["scaler_name"] == "none"

    # Data is passed through unscaled
    train = data["train_dataset"].source
    rows = {tuple(r) for r in x.tolist()}
    assert all(tuple(r) in rows for r in train.tolist())


def test_stratify_raises():
    _, source, target = make_fluid_data(3, 8)

    # Test to confirm that too few classes with data stratification on raises
    # a specific error.
    with pytest.raises(ValueError, match="The least populated class *"):
        setup_data(source, target, make_cfg(NORMALIZE, stratify=True))

    # Confirm that we can run this data if stratify is False.
    setup_data(source, target, make_cfg(NORMALIZE, stratify=False))


def test_stratified_split_preserves_class_balance():
    # Two balanced classes of 20 entries each -- a 0.25 validation split must put
    # exactly 5 of each class in the validation set. Labels are in two blocks
    # rather than alternating, which an unstratified split doesn't balance.
    _, source, target = make_fluid_data(NUM_ROWS, NUM_COLS)
    target["data"] = {
        str(i): ["a" if i < NUM_ROWS // 2 else "b"] for i in range(NUM_ROWS)
    }
    data = setup_data(source, target, make_cfg(NORMALIZE, stratify=True))

    train_classes = data["train_dataset"].target.argmax(dim=-1)
    val_classes = data["val_dataloader"].dataset.target.argmax(dim=-1)
    assert torch.bincount(val_classes).tolist() == [5, 5]
    assert torch.bincount(train_classes).tolist() == [15, 15]


def test_stratify_raises_too_few_validation_entries():
    # 8 classes with 3 entries each: a 0.2 validation split gives 5 validation
    # entries, which can't hold one entry of each class
    num_classes, per_class = 8, 3
    y = torch.eye(num_classes).repeat_interleave(per_class, dim=0)
    dataset = FluidDataset(torch.randn(len(y), 4), y)

    with pytest.raises(ValueError, match="A stratified split needs at least 8"):
        split_dataset_for_validation(dataset, 0.2, stratify=True)

    # A larger validation split leaves room for every class in both sets
    train, val = split_dataset_for_validation(dataset, 0.5, stratify=True)
    assert torch.bincount(val.target.argmax(dim=-1)).min() >= 1
    assert torch.bincount(train.target.argmax(dim=-1)).min() >= 1

    # Unstratified splits are still allowed
    split_dataset_for_validation(dataset, 0.2, stratify=False)
