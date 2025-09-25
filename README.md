# pydataknot

Audio feature selection, model training, and hyperparameter optimization for Max/MSP
DataKnot classifiers.

## Installation

Requires Python >= 3.9

Recommend creating a virtual python environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install via pip:
```bash
pip install --upgrade pip
pip install pydataknot
```

If you're performing hyperparameter optimization and want to use the 
[Optuna Dashboard](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html)
to inspect results:
```bash
pip install "pydataknot[dashboard]"
```

## Usage

`pydataknot` includes a set of CLI tools that take as input json files produced from `dk.classcreate` in Max/MSP. To start, record a dataset using `dk.classcreate` and write to a json file.

### Feature Selection

Select a subset of audio features for classification based on Minimum Redundancy Maximum Relevance (mRMR). The goal is to select features that correlate highly with classes (maximum relevancy) and have low correlation with other features (minimum redundancy).

This command will select 12 features from the dataset in `dataset.json`:
```bash
pydk-select data=dataset.json num_features=12
```

This will copy the `dataset.json` file, add the selected feature indices, and save this copy in an output directory `outputs/feature_selection/{current-date}/{current-time}/`.

### Model Training

Train MLP classifiers using PyTorch. 

This command will train a MLP on the dataset in `dataset.json`, the MLP will have two hidden layers each with 16 neurons (note the quotes around the layers list) and will be trained for 500 epochs:
```bash
pydk-train data=dataset.json mlp.hidden_layers="[16,16]" mlp.max_iters=500
```

Note: If you use the json that output from the feature selection step, training will use that subset of selected features.

Similar to feature selection, this will copy the input json file and save a copy in an output directory `outputs/feature_selection/{current-date}/{current-time}/` with the trained model. `dk.classmatch` can load this json file and will automatically know to use the model trained here.

Run `pydk-train --help` to see a list of all possible arguments or check out [`flucoma-torch`](https://github.com/jorshi/flucoma-torch?tab=readme-ov-file#arguments) for more detailed information on arguments.

### Hyperparameter Search

Perform a search over MLP parameters to find an architecture and training parameters optimized to your dataset. This uses [`optuna`](https://optuna.readthedocs.io/en/stable/) under the hood.

This command will perform a hyperparameter search consisting of 1000 trials (1000 different MLPs will be trained) where each MLP will be trained for 100 epochs.
```bash
pydk-optimize data=dataset.json mlp.max_iter=100 n_trials=1000
```

This could take several hours to complete.

The best resulting model and associated hyperparameters will be saved in the output directory `outputs/feature_selection/{current-date}/{current-time}/`. `dk.classmatch` can load the model file and will automatically know to use the model trained here.

See [`flucoma-torch`](https://github.com/jorshi/flucoma-torch?tab=readme-ov-file#arguments-1) for a 
more detailed listing of possible arguments.

#### Feature Optimization
The number of input features can also be included for optimization by setting `optimize_features=true`. This will perform mRMR feature selection with different values for `num_features` to test which leads to the best performing model.

```bash
pydk-optimize data=dataset.json mlp.max_iter=100 n_trials=1000 optimize_features=true
```

#### Optuna Dashboard
To view all results in a web dashboard you can use the optuna dashboard. This is an optional dependency, to install it:
```bash
pip install "pydataknot[dashboard]"
```

During an optimizing study the results are stored in a database file (sqlite3) in the output directory corresponding to the study. Whether or not a sqlite3 file is saved depends on the argument `sqlite`, 
which defaults to true.

For example:
```bash
optuna-dashboard sqlite:///outputs/optimize_classifier/2025-09-25/19-58-57/classifier_study.sqlite3
```
