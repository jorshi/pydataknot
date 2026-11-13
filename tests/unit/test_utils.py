import json

from pydataknot.utils import save_trained_model


def save_and_load(tmp_path, output, selected_features):
    data = {"labels": ["a", "b"], "scaler": None}
    model_dict = {"layers": []}
    output_path = tmp_path.joinpath("model_pytrained.json")
    save_trained_model(output_path, None, model_dict, data, selected_features, output)
    with open(output_path) as f:
        return json.load(f)


def test_save_feature_selection(tmp_path):
    output = {"meta": {"info": {}}}
    saved = save_and_load(tmp_path, output, [0, 5, 17])

    assert saved["meta"]["info"]["feature_select"] == 1
    assert saved["feature_select"] == [0, 5, 17]


def test_save_clears_stale_feature_selection(tmp_path):
    # Input file has a prior feature selection, but the model was trained on all
    # features (e.g. features=all) -- the saved file must not keep the old selection
    output = {"meta": {"info": {"feature_select": 1}}, "feature_select": [0, 5, 17]}
    saved = save_and_load(tmp_path, output, [])

    assert saved["meta"]["info"]["feature_select"] == 0
    assert "feature_select" not in saved


def test_save_without_feature_selection(tmp_path):
    # No selection in the input or the model -- nothing is added
    output = {"meta": {"info": {}}}
    saved = save_and_load(tmp_path, output, [])

    assert "feature_select" not in saved["meta"]["info"]
    assert "feature_select" not in saved
