import json
from typing import Any, Dict, List

from omegaconf import DictConfig


def json_dump(obj: Any, indent: int = 4) -> str:
    """Pretty-print JSON with objects indented and lists kept on a single line."""

    def _dump(x: Any, level: int) -> str:
        pad = " " * (indent * level)
        nxt = " " * (indent * (level + 1))

        if isinstance(x, dict):
            if not x:
                return "{}"
            parts = []
            for i, (k, v) in enumerate(x.items()):
                key = json.dumps(k)
                val = _dump(v, level + 1)
                parts.append(f"{nxt}{key}: {val}")
            return "{\n" + ",\n".join(parts) + "\n" + pad + "}"

        elif isinstance(x, list):
            # Keep lists inline
            if not x:
                return "[]"
            items = [_dump(v, level + 1) for v in x]
            return "[ " + ", ".join(items) + " ]"

        # Primitives (str, int, float, bool, None) get normal JSON encoding
        else:
            return json.dumps(x)

    return _dump(obj, 0)


def get_scaler_name(scaler_cfg: Any) -> str:
    if scaler_cfg is None:
        return "none"
    if not hasattr(scaler_cfg, "_target_"):
        raise ValueError("Scaler config must have a _target_ attribute")
    target = scaler_cfg._target_
    if not isinstance(target, str):
        raise ValueError("_target_ must be a string")
    if not target.startswith("flucoma_torch.scaler."):
        raise ValueError("_target_ must start with 'flucoma_torch.scaler.'")

    scaler_name = target.split(".")[-1].removeprefix("Fluid").lower()
    scaler_name = scaler_name.replace("scaler", "scale")
    if scaler_name not in ["normalize", "standardize", "robustscale"]:
        raise ValueError(f"Unknown scaler name: {scaler_name}")

    return scaler_name


def save_trained_model(
    output_path: str,
    cfg: DictConfig,
    model_dict: Dict,
    data: Dict,
    selected_features: List,
    output: Dict,
):
    # MLPClassifier needs labels corresponding to the onehot
    # prediction along with the model weights.
    labels_dict = {"labels": data["labels"], "rows": len(data["labels"])}
    classifier_dict = {
        "labels": labels_dict,
        "mlp": model_dict,
    }

    if "mlp_trained" not in output["meta"]["info"]:
        output["meta"]["info"]["mlp_trained"] = 0

    output["meta"]["info"]["python_trained"] = 1
    output["pythonclassifier"] = classifier_dict

    if data["scaler"]:
        scaler_name = get_scaler_name(cfg.scaler)
        output["meta"]["info"]["scaler"] = scaler_name
        output["input_scaler"] = data["scaler"]

    if len(selected_features) > 0:
        output["meta"]["info"]["feature_select"] = 1
        output["feature_select"] = selected_features

    with open(output_path, "w") as f:
        f.write(json_dump(output, indent=4))
