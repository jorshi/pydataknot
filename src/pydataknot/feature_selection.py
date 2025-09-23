"""
Suggest features using Maximum Relevancy Minimum Redundancy
"""

from flucoma_torch.data import (
    convert_fluid_dataset_to_tensor,
    convert_fluid_labelset_to_tensor,
)
import hydra


from pydataknot.config import DKFeatureSelectConfig
from pydataknot.data import load_data


@hydra.main(version_base=None, config_name="feature_select_config")
def main(cfg: DKFeatureSelectConfig):
    dataset, labels, output = load_data(cfg)
    dataset = convert_fluid_dataset_to_tensor(dataset)
    labels = convert_fluid_labelset_to_tensor(labels)


if __name__ == "__main__":
    main()
