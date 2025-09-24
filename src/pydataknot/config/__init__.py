from dataclasses import dataclass, field
from typing import List, Any, Optional

from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir, JobConf
from omegaconf import MISSING

from flucoma_torch.config.model import MLPConfig
from flucoma_torch.config.scaler import ScalerConfig

regressor_defaults = ["_self_", {"mlp": "regressor"}, {"scaler": "normalize"}]
classifier_defaults = ["_self_", {"mlp": "classifier"}, {"scaler": "normalize"}]
feature_select_defaults = ["_self_", {"scaler": "normalize"}]


@dataclass
class DKClassifierConfig:
    defaults: List[Any] = field(default_factory=lambda: classifier_defaults)
    mlp: MLPConfig = MISSING
    scaler: Optional[ScalerConfig] = None

    data: str = MISSING
    features: str = ""  # "all" or "0-12" or [1, 2, ...]

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


@dataclass
class DKFeatureSelectConfig:
    defaults: List[Any] = field(default_factory=lambda: feature_select_defaults)
    data: str = MISSING
    scaler: Optional[ScalerConfig] = None
    num_features: int = 10
    plot: bool = False

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


cs = ConfigStore.instance()
cs.store(name="classifier_config", node=DKClassifierConfig)
cs.store(name="feature_select_config", node=DKFeatureSelectConfig)
