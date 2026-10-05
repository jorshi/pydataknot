from dataclasses import dataclass, field
from typing import List, Any, Optional

from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir, JobConf
from omegaconf import MISSING

from pydataknot.config.scaler import ScalerConfig


outlier_detection_defaults = [
    "_self_",
    {"scaler": "normalize"},
    {"outlier": "isolation_forest"},
]
feature_select_defaults = ["_self_", {"scaler": "normalize"}]
classifier_defaults = ["_self_", {"mlp": "dk_classifier"}, {"scaler": "normalize"}]
optimize_classifier_defaults = [
    "_self_",
    {"mlp": "dk_classifier"},
    {"scaler": "normalize"},
    {"outlier": "isolation_forest"},
]


@dataclass
class MLPConfig:
    _target_: str = MISSING


@dataclass
class DKMLPConfig(MLPConfig):
    _target_: str = "pydataknot.task.FluidMLPClassifier"
    input_size: int = MISSING
    output_size: int = MISSING
    activation: int = 3
    batch_size: int = 64
    hidden_layers: list[int] = field(default_factory=lambda: [89, 74, 59, 44, 29])
    learn_rate: float = 0.01
    max_iter: int = 10
    momentum: float = 0.1
    validation: float = 0.2
    optimizer: str = "adam"
    loss_fn: str = "bce"


@dataclass
class DKOutlierMethodConfig:
    _target_: str = MISSING


@dataclass
class DKIsolationForestOutlierDetection(DKOutlierMethodConfig):
    _target_: str = "pydataknot.outlier_detection.IsolationForestOutlierDetection"
    random_state: int = 42
    num_estimators: int = 1000


@dataclass
class DKOutlierDetectionConfig:
    defaults: List[Any] = field(default_factory=lambda: outlier_detection_defaults)
    data: str = MISSING
    outlier: DKOutlierMethodConfig = MISSING
    scaler: Optional[ScalerConfig] = None
    plot: bool = False

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


@dataclass
class DKBaseConfig:
    filter_outliers: bool = True


@dataclass
class DKFeatureSelectConfig(DKBaseConfig):
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


@dataclass
class DKClassifierConfig(DKBaseConfig):
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
class DKOptimizeClassifierConfig(DKBaseConfig):
    defaults: List[Any] = field(default_factory=lambda: optimize_classifier_defaults)
    mlp: MLPConfig = MISSING
    scaler: Optional[ScalerConfig] = None
    outlier: Optional[DKOutlierMethodConfig] = None

    data: str = MISSING
    features: str = ""  # "all" or "0-12" or [1, 2, ...]
    optimize_features: bool = True

    # Optuna specific config
    study_name: str = "classifier_study"
    sqlite: bool = True
    storage_name: str = "classifier_study"
    n_trials: int = 10
    n_startup_trials: int = 10  # Number trials before start checking to prune
    n_warmup_steps: int = 100  # Number warm-up steps.
    include_default: bool = True

    # Perform a deep model training after optimization
    deep_run: bool = True
    deep_run_max_iters: int = 1000

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


cs = ConfigStore.instance()
cs.store(
    group="outlier", name="isolation_forest", node=DKIsolationForestOutlierDetection
)
cs.store(group="mlp", name="dk_classifier", node=DKMLPConfig)
cs.store(name="outlier_detection_config", node=DKOutlierDetectionConfig)
cs.store(name="feature_select_config", node=DKFeatureSelectConfig)
cs.store(name="classifier_config", node=DKClassifierConfig)
cs.store(name="optimize_classifier_config", node=DKOptimizeClassifierConfig)
