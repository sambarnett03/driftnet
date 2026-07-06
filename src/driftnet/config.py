import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    exp_name: str | Path = ""
    trial_name: str | Path = ""
    base: str = ""

    def __post_init__(self):
        # 1. Strictly enforce the base path
        if not self.base:
            raise ValueError(
                "CRITICAL: The 'base' path is missing in your YAML! "
                "You must specify a base directory to save experiment outputs."
            )

        # 2. Apply defaults if names are blank
        self.exp_name = Path(self.exp_name) or "default_experiment"
        self.trial_name = Path(self.trial_name) or "baseline_trial"

        # 3. Resolve the full path
        self.base_path = Path(self.base) / self.exp_name / self.trial_name

        # 4. Create the necessary directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.base_path / "model_data", exist_ok=True)
        os.makedirs(self.base_path / "model_data" / "temp", exist_ok=True)
        os.makedirs(self.base_path / "metrics", exist_ok=True)

        self.model_weights = self.base_path / "model_data" / "best_weights.pth"
        self.model_predictions = self.base_path / "predictions.zarr"
        self.metrics = self.base_path / "metrics"

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class OriginalResConfig:
    dir: str = ""
    zarr_name: str = "mock.zarr"

    def __post_init__(self):
        self.zarr_name = self.zarr_name or "mock.zarr"
        # Pre-compute the full Path object
        self.full_path = Path(self.dir) / self.zarr_name

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class DegradedResConfig:
    dir: str = ""
    factor: int = 5

    def __post_init__(self):
        # Auto-generate the zarr name based on the factor
        self.zarr_name = f"mock_n{self.factor}.zarr"
        # Pre-compute the full Path object
        self.full_path = Path(self.dir) / self.zarr_name

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class InterpolatedConfig:
    dir: str = ""
    zarr_name: str = "mock.zarr"

    def __post_init__(self):
        self.zarr_name = self.zarr_name or "mock.zarr"
        # Pre-compute the full Path object
        self.full_path = Path(self.dir) / self.zarr_name

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class DataConfig:
    # These contain our nested dataclasses
    # Give them a default of None so they don't crash if omitted from the YAML
    original_res: Path
    degraded_res: Path
    interpolated: Path

    nc_directory: str = ""
    grid_params: str = ""
    unet_data: str = ""

    def __post_init__(self):
        # Convert base strings to Paths
        self.nc_dir = Path(self.nc_directory)
        self.grid_params_path = Path(self.grid_params)
        self.unet_dir = Path(self.unet_data)

        # If these weren't provided in YAML, initialize them with defaults
        if self.original_res is None:
            self.original_res = Path()
        elif isinstance(self.original_res, dict):
            self.original_res = OriginalResConfig(**self.original_res).full_path

        if self.degraded_res is None:
            self.degraded_res = Path()
        elif isinstance(self.degraded_res, dict):
            self.degraded_dict = self.degraded_res
            self.degraded_res = DegradedResConfig(**self.degraded_dict).full_path
            self.degrade_factor = DegradedResConfig(**self.degraded_dict).factor

        if self.interpolated is None:
            self.interpolated = Path()
        elif isinstance(self.interpolated, dict):
            self.interpolated = InterpolatedConfig(**self.interpolated).full_path

        self.splits = self.original_res.parent.parent / "data_splits"

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class HyperparametersConfig:
    batch_size: int = 48
    micro_batch_size: int = 24
    learning_rate: float = 1e-4
    epochs: int = 100

    def __getitem__(self, key):
        return getattr(self, key)

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)


@dataclass
class RunConfig:
    random_seed: int = 42

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class MasterConfig:
    experiment: ExperimentConfig
    data: DataConfig
    hyperparameters: HyperparametersConfig
    run: RunConfig

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "MasterConfig":
        with open(yaml_path) as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls(
            experiment=ExperimentConfig(**raw_dict.get("experiment", {})),
            data=DataConfig(**raw_dict.get("data", {})),
            hyperparameters=HyperparametersConfig(**raw_dict.get("hyperparameters", {})),
            run=RunConfig(**raw_dict.get("run", {})),
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        """Converts the dataclasses to a dict, safely casting Path objects to strings."""
        from pathlib import Path

        raw_dict = asdict(self)

        def make_serializable(d):
            if isinstance(d, dict):
                return {k: make_serializable(v) for k, v in d.items()}
            elif isinstance(d, Path):
                return str(d)  # Convert paths to clean strings
            return d

        return make_serializable(raw_dict)

    def to_yaml_string(self) -> str:
        """Returns the config as a cleanly formatted YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save_to_experiment_dir(self):
        """Saves the current config state into the experiment folder."""
        # It automatically knows where to save because we defined base_path earlier!
        save_path = self.experiment.base_path / "run_config.yaml"
        with open(save_path, "w") as f:
            f.write(self.to_yaml_string())
