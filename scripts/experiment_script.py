import argparse
from pathlib import Path
from collections.abc import Sequence

from generate_types import generate_experiment_types
from metrics.diagnostics import save_metrics, plot_metrics

from driftnet.config import MasterConfig
from driftnet.utils import print_and_save_config
from driftnet.metrics.lagrange import compute_trajectories
from driftnet.generated_types import ExperimentPathType


def main():
    parser = argparse.ArgumentParser(description="Run Driftnet")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/users/sbarnett/documents/driftnet/configs/default.yml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load the configuration (automatically creates experiment folders)
    config = MasterConfig.load_from_yaml(args.config)

    # Make any edits to the config and save
    print_and_save_config(config)

    # Update the type hinting for experiments
    generate_experiment_types()

    # Code to run
    exp_names : Sequence[ExperimentPathType]
    exp_names = ['default_experiment/baseline_trial',
                 'pixelshuffle/baseline_trial',
                 'batchnorm/baseline_trial',
                 'interpolate/normalise_trajectories']

    # compute_trajectories(config.data, config.experiment, exp_names)
    save_metrics(config.data, config.experiment, exp_names)
    plot_metrics(config.data, config.experiment, exp_names)


if __name__ == "__main__":
    main()
