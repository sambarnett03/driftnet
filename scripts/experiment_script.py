import argparse
import shutil
import os

from generate_types import generate_experiment_types
from metrics.lagrange_diagnostics import evaluate_predictions, save_metrics
from driftnet.plotting import plot_several_experiments

from driftnet.config import MasterConfig
from driftnet.utils import print_and_save_config


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
    # train_downscale(config.hyperparameters, config.data, config.experiment)

    # inference_over_test_set(config.data, config.hyperparameters, config.experiment)

    # os.makedirs(config.experiment.base, exist_ok=True)
    # shutil.copytree('/gws/ssde/j25b/oxford_es/sbarnett/driftnet/experiments/interpolate/baseline_trial/predictions.zarr',
    #                 config.experiment.model_predictions)

    print('predictions copied')
    evaluate_predictions(config.data, config.experiment)

if __name__ == "__main__":
    main()
