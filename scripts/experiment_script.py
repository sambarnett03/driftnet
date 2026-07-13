import argparse

from generate_types import generate_experiment_types
from driftnet.config import MasterConfig
from driftnet.utils import print_and_save_config
from metrics.lagrange_diagnostics import evaluate_predictions, plot_metrics
from ml.training_script import train_downscale
from ml.inference_script import inference_over_test_set


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

    # evaluate_predictions(config.data, config.experiment)

    plot_metrics(config.data, config.experiment)

if __name__ == "__main__":
    main()
