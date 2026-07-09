import argparse

from driftnet.config import MasterConfig
from driftnet.plotting import plot_several_experiments
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
    # Edits
    print_and_save_config(config)

    # # Code to run
    # train_downscale(config.hyperparameters, config.data, config.experiment)

    # inference_over_test_set(config.data, config.hyperparameters, config.experiment)

    # evaluate_predictions(config.data, config.experiment)

    plot_several_experiments(
        config.experiment,
        exp_names=["default_experiment/baseline_trial", "pixelshuffle/baseline_trial"],
    )


if __name__ == "__main__":
    main()
