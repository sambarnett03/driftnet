import argparse
import warnings
from pathlib import Path

from driftnet.ml.utils import print_and_save_config
import numpy as np
import yaml


from data.preprocess import preprocess_data
from ml.training_script import train_downscale
from driftnet.config import MasterConfig

# --- Suppress Zarr V3 experimental warnings ---
warnings.filterwarnings("ignore", message=".*FixedLengthUTF32.*")
warnings.filterwarnings("ignore", message=".*vlen-utf8*")
warnings.filterwarnings("ignore", message=".*Consolidated metadata is currently not part.*")


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

    # Code to run
    preprocess_data(
        config.data,
        plot_graphs=False
    )

    train_downscale(config.hyperparameters, config.data, config.experiment)

    print('run some code')


if __name__ == "__main__":
    main()
