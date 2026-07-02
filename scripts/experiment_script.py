import argparse
import warnings

from ml.diagnostics import run_lagrangian_diagnostics

from driftnet.config import MasterConfig
from driftnet.utils import print_and_save_config

# --- Suppress Zarr V3 experimental warnings ---
warnings.filterwarnings("ignore", message=".*<U1*")
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
    # preprocess_data(config.data, plot_graphs=False)

    # train_downscale(config.hyperparameters, config.data, config.experiment)

    # inference_over_test_set(config.data, config.experiment)

    run_lagrangian_diagnostics(config.data, config.experiment)


if __name__ == "__main__":
    main()
