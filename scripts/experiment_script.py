import argparse
from pathlib import Path

import yaml
from data.preprocess import degrade_folder


def main():
    parser = argparse.ArgumentParser(description="Run Driftnet")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/users/sbarnett/documents/driftnet/configs/default.yml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Extract data dirs
    # nc_dir = Path(config["data"]["nc_directory"])
    original_res_dir = Path(config["data"]["original_res_images"])
    degraded_dir = Path(config["data"]["degraded_images"])

    # preprocess_folder(nc_dir, original_res_dir)

    # Degrade image
    degrade_folder(original_res_dir / "mock", 2, degraded_dir / "mock")


if __name__ == "__main__":
    main()
