import argparse
from pathlib import Path

import yaml

from driftnet.data import nc_file_to_npys


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
    # nc_dir = Path(config['data']['nc_directory'])
    original_res_dir = Path(config["data"]["original_res_images"])

    # Run script
    fname = Path("/gws/pw/j07/wio/WIOInterannual/WINDS_SFC_201504.nc")
    nc_file_to_npys(fname, original_res_dir)


if __name__ == "__main__":
    main()
