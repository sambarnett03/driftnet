import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from driftnet.data import degrade_zarr_store, preprocess_folder
from driftnet.plotting import plot_velocity_quiver

# --- Suppress Zarr V3 experimental warnings ---
warnings.filterwarnings("ignore", message=".*FixedLengthUTF32.*")
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

    # Load the configuration file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    nc_dir = Path(config["data"]["nc_directory"])
    coord_data = np.load(config["data"]["grid_params"])

    # We now target .zarr datasets instead of empty directories
    original_res_zarr = Path(config["data"]["original_res_images"]) / "mock.zarr"
    degraded_zarr = Path(config["data"]["degraded_images"]) / "mock.zarr"

    # 1. Take all raw NetCDFs and pack them into a High-Res Zarr store
    preprocess_folder(nc_dir, original_res_zarr)

    # 2. Plot the high resolution map
    ds = xr.open_dataset("/gws/nopw/j04/oxford_es/sbarnett/driftnet/images/original_res/mock.zarr")
    ds_filtered = ds.isel(time_counter=0)
    vels = ds_filtered.velocity.values

    plot_velocity_quiver(
        coord_data=coord_data,
        u_input=vels[0],
        v_input=vels[1],
        stride=1,
        gridline_interval=0.02,
        corners=[35, 35.1, -20, -19.9],
        title="u and v velocity component",
        output_path="images/original_image.png",
    )

    # 3. Open the High-Res Zarr store, degrade it, and save it to a Low-Res Zarr store
    degrade_zarr_store(original_res_zarr, 2, degraded_zarr)

    # 4. Check degraded velocity maps
    ds = xr.open_dataset("/gws/nopw/j04/oxford_es/sbarnett/driftnet/images/degraded/mock_n2.zarr")
    ds_filtered = ds.isel(time_counter=0)
    vels = ds_filtered.velocity.values

    plot_velocity_quiver(
        coord_data=coord_data,
        u_input=vels[0],
        v_input=vels[1],
        stride=1,
        gridline_interval=0.02,
        corners=[35, 35.1, -20, -19.9],
        title="u and v velocity component",
        output_path="images/degraded_image.png",
    )


if __name__ == "__main__":
    main()
