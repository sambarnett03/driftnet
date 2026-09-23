"""
Plot surface current speed over the whole West Indian Ocean model domain.

Examples
--------
Single snapshot (first time step of the first file):

    python scripts/plot_surface_speed.py

Snapshot nearest a given date:

    python scripts/plot_surface_speed.py --time 2010-03-15

Time-mean speed over every file in the data directory:

    python scripts/plot_surface_speed.py --mean
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from driftnet.plotting import plot_speed_map, surface_speed, velocity_to_t_points


def main():
    parser = argparse.ArgumentParser(description="Plot WIO surface current speed")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/users/sbarnett/documents/driftnet/configs/default.yml",
        help="Path to the config file",
    )
    parser.add_argument("--time", type=str, default=None, help="Date/time of the snapshot")
    parser.add_argument("--mean", action="store_true", help="Plot the time-mean speed")
    parser.add_argument("--vmax", type=float, default=None, help="Top of the colour scale")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--no-inset", action="store_true", help="Omit the locator globe")
    parser.add_argument("--output", type=str, default="images/wio_surface_speed.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    nc_files = sorted(Path(config["data"]["nc_directory"]).glob("WINDS*.nc"))
    coord_data = dict(np.load(config["data"]["grid_params"]))

    if args.mean:
        ds = xr.open_mfdataset(nc_files, chunks={"time_counter": 24})
        # Mean of speed (not speed of the mean) so eddy energy is kept.
        lon, lat, u_t, v_t = velocity_to_t_points(coord_data, ds.u_surf.data, ds.v_surf.data)
        speed_all = np.hypot(u_t, v_t)
        speed = speed_all.mean(axis=0).compute()
        speed[speed == 0] = np.nan
        title = "Mean surface current speed"
    else:
        ds = xr.open_mfdataset(nc_files) if args.time else xr.open_dataset(nc_files[0])
        snap = (
            ds.sel(time_counter=args.time, method="nearest")
            if args.time
            else ds.isel(time_counter=0)
        )
        lon, lat, u_t, v_t = velocity_to_t_points(
            coord_data, snap.u_surf.values, snap.v_surf.values
        )
        speed = surface_speed(u_t, v_t)
        title = "Surface current speed"

    plot_speed_map(
        lon,
        lat,
        speed,
        title=title,
        cmap=args.cmap,
        vmax=args.vmax,
        inset=not args.no_inset,
        output_path=args.output,
    )
    print(f"Saved {args.output} and {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
