"""
Plot a poster figure of the downscaling method: truth -> degraded -> U-Net -> output,
with the training loss loop, using real fields from one test-set time step.

Examples
--------
Auto-pick a 3 degree box around the most energetic currents:

    python scripts/plot_method_figure.py --config configs/default.yml

Choose the time step and zoom box yourself (lon_min lon_max lat_min lat_max):

    python scripts/plot_method_figure.py --time-index 100 --corners 40 43 -25 -22
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from driftnet.config import MasterConfig
from driftnet.plotting import _block_average_2d, plot_method_figure
from driftnet.utils import get_spatial_trim_slices


def _speed(ds: xr.Dataset) -> np.ndarray:
    """Return speed with land (u = v = 0 or NaN) set to NaN."""
    u = ds.velocity.isel(component=0).values.astype(float)
    v = ds.velocity.isel(component=1).values.astype(float)
    speed = np.hypot(u, v)
    speed[(u == 0) & (v == 0)] = np.nan
    return speed


def _most_energetic_box(
    lon: np.ndarray, lat: np.ndarray, speed: np.ndarray, size_deg: float
) -> tuple[float, float, float, float]:
    """Return the size_deg x size_deg box with the highest mean speed (land counts as 0)."""
    spacing = float(np.nanmedian(np.abs(np.diff(lon, axis=1))))
    window = max(int(round(size_deg / spacing)), 1)
    mean_speed = uniform_filter(np.nan_to_num(speed), size=window, mode="constant")

    # Keep the whole box inside the domain.
    half = window // 2
    mean_speed[:half], mean_speed[-half:] = 0, 0
    mean_speed[:, :half], mean_speed[:, -half:] = 0, 0

    j, i = np.unravel_index(np.argmax(mean_speed), mean_speed.shape)
    lon_c, lat_c = float(lon[j, i]), float(lat[j, i])
    return (
        lon_c - size_deg / 2,
        lon_c + size_deg / 2,
        lat_c - size_deg / 2,
        lat_c + size_deg / 2,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot the downscaling method figure")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/users/sbarnett/documents/driftnet/configs/default.yml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--time-index", type=int, default=0, help="Index into the saved test-set predictions"
    )
    parser.add_argument(
        "--corners",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Zoom box. Defaults to the most energetic --box-size box.",
    )
    parser.add_argument("--box-size", type=float, default=3.0, help="Auto box size (degrees)")
    parser.add_argument("--vmax", type=float, default=None, help="Top of the colour scale")
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Matplotlib or cmocean (cmo.*) colormap"
    )
    parser.add_argument("--output", type=str, default="images/method_figure.png")
    args = parser.parse_args()

    if args.cmap.startswith("cmo."):
        import cmocean  # noqa: F401  (importing registers the cmo.* colormaps)

    config = MasterConfig.load_from_yaml(args.config)
    data, experiment = config.data, config.experiment
    factor = data.degrade_factor

    predicted = xr.open_zarr(experiment.model_predictions).isel(time_counter=args.time_index)
    time = predicted.time_counter.values
    truth = xr.open_zarr(data.original_res).sel(time_counter=time)
    degraded = xr.open_zarr(data.degraded_res).sel(time_counter=time)

    # Truth is stored untrimmed; predictions and the degraded field match the trimmed grid.
    y_slice, x_slice = get_spatial_trim_slices(truth.sizes["x"], truth.sizes["y"], factor)
    truth = truth.isel(y=y_slice, x=x_slice)

    grid = np.load(data.grid_params)
    hr_lon = grid["rho_lon"][y_slice, x_slice]
    hr_lat = grid["rho_lat"][y_slice, x_slice]
    lr_lon = _block_average_2d(hr_lon, factor, factor)
    lr_lat = _block_average_2d(hr_lat, factor, factor)

    truth_speed = _speed(truth)
    degraded_speed = _speed(degraded)
    predicted_speed = _speed(predicted)
    # The model predicts over land too; blank it out with the truth's land mask.
    predicted_speed[np.isnan(truth_speed)] = np.nan

    corners = args.corners or _most_energetic_box(lr_lon, lr_lat, degraded_speed, args.box_size)
    print(f"Time: {str(time)[:16]}, zoom box (lon_min, lon_max, lat_min, lat_max): {corners}")

    plot_method_figure(
        hr_lon,
        hr_lat,
        lr_lon,
        lr_lat,
        truth_speed,
        degraded_speed,
        predicted_speed,
        corners=corners,
        factor=factor,
        cmap=args.cmap,
        vmax=args.vmax,
        output_path=args.output,
    )
    print(f"Saved {args.output} and {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
