"""
Compare ground truth, interpolation, U-Net and diffusion velocity fields in a 2x2 grid.

Each experiment name points at ``<experiment base>/<name>/predictions.zarr``.

Examples
--------
Auto-pick a 3 degree box around the most energetic currents:

    python scripts/plot_velocity_comparison.py --diffusion-experiment diffusion/baseline_trial

Choose the time step and box yourself (lon_min lon_max lat_min lat_max):

    python scripts/plot_velocity_comparison.py --diffusion-experiment diffusion/baseline_trial
        --time-index 100 --corners 40 43 -25 -22
"""

import argparse

from driftnet.config import MasterConfig
from driftnet.plotting import plot_velocity_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare velocity fields in a 2x2 grid")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/users/sbarnett/documents/driftnet/configs/default.yml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--interpolation-experiment",
        type=str,
        default="interpolate/baseline_trial",
        help="Experiment holding the interpolation baseline",
    )
    parser.add_argument(
        "--unet-experiment",
        type=str,
        default="default_experiment/baseline_trial",
        help="Experiment holding the U-Net predictions",
    )
    parser.add_argument(
        "--diffusion-experiment",
        type=str,
        required=True,
        help="Experiment holding the diffusion predictions, e.g. diffusion/baseline_trial",
    )
    parser.add_argument(
        "--diffusion-residual-only",
        action="store_true",
        help="Set if the diffusion predictions are a residual to add to the U-Net output",
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
        help="Box to plot. Defaults to the most energetic --box-size box.",
    )
    parser.add_argument("--box-size", type=float, default=3.0, help="Auto box size (degrees)")
    parser.add_argument("--vmax", type=float, default=None, help="Top of the colour scale")
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Matplotlib or cmocean (cmo.*) colormap"
    )
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.cmap.startswith("cmo."):
        import cmocean  # noqa: F401  (importing registers the cmo.* colormaps)

    config = MasterConfig.load_from_yaml(args.config)

    plot_velocity_comparison(
        config.data,
        config.experiment,
        interpolation_exp=args.interpolation_experiment,
        unet_exp=args.unet_experiment,
        diffusion_exp=args.diffusion_experiment,
        time_idx=args.time_index,
        corners=args.corners,
        box_size=args.box_size,
        diffusion_residual_only=args.diffusion_residual_only,
        cmap=args.cmap,
        vmax=args.vmax,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
