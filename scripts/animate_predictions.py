"""
Animate truth, U-Net and diffusion predictions over consecutive test time steps, to
show temporal consistency (e.g. flicker in diffusion samples drawn independently at
each time step).

Writes three files to --output-dir:
    speed_animation.mp4 (or .gif if ffmpeg is unavailable) - side-by-side animation
    frame_strip.png - the first --strip-frames consecutive frames as a static grid
    flicker.png - RMS frame-to-frame change in speed for each field

Examples
--------
One day (48 half-hourly frames) in the most energetic 3 degree box:

    python scripts/animate_predictions.py --config configs/default.yml

A chosen box, starting 10 days into the test month:

    python scripts/animate_predictions.py --start-index 480 --corners 40 43 -25 -22
"""

import argparse
from pathlib import Path

from driftnet.config import MasterConfig
from driftnet.plotting import (
    animate_speed_series,
    load_speed_series,
    plot_flicker,
    plot_frame_strip,
)


def main():
    parser = argparse.ArgumentParser(description="Animate model predictions over time")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/users/sbarnett/documents/driftnet/configs/default.yml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--diffusion-experiment",
        type=str,
        default="diffusioncfg/ensemble_member_0",
        help="Experiment holding the diffusion predictions",
    )
    parser.add_argument(
        "--unet-experiment",
        type=str,
        default="resblock/l1",
        help="Experiment holding the U-Net predictions. Pass '' to leave the U-Net out",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="First test-set time step to show"
    )
    parser.add_argument("--frames", type=int, default=48, help="Number of frames")
    parser.add_argument("--stride", type=int, default=1, help="Time steps between frames")
    parser.add_argument(
        "--corners",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Box to show. Defaults to the most energetic --box-size box.",
    )
    parser.add_argument("--box-size", type=float, default=3.0, help="Auto box size (degrees)")
    parser.add_argument("--strip-frames", type=int, default=6, help="Columns in frame strip")
    parser.add_argument("--fps", type=int, default=6, help="Animation frames per second")
    parser.add_argument("--vmax", type=float, default=None, help="Top of the colour scale")
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Matplotlib or cmocean (cmo.*) colormap"
    )
    parser.add_argument("--output-dir", type=str, default="images/animation")
    args = parser.parse_args()

    if args.cmap.startswith("cmo."):
        import cmocean  # noqa: F401  (importing registers the cmo.* colormaps)

    config = MasterConfig.load_from_yaml(args.config)

    # The diffusion experiment comes first, so its times define the frames.
    experiments = {"Diffusion (one member)": args.diffusion_experiment}
    if args.unet_experiment:
        experiments["U-Net"] = args.unet_experiment

    lon, lat, times, speeds = load_speed_series(
        config.data,
        config.experiment,
        experiments,
        start_idx=args.start_index,
        n_frames=args.frames,
        stride=args.stride,
        corners=args.corners,
        box_size=args.box_size,
    )
    # Show truth, then U-Net, then diffusion.
    order = ["Ground truth", "U-Net", "Diffusion (one member)"]
    speeds = {label: speeds[label] for label in order if label in speeds}
    print(f"Loaded {len(times)} frames from {str(times[0])[:16]} to {str(times[-1])[:16]}")

    out = Path(args.output_dir)
    plot_frame_strip(
        lon,
        lat,
        times,
        speeds,
        n_columns=args.strip_frames,
        cmap=args.cmap,
        vmax=args.vmax,
        output_path=out / "frame_strip.png",
    )
    plot_flicker(times, speeds, output_path=out / "flicker.png")
    animation_path = animate_speed_series(
        lon,
        lat,
        times,
        speeds,
        cmap=args.cmap,
        vmax=args.vmax,
        fps=args.fps,
        output_path=out / "speed_animation.mp4",
    )
    print(f"Saved {animation_path}, {out / 'frame_strip.png'} and {out / 'flicker.png'}")


if __name__ == "__main__":
    main()
