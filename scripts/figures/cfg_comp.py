import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cartopy.mpl.geoaxes import GeoAxes

from driftnet.config import MasterConfig, DataConfig
from driftnet.ml.dataset import OceanDownscaleSet, _read_indices_from_csv
from driftnet.ml.diffusion import DDIMScheduler
from driftnet.ml.models import Strict5xDiffusionUNet
from driftnet.plotting import _get_extent, _get_valid_spatial_slices, _style_map_axis


def plot_3_panels_cfg(
    lr_u: np.ndarray,
    gen_u: np.ndarray,
    true_u: np.ndarray,
    data_config: DataConfig,
    cfg_scale: float,
    save_path: Path,
    corners: list[float] | None = None,
):
    """
    Plots a 3-panel Cartopy map comparing the low-res, diffusion generation, and high-res truth.
    """
    # 1. Load and slice grid coordinates to match the tensor shapes
    x_slice, y_slice = _get_valid_spatial_slices(data_config)
    grid = np.load(data_config.grid_params)
    lons = grid["rho_lon"][y_slice, x_slice]
    lats = grid["rho_lat"][y_slice, x_slice]

    # 2. Get map extent (with optional zoom)
    extent = _get_extent(lons.flatten(), lats.flatten(), corners=corners, padding=0.5)

    fig, axes_raw = plt.subplots(
        1, 3, figsize=(18, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = np.atleast_1d(axes_raw).flatten()

    # Create a symmetric color scale centered at 0
    abs_max = np.percentile(np.abs(true_u), 99)
    vmin, vmax = -abs_max, abs_max

    # Custom titles for the CFG breakdown
    panels = [
        ("Low-Res Blueprint (Interpolated)", lr_u),
        (f"Diffusion Generated (CFG = {cfg_scale})", gen_u),
        ("Ground Truth (High-Res)", true_u),
    ]

    mesh = None
    for i, (title, data) in enumerate(panels):
        ax = axes[i]
        _style_map_axis(ax, extent)
        ax.set_title(title, fontsize=14, pad=10)

        # Use pcolormesh with cartopy transform
        mesh = ax.pcolormesh(
            lons,
            lats,
            data,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

    # Add a shared colorbar
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.5, pad=0.08, aspect=40)
        cbar.set_label("Zonal Velocity ($U$)", fontsize=12)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'CFG {cfg_scale} plot saved to {save_path}')


@torch.inference_mode()
def generate_and_plot_cfg(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    data_config: DataConfig,
    x_lr: torch.Tensor,
    y_hr: torch.Tensor,
    cfg_scale: float,
    device: torch.device,
    out_dir: Path
):
    """
    Runs a single forward pass of the DDIM sampler for a specific CFG scale, then plots it.
    """
    print(f"\n--- Generating for CFG Scale: {cfg_scale} ---")

    VELOCITY_SCALE = 3.0
    x_scaled = x_lr.to(device, non_blocking=True) / VELOCITY_SCALE

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Generate high-res scaled image
        gen_scaled = scheduler.sample_fast(
            model, x_scaled, num_inference_steps=20,
            guidance_scale=cfg_scale, eta=0.0
        )

        # Upsample Low-Res condition to high-res so it maps correctly to the plotting grid
        lr_u_tensor = F.interpolate(
            x_scaled[:, 2:3, :, :], scale_factor=5.0, mode="bilinear", align_corners=True
        )

    # Convert back to physical velocity speeds
    gen_physical = gen_scaled * VELOCITY_SCALE
    lr_u_physical = lr_u_tensor * VELOCITY_SCALE

    # Extract numpy arrays for plotting (first item in the batch, U component)
    lr_u_np = lr_u_physical[0, 0].cpu().numpy()
    gen_u_np = gen_physical[0, 0].cpu().numpy()
    true_u_np = y_hr[0, 0].cpu().numpy()

    # Zoom into a specific region (you can edit these corners)
    zoom_corners = [40.0, 42.5, -20.0, -17.5]

    save_path = out_dir / f"cfg_comparison_scale_{cfg_scale}.png"

    plot_3_panels_cfg(
        lr_u=lr_u_np,
        gen_u=gen_u_np,
        true_u=true_u_np,
        data_config=data_config,
        cfg_scale=cfg_scale,
        save_path=save_path,
        corners=zoom_corners
    )


def wrapper_evaluate_cfgs(config: MasterConfig, cfg_scales: list[float]):
    """
    Main orchestrator: Loads model, extracts a sample, and loops over the CFG scales.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize and Load the Diffusion Model
    model = Strict5xDiffusionUNet(cond_channels=6, noise_channels=2, base_features=32)
    model = model.to(device)

    weights_path = config.experiment.model_weights
    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)

    # Strip compiled "_orig_mod." wrapper if it exists
    state_dict = checkpoint.get("unet", checkpoint)
    if list(state_dict.keys())[0].startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Optional: Compile model for speed if evaluating many CFG scales
    # model = torch.compile(model)

    scheduler = DDIMScheduler(num_train_timesteps=1000, device=str(device))

    # 2. Extract a Single Batch from the Test Set
    test_csv = config.data.splits / "test_indices.csv"
    test_indices = _read_indices_from_csv(test_csv)

    print("Loading test dataset sample...")
    # Initialize the dataset and grab the first batch
    dataset = OceanDownscaleSet(config.data, config.hyperparameters, time_indices=test_indices)
    x_batch, y_batch = dataset[0]

    # Isolate just a single image block (Batch size = 1) to keep memory overhead tiny
    x_single = x_batch[0:1]
    y_single = y_batch[0:1]

    # 3. Create output directory
    out_dir = Path("/home/users/sbarnett/documents/driftnet/images/cfg_comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Images will be saved to {out_dir}")

    # 4. Loop over requested CFG scales
    for scale in cfg_scales:
        generate_and_plot_cfg(
            model=model,
            scheduler=scheduler,
            data_config=config.data,
            x_lr=x_single,
            y_hr=y_single,
            cfg_scale=scale,
            device=device,
            out_dir=out_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CFG scales visually.")
    parser.add_argument(
        "--config", type=str,
        default="configs/default.yml",
        help="Path to the config file"
    )
    args = parser.parse_args()

    config = MasterConfig.load_from_yaml(args.config)

    # Define the list of Guidance Scales you want to evaluate
    scales_to_test = [2.0]

    wrapper_evaluate_cfgs(config, scales_to_test)