"""Early stopping class for training loops"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
import xarray as xr
from rich.console import Console
from rich.syntax import Syntax

from driftnet.config import DataConfig, MasterConfig


class EarlyStopping:
    """Class to handle early stopping and temporary checkpoint saves"""

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, models_dict, save_path):
        # Create save path if needed
        if not Path(save_path).exists():
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # If the loss improved
        if val_loss < self.best_loss - self.min_delta:
            print(
                f"  --> Validation loss decreased"
                f"({self.best_loss:.6f} --> {val_loss:.6f}). Saving models..."
            )
            self.best_loss = val_loss
            self.counter = 0

            # Extract the state_dict from every model in the dictionary
            checkpoint = {name: model.state_dict() for name, model in models_dict.items()}

            # Save the bundled dictionary
            torch.save(checkpoint, save_path)
        else:
            self.counter += 1
            print(f"  --> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def print_gpu_stats(batch_idx: int, num_batches: int):
    """Prints GPU memory and compute utilization."""
    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()

    # Compute Utilization (0-100%)
    utilization = torch.cuda.utilization(device)

    # Memory currently in use by tensors
    mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)

    # The max memory reached (useful to prevent OOM errors)
    max_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)

    # Memory reserved by PyTorch's caching allocator
    mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)

    print(
        f"[Batch {batch_idx:04d}/{num_batches}] "
        f"GPU Compute: {utilization:3d}% | "
        f"VRAM Used: {mem_alloc:.2f} GB (Max: {max_alloc:.2f} GB) | "
        f"VRAM Reserved: {mem_reserved:.2f} GB"
    )


def print_and_save_config(config: MasterConfig):
    console = Console(width=200)

    yaml_str = config.to_yaml_string()

    # 3. Print beautifully to the console/logs
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False, word_wrap=True)
    console.print("\n[bold cyan]=== Current Run Configuration ===[/bold cyan]")
    console.print(syntax)
    console.print("[bold cyan]===================================[/bold cyan]\n")

    # 4. Save the exact configuration to the experiment directory
    config.save_to_experiment_dir()
    console.print(
        f"[green] Configuration saved to {config.experiment.base_path}/run_config.yaml[/green]"
    )


def meters_to_degrees(u: float, v: float, lat: float) -> tuple[float, float]:
    """
    Converts velocities from meters per second to degrees per second.

    Args:
        u: Zonal velocity component (m/s)
        v: Meridional velocity component (m/s)
        lat: Latitude (degrees)

    Returns:
        Tuple[float, float]: (dlon/dt, dlat/dt) in degrees per second.
    """

    EARTH_RADIUS = 6371000.0

    lat_rad = np.radians(lat)
    # Prevent division by zero at the poles
    cos_lat = np.cos(lat_rad) if abs(lat) < 89.9 else np.cos(np.radians(89.9))

    dlon_dt = (u / (EARTH_RADIUS * cos_lat)) * (180.0 / np.pi)
    dlat_dt = (v / EARTH_RADIUS) * (180.0 / np.pi)
    return dlon_dt, dlat_dt


def get_spatial_trim_slices(nx: int, ny: int, n: int) -> tuple[slice, slice]:
    """
    Calculates standard Python slices to trim the West (x) and Bottom (y) boundaries
    so the grid dimensions are divisible by the degradation factor, n.

    Returns:
        tuple: (y_slice, x_slice)
    """
    x_remainder = nx % n
    y_remainder = ny % n

    # If remainder is > 0, start slicing from the remainder index. Otherwise, start from 0 (None).
    x_slice = slice(x_remainder if x_remainder != 0 else None, None)
    y_slice = slice(y_remainder if y_remainder != 0 else None, None)

    return y_slice, x_slice


def _get_valid_spatial_slices(data_config: DataConfig):
    ds_orig_lazy = xr.open_zarr(data_config.original_res)
    nx, ny = ds_orig_lazy.sizes["x"], ds_orig_lazy.sizes["y"]
    y_slice, x_slice = get_spatial_trim_slices(nx, ny, data_config.degrade_factor)
    return x_slice, y_slice


def extract_trajectories(ds_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extracts lists of 1D lon and lat arrays for each particle trajectory.
    Automatically filters out NaNs caused by out-of-bounds deletion.
    """
    # Since your data is backed by Dask, calling .values loads it into memory once,
    # making individual row access significantly faster than repeated indexing.

    ds = xr.open_zarr(ds_path)
    lons_all = ds["lon"].values
    lats_all = ds["lat"].values

    lons_list: list[np.ndarray] = []
    lats_list: list[np.ndarray] = []

    # Loop over the number of particles (trajectory dimension)
    for i in range(ds.sizes["trajectory"]):
        lon_p = lons_all[i, :]
        lat_p = lats_all[i, :]

        # Filter out NaNs from out-of-bounds deletions
        valid_mask = ~np.isnan(lon_p) & ~np.isnan(lat_p)

        # Only keep the track if it has at least some valid points
        if np.any(valid_mask):
            lons_list.append(lon_p[valid_mask])
            lats_list.append(lat_p[valid_mask])

    return lons_list, lats_list


def append_mean_row(agg_df: pl.DataFrame, time_col: str = "time") -> pl.DataFrame:
    """
    Calculates the mean of all numeric columns and appends it as a final row.
    Casts the time column to a string to insert an 'Average' label.
    """
    # 1. Calculate the mean for all columns EXCEPT the time column
    mean_row = agg_df.select(pl.exclude(time_col)).mean()

    # 2. Add the time column back to this single row with the label "Average"
    mean_row = mean_row.with_columns(pl.lit("Average").alias(time_col))

    # 3. Reorder the columns to match the original dataframe exactly
    mean_row = mean_row.select(agg_df.columns)

    # 4. Cast the original dataframe's time column to String so the types match
    # (Using pl.String for modern Polars, or pl.Utf8 if on an older version)
    agg_df_str = agg_df.with_columns(pl.col(time_col).cast(pl.String))

    # 5. Append the row to the bottom
    return pl.concat([agg_df_str, mean_row])
