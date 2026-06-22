"""Early stopping class for training loops"""

from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.syntax import Syntax

from driftnet.config import MasterConfig


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
