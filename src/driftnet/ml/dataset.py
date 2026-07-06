import csv
import random
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from driftnet.config import DataConfig, HyperparametersConfig
from driftnet.utils import get_spatial_trim_slices


class OceanDownscaleSet(Dataset):
    def __init__(
        self,
        data_config: DataConfig,
        hyper_config: HyperparametersConfig,
        time_indices: range | list | None = None,
    ):

        low_vel_path = data_config.degraded_res / "velocity"
        high_vel_path = data_config.original_res / "velocity"

        self.X_low_res: zarr.Array = zarr.open_array(str(low_vel_path), mode="r")
        self.Y_high_res: zarr.Array = zarr.open_array(str(high_vel_path), mode="r")
        self.batch_size = hyper_config.batch_size

        total_time_steps = self.X_low_res.shape[0]
        self.time_indices = time_indices if time_indices is not None else range(total_time_steps)

        # --- NEW: Calculate target trim slices ---
        # Assuming shape is (time, component, y, x)
        ny, nx = self.Y_high_res.shape[2], self.Y_high_res.shape[3]
        self.y_slice, self.x_slice = get_spatial_trim_slices(nx, ny, data_config.degrade_factor)
        # -----------------------------------------

    def __len__(self) -> int:
        return int(np.ceil(len(self.time_indices) / self.batch_size))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.time_indices))

        batch_indices = self.time_indices[start_idx:end_idx]

        # Use efficient Zarr slicing if indices are sequential
        if isinstance(self.time_indices, range) or list(batch_indices) == list(
            range(batch_indices[0], batch_indices[-1] + 1)
        ):
            # X is already degraded/trimmed on disk, so load normally
            x_np = self.X_low_res[batch_indices[0] : batch_indices[-1] + 1]

            # Y is high-res and untrimmed on disk, so we slice spatially directly from Zarr
            y_np = self.Y_high_res[
                batch_indices[0] : batch_indices[-1] + 1, :, self.y_slice, self.x_slice
            ]
        else:
            # Fallback to orthogonal indexing (if you ever shuffle)
            x_np = self.X_low_res[list(batch_indices)]

            # Fetch the whole spatial domain for these times into memory, then slice.
            y_np_untrimmed = self.Y_high_res[list(batch_indices)]
            y_np = y_np_untrimmed[:, :, self.y_slice, self.x_slice]  # type: ignore

        return torch.from_numpy(x_np).float(), torch.from_numpy(y_np).float()


def _write_indices_to_csv(csv_path: Path, indices: list[int]) -> None:
    """Helper to write a list of integer indices to a single-column CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_index"])
        for idx in indices:
            writer.writerow([idx])


def _read_indices_from_csv(csv_path: Path) -> list[int]:
    """Helper to read a single-column CSV of integer indices."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [int(row[0]) for row in reader]


def _generate_block_splits(
    total_steps: int,
    splits_dir: Path,
    train_keep_fraction: float,
    random_seed: int,
    steps_per_day: int = 48,
) -> None:
    """
    Calculates train, validation, and test indices using block sampling
    and saves them directly to CSV files.
    """
    test_steps = 30 * steps_per_day  # 1 consecutive month for Lagrangian testing
    val_steps = 14 * steps_per_day  # 2 consecutive weeks for validation
    block_steps = 4 * steps_per_day  # 4-day blocks for training (multiple of batch size)

    # 1. Allocate the test and validation ranges from the end of the dataset
    test_indices = list(range(total_steps - test_steps, total_steps))
    val_indices = list(range(total_steps - test_steps - val_steps, total_steps - test_steps))

    # 2. Slice the remaining training pool into discrete blocks
    train_pool_limit = total_steps - test_steps - val_steps
    num_available_blocks = train_pool_limit // block_steps
    block_starts = [i * block_steps for i in range(num_available_blocks)]

    # 3. Randomly sample a fraction of the blocks using the configured seed
    random.seed(random_seed)
    num_to_sample = int(num_available_blocks * train_keep_fraction)
    sampled_starts = random.sample(block_starts, num_to_sample)

    # CRITICAL: Sort chronologically so indices inside blocks stay sequential for Zarr
    sampled_starts.sort()

    # 4. Unroll the selected block start positions back into individual frame indices
    train_indices = []
    for start in sampled_starts:
        train_indices.extend(range(start, start + block_steps))

    # 5. Save all index lists to CSV files
    _write_indices_to_csv(splits_dir / "train_indices.csv", train_indices)
    _write_indices_to_csv(splits_dir / "val_indices.csv", val_indices)
    _write_indices_to_csv(splits_dir / "test_indices.csv", test_indices)


def get_train_val_test_datasets(
    data_config: DataConfig, hyper_config: HyperparametersConfig, train_keep_fraction: float = 0.5
):
    """
    Main orchestrator: Retrieves block-sampled dataset splits using project dataclasses.
    Generates the split CSVs if they don't already exist, then instantiates the PyTorch Datasets.
    """
    # Extract structural paths out of the nested dataclasses
    low_res_path = data_config.degraded_res
    batch_size = hyper_config.batch_size
    seed = 42

    # Define directory for tracking dataset splits next to the Zarr file
    splits_dir = data_config.splits
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train_indices.csv"
    val_csv = splits_dir / "val_indices.csv"
    test_csv = splits_dir / "test_indices.csv"

    # 1. If the CSV trackers don't exist, generate them once
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        print(f"Creating new block-sampled splits in {splits_dir}...")
        temp_store = zarr.open_array(str(low_res_path / "velocity"), mode="r")
        total_steps = temp_store.shape[0]

        _generate_block_splits(total_steps, splits_dir, train_keep_fraction, random_seed=seed)

    # 2. Load the splits from the CSV files
    print("Loading data splits from CSV trackers...")
    train_indices = _read_indices_from_csv(train_csv)
    val_indices = _read_indices_from_csv(val_csv)
    test_indices = _read_indices_from_csv(test_csv)

    print(
        f"Split Summary -> Train batches: {int(np.ceil(len(train_indices) / batch_size))}, "
        f"Val batches: {int(np.ceil(len(val_indices) / batch_size))}, "
        f"Test batches: {int(np.ceil(len(test_indices) / batch_size))}"
    )

    # 3. Instantiate and return the Datasets
    train_set = OceanDownscaleSet(data_config, hyper_config, time_indices=train_indices)
    val_set = OceanDownscaleSet(data_config, hyper_config, time_indices=val_indices)
    test_set = OceanDownscaleSet(data_config, hyper_config, time_indices=test_indices)

    return train_set, val_set, test_set
