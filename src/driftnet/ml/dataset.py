from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class OceanDownscaleSet(Dataset):
    def __init__(
        self,
        low_res_zarr_path: Path,
        high_res_zarr_path: Path,
        time_indices: range | list | None = None,
        batch_size: int = 32,
    ):
        low_vel_path = low_res_zarr_path / "velocity"
        high_vel_path = high_res_zarr_path / "velocity"

        self.X_low_res: zarr.Array = zarr.open_array(str(low_vel_path), mode="r")
        self.Y_high_res: zarr.Array = zarr.open_array(str(high_vel_path), mode="r")
        self.batch_size = batch_size

        total_time_steps = self.X_low_res.shape[0]
        if time_indices is None:
            self.time_indices = range(total_time_steps)
        else:
            self.time_indices = time_indices

    def __len__(self) -> int:
        # Length is now the number of BATCHES, not individual items
        return int(np.ceil(len(self.time_indices) / self.batch_size))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.time_indices))

        batch_indices = self.time_indices[start_idx:end_idx]

        # Use efficient Zarr slicing if indices are sequential
        if isinstance(self.time_indices, range) or list(batch_indices) == list(
            range(batch_indices[0], batch_indices[-1] + 1)
        ):
            x_np = self.X_low_res[batch_indices[0] : batch_indices[-1] + 1]
            y_np = self.Y_high_res[batch_indices[0] : batch_indices[-1] + 1]
        else:
            # Fallback to orthogonal indexing (if you ever shuffle)
            x_np = self.X_low_res[list(batch_indices)]
            y_np = self.Y_high_res[list(batch_indices)]

        return torch.from_numpy(x_np).float(), torch.from_numpy(y_np).float()


def get_train_val_test_datasets(low_res_path: Path, high_res_path: Path, batch_size: int = 32):
    temp_store = zarr.open_array(str(low_res_path / "velocity"), mode="r")
    total_steps = temp_store.shape[0]

    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.85)

    train_indices = range(0, train_end)
    val_indices = range(train_end, val_end)
    test_indices = range(val_end, total_steps)

    # Pass batch size down to the Dataset
    train_set = OceanDownscaleSet(
        low_res_path, high_res_path, time_indices=train_indices, batch_size=batch_size
    )
    val_set = OceanDownscaleSet(
        low_res_path, high_res_path, time_indices=val_indices, batch_size=batch_size
    )
    test_set = OceanDownscaleSet(
        low_res_path, high_res_path, time_indices=test_indices, batch_size=batch_size
    )

    return train_set, val_set, test_set
