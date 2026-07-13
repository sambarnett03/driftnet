from pathlib import Path

import dask.array as da
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from numpy.typing import NDArray

from driftnet.config import DataConfig, HyperparametersConfig, ExperimentConfig
from driftnet.ml.dataset import _read_indices_from_csv
from driftnet.utils import get_spatial_trim_slices

# ==========================================
# General utility functions
# ==========================================


def degrade_coords(arr: NDArray[np.floating], res: int, component: str) -> NDArray[np.floating]:
    if component not in ["u", "v"]:
        raise ValueError(f'component must be either "u" or "v", received {component}')

    if component == "u":
        if arr.shape[1] % res != 0:
            raise ValueError(
                f"The coordinates are not divisible by {res} - pad the coordinates first"
            )

        arr = arr.reshape(-1, arr.shape[1] // res, res)
        return np.average(arr, res)

    else:
        if arr.shape[0] % res != 0:
            raise ValueError(
                f"The coordinates are not divisible by {res} - pad the coordinates first"
            )

        arr = arr.reshape(arr.shape[0] // res, -1, res)
        return np.average(arr, res)


# ==========================================
# Original .npy & Zarr Loading Functions
# ==========================================


def _extract_u_and_v(ds: xr.Dataset) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if "u_surf" not in ds or "v_surf" not in ds:
        raise KeyError("The variables u_surf or v_surf were not found")
    u = ds.u_surf.values
    v = ds.v_surf.values
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u or v does not have dimensions 2. Ensure time axis is removed")
    return u, v


def _create_npy_file_name(time: np.datetime64) -> str:
    date_time = str(time).replace(":", "-")
    return f"velocities_{date_time}.npy"


def nc_file_to_npys(fname: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(fname) as ds:
        num_times = ds.sizes["time_counter"]
        for i in range(num_times):
            ds_filtered = ds.isel(time_counter=i)
            t = ds_filtered.time_counter.values
            save_fname = _create_npy_file_name(t)
            if (save_dir / save_fname).exists():
                continue
            u, v = _extract_u_and_v(ds_filtered)
            np.save(save_dir / save_fname, np.stack((u, v)))


def _format_dataset_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """
    Extracts U and V, bypasses coordinate broadcasting, and stacks them into a
    single DataArray, preserving the time_counter dimension for chunking.
    """
    if "u_surf" not in ds or "v_surf" not in ds:
        raise KeyError("The variables u_surf or v_surf were not found")

    # Extract the underlying lazy Dask arrays (bypassing Xarray's coordinate alignment)
    # Shapes are currently: (time_counter, y_grid, x_grid)
    u_lazy = ds.u_surf.data
    v_lazy = ds.v_surf.data

    # Stack them along a new component axis (axis=1)
    # Shape becomes: (time_counter, 2, y, x)
    vel_lazy = da.stack([u_lazy, v_lazy], axis=1)

    # Wrap back into a clean Xarray dataset with unified dimension names
    vel_field = xr.DataArray(
        data=vel_lazy,
        dims=["time_counter", "component", "y", "x"],
        coords={"time_counter": ds.time_counter, "component": [0, 1]},
        name="velocity",
    )

    return vel_field.to_dataset()


def nc_file_to_zarr(fname: Path, zarr_store: Path) -> None:
    with xr.open_dataset(fname, chunks={"time_counter": 48}) as ds:
        # If the store exists, check if this file's times are already inside it
        if zarr_store.exists():
            existing_ds = xr.open_zarr(zarr_store)
            existing_times = existing_ds.time_counter.values
            new_times = ds.time_counter.values

            # If all time steps in this .nc file are already in the Zarr store, skip entirely!
            if np.isin(new_times, existing_times).all():
                print(f"Skipping {fname.name}: Time steps already exist in Zarr store.")
                return
        # ----------------------

        ds_out = _format_dataset_for_zarr(ds)
        ds_chunked = ds_out.chunk({"time_counter": 48, "component": 2, "y": -1, "x": -1})

        if not zarr_store.exists():
            ds_chunked.to_zarr(zarr_store, mode="w")  # type: ignore
        else:
            ds_chunked.to_zarr(zarr_store, append_dim="time_counter")  # type: ignore


def preprocess_folder(nc_dir: Path | str, zarr_store: Path | str) -> None:
    """
    Finds all NetCDF files in a folder and iteratively appends them to a Zarr store.
    """
    nc_dir = Path(nc_dir)
    zarr_store = Path(zarr_store)

    # Sort files to ensure time is sequential!
    nc_files = sorted(nc_dir.glob("WINDS*.nc"))

    nc_files = nc_files[:120]

    for nc_file in nc_files:
        print(f"Processing {nc_file.name} to Zarr...")
        nc_file_to_zarr(nc_file, zarr_store)


# ==========================================
# Degradation Functions (Now support batches!)
# ==========================================


def _validate_dimensions(vel_array: np.ndarray, n: int) -> tuple[int, int]:
    # Support both 3D (component, y, x) and 4D (batch, component, y, x)
    if vel_array.ndim == 3:
        comp, num_lats, num_lons = vel_array.shape
    elif vel_array.ndim == 4:
        batch, comp, num_lats, num_lons = vel_array.shape
    else:
        raise ValueError("Input must be shape (2, lats, lons) or (batch, 2, lats, lons)")

    if comp != 2:
        raise ValueError("Input array must have 2 components (u, v).")
    if num_lats % n != 0 or num_lons % n != 0:
        raise ValueError(f"Grid dimensions must be cleanly divisible by n={n}.")

    return num_lats // n, num_lons // n


def _degrade_u(u: np.ndarray, n: int, coarse_lats: int, coarse_lons: int) -> np.ndarray:
    if u.ndim == 2:
        u_blocks = u.reshape(coarse_lats, n, coarse_lons, n)
        return u_blocks[:, :, :, n - 1].mean(axis=1)
    else:  # Batch mode
        batch_size = u.shape[0]
        u_blocks = u.reshape(batch_size, coarse_lats, n, coarse_lons, n)
        return u_blocks[:, :, :, :, n - 1].mean(axis=2)


def _degrade_v(v: np.ndarray, n: int, coarse_lats: int, coarse_lons: int) -> np.ndarray:
    if v.ndim == 2:
        v_blocks = v.reshape(coarse_lats, n, coarse_lons, n)
        return v_blocks[:, n - 1, :, :].mean(axis=2)
    else:  # Batch mode
        batch_size = v.shape[0]
        v_blocks = v.reshape(batch_size, coarse_lats, n, coarse_lons, n)
        return v_blocks[:, :, n - 1, :, :].mean(axis=3)


def degrade_velocities(vel_array: np.ndarray, n: int) -> np.ndarray:
    coarse_lats, coarse_lons = _validate_dimensions(vel_array, n)

    if vel_array.ndim == 3:
        u_coarse = _degrade_u(vel_array[0], n, coarse_lats, coarse_lons)
        v_coarse = _degrade_v(vel_array[1], n, coarse_lats, coarse_lons)
        return np.stack((u_coarse, v_coarse), axis=0)
    else:
        u_coarse = _degrade_u(vel_array[:, 0], n, coarse_lats, coarse_lons)
        v_coarse = _degrade_v(vel_array[:, 1], n, coarse_lats, coarse_lons)
        return np.stack((u_coarse, v_coarse), axis=1)  # Keeps batch dimension


def degrade_zarr_store(config: DataConfig) -> None:
    """
    Reads a high-resolution Zarr store in batches, degrades the velocity fields,
    and writes them to a new low-resolution Zarr store. Data on the West and Bottom
    boundaries are trimmed to ensure the grid is divisible by the degradation factor.
    """

    input_zarr = config.original_res
    output_zarr = config.degraded_res

    n = config.degrade_factor
    print(f"Degradation factor: {n}")

    ds_in = xr.open_zarr(input_zarr)

    nx = ds_in.sizes["x"]
    ny = ds_in.sizes["y"]

    nx, ny = ds_in.sizes["x"], ds_in.sizes["y"]
    y_slice, x_slice = get_spatial_trim_slices(nx, ny, n)

    ds_in = ds_in.isel(x=x_slice, y=y_slice)

    num_times = ds_in.sizes["time_counter"]
    chunk_size = 48  # Number of time steps to degrade in memory at once

    # Pre-load the times we have already degraded (if the target store exists)
    existing_times = None
    if output_zarr.exists():
        print(f"{output_zarr} already exists")
        existing_ds = xr.open_zarr(output_zarr)
        if "time_counter" in existing_ds:
            existing_times = existing_ds.time_counter.values
    # ----------------------

    for start in range(0, num_times, chunk_size):
        end = min(start + chunk_size, num_times)

        # Check the times for this specific batch
        batch_times = ds_in.time_counter.isel(time_counter=slice(start, end)).values

        # If we have an existing store, check if this entire batch is already inside it
        if existing_times is not None and np.isin(batch_times, existing_times).all():
            print(f"Skipping time steps {start} to {end} (Already degraded).")
            continue

        print(f"Degrading time steps {start} to {end}...")

        # Load batch of times into memory (Shape: batch, 2, y, x)
        # Note: vel_chunk is now already trimmed on both x and y axes
        vel_chunk = ds_in.velocity.isel(time_counter=slice(start, end)).values

        # Degrade entire batch simultaneously
        degraded_chunk = degrade_velocities(vel_chunk, n)

        # Format back into an Xarray dataset
        ds_out = xr.Dataset(
            data_vars=dict(velocity=(["time_counter", "component", "y", "x"], degraded_chunk)),
            coords=dict(time_counter=batch_times, component=["u", "v"]),
        )

        ds_out_chunked = ds_out.chunk(
            {"time_counter": chunk_size, "component": 2, "y": -1, "x": -1}
        )

        if not output_zarr.exists():
            ds_out_chunked.to_zarr(output_zarr, mode="w")  # type: ignore
        else:
            ds_out_chunked.to_zarr(output_zarr, append_dim="time_counter")  # type: ignore


def _compute_c_grid_divergence(
    vel_array: np.ndarray, dx: float = 1.0, dy: float = 1.0
) -> np.ndarray:
    """
    Computes the discrete divergence of a 2D velocity field on an Arakawa C-grid.

    Parameters:
    vel_array: np.ndarray of shape (2, num_lats, num_lons)
               vel_array[0] is U (defined on Eastern faces)
               vel_array[1] is V (defined on Northern faces)
    dx: Grid spacing in the X (longitude) direction (default 1.0)
    dy: Grid spacing in the Y (latitude) direction (default 1.0)

    Returns:
    np.ndarray of shape (num_lats, num_lons) representing divergence at cell centers.
    """
    if vel_array.ndim != 3 or vel_array.shape[0] != 2:
        raise ValueError("Input array must have a shape of (2, num_lats, num_lons).")

    u = vel_array[0]
    v = vel_array[1]

    # --- 1. Zonal Divergence (dU/dx) ---
    # For cell (y, x), the Eastern face is u[y, x].
    # The Western face is the Eastern face of the cell to its left: u[y, x-1].
    u_west = np.zeros_like(u)
    u_west[:, 1:] = u[:, :-1]  # Shift right to align western faces

    du_dx = (u - u_west) / dx

    # --- 2. Meridional Divergence (dV/dy) ---
    # For cell (y, x), the Northern face is v[y, x].
    # The Southern face is the Northern face of the cell below it: v[y-1, x].
    v_south = np.zeros_like(v)
    v_south[1:, :] = v[:-1, :]  # Shift down to align southern faces

    dv_dy = (v - v_south) / dy

    # --- 3. Total Divergence ---
    return du_dx + dv_dy


# ===============================
# Linearly Interpolate Zarr
# ===============================


def interpolate_zarr_store(
    data_config: DataConfig,
    exp_config: ExperimentConfig,
    hyper_config: HyperparametersConfig,
) -> None:
    """
    Reads a low-resolution (degraded) Zarr store in batches, bilinearly
    interpolates the velocity fields back to the original resolution, and writes
    them to a new Zarr store.
    """
    degraded_zarr_path = data_config.degraded_res
    target_resolution_zarr = data_config.original_res
    output_zarr_path = exp_config.model_predictions

    # Open both datasets lazily (only reads metadata)
    ds_deg = xr.open_zarr(degraded_zarr_path)
    ds_orig = xr.open_zarr(target_resolution_zarr)

    # --- NEW: Trim target dataset to get correct dimensions ---
    # This is highly efficient; no array data is loaded into memory.
    nx, ny = ds_orig.sizes["x"], ds_orig.sizes["y"]
    y_slice, x_slice = get_spatial_trim_slices(nx, ny, data_config.degrade_factor)

    ds_orig_trimmed = ds_orig.isel(x=x_slice, y=y_slice)

    # Extract target (y, x) shape directly from the lazy trimmed dataset
    target_shape = (ds_orig_trimmed.sizes["y"], ds_orig_trimmed.sizes["x"])
    # ----------------------------------------------------------

    # Only run interpolation on test indices
    csv_path = data_config.splits / "test_indices.csv"
    test_indices = _read_indices_from_csv(csv_path)
    ds_deg = ds_deg.isel(time_counter=test_indices)

    num_times = ds_deg.sizes["time_counter"]

    # Pre-load the times we have already interpolated (if the target store exists)
    existing_times = None
    if output_zarr_path.exists():
        print(f"{output_zarr_path} already exists. Checking for existing times...")
        existing_ds = xr.open_zarr(output_zarr_path)
        if "time_counter" in existing_ds:
            existing_times = existing_ds.time_counter.values

    for start in range(0, num_times, hyper_config.batch_size):
        end = min(start + hyper_config.batch_size, num_times)

        # Check the times for this specific batch
        batch_times = ds_deg.time_counter.isel(time_counter=slice(start, end)).values

        # If we have an existing store, check if this entire batch is already inside it
        if existing_times is not None and np.isin(batch_times, existing_times).all():
            print(f"Skipping time steps {start} to {end} (Already interpolated).")
            continue

        print(f"Interpolating time steps {start} to {end}...")

        # Load batch of times into memory (Shape: batch, 2, y, x)
        vel_chunk_low = ds_deg.velocity.isel(time_counter=slice(start, end)).values

        # Convert to PyTorch tensor to utilize fast vectorized bilinear interpolation
        vel_tensor = torch.from_numpy(vel_chunk_low).float()

        # Interpolate spatially to the target shape
        # align_corners=True matches the behavior of the Up() block in your U-Net model
        vel_interp = F.interpolate(
            vel_tensor, size=target_shape, mode="bilinear", align_corners=True
        ).numpy()

        # Format back into an Xarray dataset
        ds_out = xr.Dataset(
            data_vars=dict(velocity=(["time_counter", "component", "y", "x"], vel_interp)),
            coords=dict(time_counter=batch_times, component=ds_deg.component.values),
        )

        # Re-chunk data to match standard format
        ds_out_chunked = ds_out.chunk(
            {"time_counter": hyper_config.batch_size, "component": 2, "y": -1, "x": -1}
        )

        # Save or append to the target Zarr store
        if not output_zarr_path.exists():
            ds_out_chunked.to_zarr(output_zarr_path, mode="w")  # type: ignore
        else:
            ds_out_chunked.to_zarr(output_zarr_path, append_dim="time_counter")  # type: ignore

    print(f"Interpolation complete. Output saved to {output_zarr_path}")
