from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def _extract_u_and_v(
    ds: xr.Dataset,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # Direct dictionary lookup is cleaner and faster
    if "u_surf" not in ds or "v_surf" not in ds:
        raise KeyError("The variables u_surf or v_surf were not found")

    u = ds.u_surf.values
    v = ds.v_surf.values

    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u or v does not have dimensions 2. Ensure time axis is removed")

    return u, v


def _create_npy_file_name(time: np.datetime64) -> str:
    # Replace colons to prevent Windows crashing and command-line headaches
    date_time = str(time).replace(":", "-")
    return f"velocities_{date_time}.npy"


def nc_file_to_npys(fname: Path, save_dir: Path):
    # Ensure the save directory actually exists before writing to it
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use a context manager to auto-close the file when finished
    with xr.open_dataset(fname) as ds:
        # Loop by index (isel) instead of coordinate value (sel) for speed
        num_times = ds.sizes["time_counter"]

        for i in range(num_times):
            ds_filtered = ds.isel(time_counter=i)
            t = ds_filtered.time_counter.values

            # Skip if file already exists
            save_fname = _create_npy_file_name(t)
            if (save_dir / save_fname).exists():
                print(f"{save_dir / save_fname} already exists")
                continue

            print(f"Processing time step {i + 1}/{num_times}: {t}")

            # Stack u_surf and v_surf
            u, v = _extract_u_and_v(ds_filtered)
            vel_field = np.stack((u, v))

            np.save(save_dir / save_fname, vel_field)


def _validate_dimensions(vel_array: np.ndarray, n: int) -> tuple[int, int]:
    """
    Validates that the input array has the correct dimensions and shapes,
    and returns the expected coarse grid dimensions.
    """
    if vel_array.ndim != 3 or vel_array.shape[0] != 2:
        raise ValueError("Input array must have a shape of (2, num_lats, num_lons).")

    _, num_lats, num_lons = vel_array.shape

    if num_lats % n != 0 or num_lons % n != 0:
        raise ValueError(
            f"Grid dimensions ({num_lats}, {num_lons}) must be cleanly divisible by n={n}."
        )

    return num_lats // n, num_lons // n


def _degrade_u(u: np.ndarray, n: int, coarse_lats: int, coarse_lons: int) -> np.ndarray:
    """
    Degrades the U velocity component on an Arakawa C-grid by averaging
    the high-resolution velocities along the Eastern faces of the blocks.
    """
    u_blocks = u.reshape(coarse_lats, n, coarse_lons, n)
    u_east_faces = u_blocks[:, :, :, n - 1]
    return u_east_faces.mean(axis=1)


def _degrade_v(v: np.ndarray, n: int, coarse_lats: int, coarse_lons: int) -> np.ndarray:
    """
    Degrades the V velocity component on an Arakawa C-grid by averaging
    the high-resolution velocities along the Northern faces of the blocks.
    """
    v_blocks = v.reshape(coarse_lats, n, coarse_lons, n)
    v_north_faces = v_blocks[:, n - 1, :, :]
    return v_north_faces.mean(axis=2)


def _save_array(vel_array: np.ndarray, save_dir: str | Path, filename: str) -> Path:
    """
    Safely creates the directory if it doesn't exist and saves the array as a .npy file.
    Returns the final Path object pointing to the saved file.
    """
    output_dir = Path(save_dir)
    # create parents (like mkdir -p) if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename.replace(":", "-")
    np.save(file_path, vel_array)
    return file_path


def degrade_velocities(
    fname: Path,
    n: int,
    save_dir: str | Path,
) -> np.ndarray:
    """
    Main entry point to degrade high-resolution U and V velocity fields.

    Parameters:
    vel_array: A numpy array of shape (2, num_lats, num_lons)
    n: The integer degradation factor.
    save_dir: Optional directory path. If provided, saves the output array.
    filename: Name of the file to save (defaults to 'degraded_velocities.npy').

    Returns:
    A numpy array of shape (2, num_lats // n, num_lons // n)
    """
    vel_array = np.load(fname)

    coarse_lats, coarse_lons = _validate_dimensions(vel_array, n)

    u_coarse = _degrade_u(vel_array[0], n, coarse_lats, coarse_lons)
    v_coarse = _degrade_v(vel_array[1], n, coarse_lats, coarse_lons)

    degraded = np.stack((u_coarse, v_coarse), axis=0)

    _save_array(degraded, save_dir, f"degraded_n_{n}_" + fname.name)

    return degraded
