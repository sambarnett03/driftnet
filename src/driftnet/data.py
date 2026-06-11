from pathlib import Path
import xarray as xr
import numpy as np
from numpy.typing import NDArray


def _extract_u_and_v(
    ds: xr.Dataset,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # Direct dictionary lookup is cleaner and faster
    if 'u_surf' not in ds or 'v_surf' not in ds:
        raise KeyError('The variables u_surf or v_surf were not found')

    u = ds.u_surf.values
    v = ds.v_surf.values

    if u.ndim != 2 or v.ndim != 2:
        raise ValueError('u or v does not have dimensions 2. Ensure time axis is removed')

    return u, v


def _create_npy_file_name(time: np.datetime64) -> str:
    # Replace colons to prevent Windows crashing and command-line headaches
    date_time = str(time).replace(':', '-')
    return f"velocities_{date_time}.npy"


def nc_file_to_npys(fname: Path, save_dir: Path):
    # Ensure the save directory actually exists before writing to it
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use a context manager to auto-close the file when finished
    with xr.open_dataset(fname) as ds:

        # Loop by index (isel) instead of coordinate value (sel) for speed
        num_times = ds.sizes['time_counter']

        for i in range(num_times):
            ds_filtered = ds.isel(time_counter=i)
            t = ds_filtered.time_counter.values
            print(f"Processing time step {i+1}/{num_times}: {t}")

            # Stack u_surf and v_surf
            u, v = _extract_u_and_v(ds_filtered)
            vel_field = np.stack((u, v))

            # Save file
            save_fname = _create_npy_file_name(t)
            np.save(save_dir / save_fname, vel_field)

    return