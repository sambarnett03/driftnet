from pathlib import Path
import xarray as xr
import numpy as np
from numpy.typing import NDArray


def _extract_u_and_v(
    ds: xr.Dataset,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    if 'u_surf' not in list(ds.keys()) or 'v_surf' not in list(ds.keys()):
        raise KeyError('The variables u_surf or v_surf were not found')

    u = ds.u_surf.values
    v = ds.v_surf.values

    if u.ndim != 2 or v.ndim != 2:
        raise ValueError('u or v does not have dimensions 2. Ensure time axis is removed')

    return u, v


def _create_npy_file_name(time: np.datetime64) -> str:
    date_time = str(time)
    return 'velocities' + date_time + '.npy'


def load_file(fname: Path) -> xr.Dataset:
    if not fname.exists():
        raise FileNotFoundError(f'File {fname} could not be found ')
    return xr.open_dataset(fname)


def nc_file_to_npys(fname: Path, save_dir: Path):
    # Load .nc file
    ds = load_file(fname)

    # Loop through each time counter (1440 total) as too big to load all at once
    for t in ds.time_counter.values:
        print(t)
        ds_filtered = ds.sel({"time_counter": t})

        # Stack u_surf and v_surf
        u, v = _extract_u_and_v(ds_filtered)
        vel_field = np.stack((u, v))

        # Save as .npy file to data folder (read from config) - use datetime as filename
        save_fname = _create_npy_file_name(t)
        np.save(save_dir / save_fname, vel_field)
    return
