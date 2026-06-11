from pathlib import Path

import xarray as xr


def load_file(fname: Path) -> xr.Dataset:
    return xr.open_dataset(fname)


def nc_file_to_npys():
    # Load .nc file

    # Loop through each time counter (1440 total) as too big to load all at once

    # Stack u_surf and v_surf

    # Save as .npy file to data folder (read from config) - use datetime as filename
    pass
