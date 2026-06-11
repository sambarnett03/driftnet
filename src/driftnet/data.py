from pathlib import Path
import xarray as xr

def load_file(fname: Path) -> xr.Dataset:
    return xr.open_dataset(fname)