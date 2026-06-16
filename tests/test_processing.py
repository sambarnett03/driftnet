from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Import your functions from your script file
from driftnet.data import _format_dataset_for_zarr, nc_file_to_zarr


@pytest.fixture
def base_coords() -> dict:
    """Fixture to provide standardized dimensions for dummy data."""
    return {"y": np.arange(10), "x": np.arange(20)}


def create_dummy_dataset(times: np.ndarray, coords: dict) -> xr.Dataset:
    """Helper function to quickly manufacture ocean-model-like NetCDF structures."""
    shape = (len(times), len(coords["y"]), len(coords["x"]))

    u_surf = xr.DataArray(
        np.ones(shape, dtype="float32"),
        coords={"time_counter": times, "y": coords["y"], "x": coords["x"]},
        dims=["time_counter", "y", "x"],
    )
    v_surf = xr.DataArray(
        np.zeros(shape, dtype="float32"),
        coords={"time_counter": times, "y": coords["y"], "x": coords["x"]},
        dims=["time_counter", "y", "x"],
    )
    return xr.Dataset({"u_surf": u_surf, "v_surf": v_surf})


# =====================================================================
# Unit Tests for _format_dataset_for_zarr
# =====================================================================


def test_format_dataset_for_zarr_success(base_coords):
    """Verifies that datasets are formatted, concatenated, and named properly."""
    times = np.arange(48)
    ds_input = create_dummy_dataset(times, base_coords)

    ds_output = _format_dataset_for_zarr(ds_input)

    # 1. Ensure it returns a dataset containing the variable 'velocity'
    assert isinstance(ds_output, xr.Dataset)
    assert "velocity" in ds_output.data_vars

    # 2. Check that stacking combined the dimensions correctly
    # Input was (48, 10, 20). Output shape should have a new component dimension of size 2.
    assert ds_output.velocity.shape == (48, 2, 10, 20)
    assert list(ds_output.velocity.dims) == ["time_counter", "component", "y", "x"]


def test_format_dataset_for_zarr_missing_keys():
    """Verifies that a KeyError is thrown if expected parameters are missing."""
    invalid_ds = xr.Dataset({"something_else": xr.DataArray([1, 2, 3])})

    with pytest.raises(KeyError, match="The variables u_surf or v_surf were not found"):
        _format_dataset_for_zarr(invalid_ds)


# =====================================================================
# Integration Tests for nc_file_to_zarr
# =====================================================================


def test_nc_file_to_zarr_initial_creation(tmp_path: Path, base_coords):
    """Tests that the function successfully creates a fresh Zarr store with correct chunking."""
    # Create mock NetCDF file
    times = np.arange(48)
    ds_input = create_dummy_dataset(times, base_coords)
    nc_file = tmp_path / "half_hour_01.nc"
    ds_input.to_netcdf(nc_file)

    zarr_store = tmp_path / "output.zarr"

    # Run pipeline function
    nc_file_to_zarr(nc_file, zarr_store)

    # Verify file layout was built
    assert zarr_store.exists()

    # Open the written Zarr database to check consistency
    ds_zarr = xr.open_zarr(zarr_store)
    assert "velocity" in ds_zarr.data_vars
    assert len(ds_zarr.time_counter) == 48

    # Assert Chunking matches specification exactly
    # Expected dict: {'component': (2,), 'time_counter': (48,), 'y': (10,), 'x': (20,)}
    chunks = ds_zarr.velocity.chunksizes
    assert chunks["time_counter"] == (48,)
    assert chunks["component"] == (2,)
    assert chunks["y"] == (10,)  # -1 translates to full dimension length
    assert chunks["x"] == (20,)  # -1 translates to full dimension length


def test_nc_file_to_zarr_append_functionality(tmp_path: Path, base_coords):
    """Tests that subsequent calls cleanly append data along the time axis."""
    zarr_store = tmp_path / "output.zarr"

    # 1. Process Day 1 (Timestamps 0 to 47)
    day1_times = np.arange(0, 48)
    ds_day1 = create_dummy_dataset(day1_times, base_coords)
    nc_file1 = tmp_path / "day1.nc"
    ds_day1.to_netcdf(nc_file1)

    nc_file_to_zarr(nc_file1, zarr_store)

    # 2. Process Day 2 (Timestamps 48 to 95)
    day2_times = np.arange(48, 96)
    ds_day2 = create_dummy_dataset(day2_times, base_coords)
    nc_file2 = tmp_path / "day2.nc"
    ds_day2.to_netcdf(nc_file2)

    nc_file_to_zarr(nc_file2, zarr_store)

    # 3. Validation
    ds_zarr = xr.open_zarr(zarr_store)

    # Time dimension should now be 48 + 48 = 96
    assert len(ds_zarr.time_counter) == 96

    # Chunking strategy must be intact across chunks: (48, 48)
    assert ds_zarr.velocity.chunksizes["time_counter"] == (48, 48)
