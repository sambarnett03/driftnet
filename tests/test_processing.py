import numpy as np
import pytest
import xarray as xr

# Import your functions here.
from driftnet.data import _create_npy_file_name, _extract_u_and_v, nc_file_to_npys

# --- Fixtures ---


@pytest.fixture
def dummy_dataset():
    """Creates a dummy 3D xarray dataset (time, y, x) for testing."""
    times = np.array(["2026-06-11T10:00:00", "2026-06-11T11:00:00"], dtype="datetime64[ns]")
    y = np.arange(5)
    x = np.arange(5)

    # Create random 3D data arrays
    u_data = np.random.rand(len(times), len(y), len(x))
    v_data = np.random.rand(len(times), len(y), len(x))

    ds = xr.Dataset(
        data_vars=dict(
            u_surf=(["time_counter", "y", "x"], u_data),
            v_surf=(["time_counter", "y", "x"], v_data),
        ),
        coords=dict(
            time_counter=times,
            y=y,
            x=x,
        ),
    )
    return ds


# --- Tests for _extract_u_and_v ---


def test_extract_u_and_v_success(dummy_dataset):
    # Slice the dataset to 2D to simulate what happens inside the loop
    ds_2d = dummy_dataset.isel(time_counter=0)

    u, v = _extract_u_and_v(ds_2d)

    assert u.shape == (5, 5)
    assert v.shape == (5, 5)
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)


def test_extract_u_and_v_missing_keys(dummy_dataset):
    ds_2d = dummy_dataset.isel(time_counter=0)
    # Drop a required variable
    ds_missing = ds_2d.drop_vars("u_surf")

    with pytest.raises(KeyError, match="The variables u_surf or v_surf were not found"):
        _extract_u_and_v(ds_missing)


def test_extract_u_and_v_wrong_dimensions(dummy_dataset):
    # Pass the full 3D dataset without slicing time
    with pytest.raises(ValueError, match="u or v does not have dimensions 2"):
        _extract_u_and_v(dummy_dataset)


# --- Tests for _create_npy_file_name ---


def test_create_npy_file_name():
    test_time = np.datetime64("2026-06-11T11:15:59")
    expected_name = "velocities_2026-06-11T11-15-59.npy"

    result = _create_npy_file_name(test_time)

    assert result == expected_name


# --- Tests for nc_file_to_npys ---


def test_nc_file_to_npys(dummy_dataset, tmp_path):
    # tmp_path is a built-in pytest fixture that provides a temporary
    # directory unique to the test invocation
    input_nc = tmp_path / "test_input.nc"
    save_dir = tmp_path / "output_npys"

    # Save the dummy dataset to the temporary path
    dummy_dataset.to_netcdf(input_nc)

    # Run the main function
    nc_file_to_npys(input_nc, save_dir)

    # Verify the save directory was created
    assert save_dir.exists()

    # Verify the correct number of files were created (2 time steps = 2 files)
    saved_files = list(save_dir.glob("*.npy"))
    assert len(saved_files) == 2

    # Load one of the saved numpy arrays to verify its contents and shape
    first_time = dummy_dataset.time_counter.values[0]
    expected_filename = _create_npy_file_name(first_time)

    loaded_array = np.load(save_dir / expected_filename)

    # Shape should be (2 variables, 5 y-coords, 5 x-coords)
    assert loaded_array.shape == (2, 5, 5)

    # Verify the values match the original dataset
    np.testing.assert_array_almost_equal(
        loaded_array[0], dummy_dataset.isel(time_counter=0).u_surf.values
    )
    np.testing.assert_array_almost_equal(
        loaded_array[1], dummy_dataset.isel(time_counter=0).v_surf.values
    )
