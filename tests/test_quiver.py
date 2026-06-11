from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Adjust this import to match your actual module name
from driftnet.plotting import (
    _clip_to_extent,
    _downsample,
    _get_extent,
    _get_gridline_values,
    _parse_corners,
    plot_velocity_quiver,
)

# --- Fixtures ---


@pytest.fixture
def mock_coord_data():
    """Fixture providing a mock xarray Dataset with required coordinate variables."""
    lon = np.array([[10, 11, 12], [10, 11, 12], [10, 11, 12]], dtype=float)
    lat = np.array([[20, 20, 20], [21, 21, 21], [22, 22, 22]], dtype=float)

    return {
        "u_lon": lon,
        "u_lat": lat,
        "v_lon": lon + 0.5,  # Slight offset for v grid
        "v_lat": lat + 0.5,
    }


# --- Tests for Helper Functions ---


def test_parse_corners_none():
    assert _parse_corners(None) is None


def test_parse_corners_bounds():
    corners = (10.0, 20.0, 30.0, 40.0)
    assert _parse_corners(corners) == (10.0, 20.0, 30.0, 40.0)


def test_parse_corners_points():
    corners = [(10.0, 30.0), (20.0, 30.0), (20.0, 40.0), (10.0, 40.0)]
    assert _parse_corners(corners) == (10.0, 20.0, 30.0, 40.0)


def test_parse_corners_invalid():
    with pytest.raises(ValueError, match="corners must be either"):
        _parse_corners([10.0, 20.0])


def test_get_extent_with_corners():
    lon = np.array([0.0, 5.0])
    lat = np.array([0.0, 5.0])
    corners = (1.0, 4.0, 1.0, 4.0)
    assert _get_extent(lon, lat, corners=corners) == (1.0, 4.0, 1.0, 4.0)


def test_get_extent_without_corners():
    lon = np.array([10.0, 15.0])
    lat = np.array([20.0, 25.0])
    # Padding is 2 by default
    expected = (8.0, 17.0, 18.0, 27.0)
    assert _get_extent(lon, lat) == expected


def test_clip_to_extent():
    lon = np.array([10, 15, 20], dtype=float)
    lat = np.array([30, 35, 40], dtype=float)
    u = np.array([1, 2, 3], dtype=float)
    v = np.array([4, 5, 6], dtype=float)
    extent = (12.0, 22.0, 32.0, 42.0)

    lon_c, lat_c, u_c, v_c = _clip_to_extent(lon, lat, u, v, extent)

    np.testing.assert_array_equal(lon_c, [15, 20])
    np.testing.assert_array_equal(lat_c, [35, 40])
    np.testing.assert_array_equal(u_c, [2, 3])
    np.testing.assert_array_equal(v_c, [5, 6])


def test_downsample():
    lon = np.arange(4, dtype=float).reshape(2, 2)
    lat = np.arange(4, 8, dtype=float).reshape(2, 2)
    u = np.arange(8, 12, dtype=float).reshape(2, 2)
    v = np.arange(12, 16, dtype=float).reshape(2, 2)

    lon_d, lat_d, u_d, v_d = _downsample(lon, lat, u, v, stride=2)

    np.testing.assert_array_equal(lon_d, [[0]])
    np.testing.assert_array_equal(lat_d, [[4]])
    np.testing.assert_array_equal(u_d, [[8]])
    np.testing.assert_array_equal(v_d, [[12]])


def test_get_gridline_values():
    values = _get_gridline_values(12.5, 17.5, 2.0)
    np.testing.assert_array_equal(values, [12.0, 14.0, 16.0, 18.0])


# --- Tests for Main Plotting Function ---


def test_plot_velocity_quiver_missing_inputs(mock_coord_data):
    with pytest.raises(TypeError, match="Received None for both u_input and v_input."):
        plot_velocity_quiver(mock_coord_data, u_input=None, v_input=None)


def test_plot_velocity_quiver_invalid_stride(mock_coord_data):
    u = np.ones((3, 3))
    with pytest.raises(ValueError, match="stride must be a positive integer."):
        plot_velocity_quiver(mock_coord_data, u_input=u, stride=0)


@patch("matplotlib.figure.Figure.savefig")
def test_plot_velocity_quiver_execution(mock_savefig, mock_coord_data, tmp_path):
    """Test that the plotting function runs without errors and attempts to save."""
    u = np.ones((3, 3))
    v = np.ones((3, 3)) * 2

    # Use a tmp_path to ensure cross-platform path handling is working
    test_out_path = tmp_path / "test_plot.png"

    fig, ax = plot_velocity_quiver(
        coord_data=mock_coord_data,
        u_input=u,
        v_input=v,
        stride=1,
        title="Test Plot",
        output_path=test_out_path,
        gridline_interval=1.0,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Test Plot"

    # Verify savefig was called once with the correct arguments
    mock_savefig.assert_called_once_with(Path(test_out_path), bbox_inches="tight")


@patch("matplotlib.figure.Figure.savefig")
def test_plot_velocity_quiver_no_save(mock_savefig, mock_coord_data):
    """Test that setting output_path to None bypasses saving."""
    u = np.ones((3, 3))

    plot_velocity_quiver(coord_data=mock_coord_data, u_input=u, output_path=None)

    mock_savefig.assert_not_called()
