from unittest.mock import patch

import numpy as np
import pytest

from driftnet.plotting import _box_slices, most_energetic_box, plot_velocity_grid


@pytest.fixture
def grid():
    return np.meshgrid(np.arange(40.0, 44.0, 0.05), np.arange(-12.0, -8.0, 0.05))


def test_box_slices_cover_box(grid):
    lon, lat = grid
    rows, cols = _box_slices(lon, lat, (41.0, 42.0, -11.0, -10.0), margin=0)
    assert lon[rows, cols].min() >= 41.0 and lon[rows, cols].max() <= 42.0
    assert lat[rows, cols].min() >= -11.0 and lat[rows, cols].max() <= -10.0


def test_most_energetic_box_finds_jet(grid):
    lon, lat = grid
    speed = np.exp(-((lon - 42.5) ** 2 + (lat + 9.5) ** 2) / 0.1)
    lon_min, lon_max, lat_min, lat_max = most_energetic_box(lon, lat, speed, size_deg=1.0)
    assert lon_min < 42.5 < lon_max
    assert lat_min < -9.5 < lat_max
    assert lon_max - lon_min == pytest.approx(1.0)


def test_plot_velocity_grid_rejects_too_many_fields(grid):
    lon, lat = grid
    fields = {str(i): (lon, lat) for i in range(5)}
    with pytest.raises(ValueError):
        plot_velocity_grid(lon, lat, fields, corners=(41, 42, -11, -10), output_path=None)


@patch("matplotlib.figure.Figure.savefig")
def test_plot_velocity_grid_titles_and_rmse(mock_savefig, grid, tmp_path):
    lon, lat = grid
    u, v = np.ones_like(lon), np.zeros_like(lon)
    fields = {
        "Ground truth": (u, v),
        "Interpolation": (u + 0.1, v),
        "U-Net": (u, v),
        "Diffusion": (u, v - 0.2),
    }
    output = tmp_path / "grid.png"
    fig = plot_velocity_grid(lon, lat, fields, corners=(41, 43, -11, -9), output_path=output)

    geo_axes = [ax for ax in fig.axes if ax.get_title()]
    assert [ax.get_title() for ax in geo_axes] == list(fields)
    texts = [t.get_text() for ax in geo_axes for t in ax.texts]
    assert "reference" in texts
    assert any("0.100" in t for t in texts)
    assert any("0.200" in t for t in texts)

    saved = [call.args[0] for call in mock_savefig.call_args_list]
    assert saved == [output, output.with_suffix(".pdf")]
