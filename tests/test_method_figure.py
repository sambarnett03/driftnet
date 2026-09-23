from unittest.mock import patch

import numpy as np
import pytest

from driftnet.plotting import (
    _block_average_2d,
    _crop_to_box,
    grid_spacing_km,
    plot_method_figure,
)


@pytest.fixture
def grids():
    lon, lat = np.meshgrid(np.arange(40.0, 42.0, 0.1), np.arange(-11.0, -9.0, 0.1))
    return lon, lat, _block_average_2d(lon, 5, 5), _block_average_2d(lat, 5, 5)


def test_grid_spacing_km_at_equator():
    lon, lat = np.meshgrid(np.arange(0.0, 1.0, 0.1), np.zeros(3))
    assert grid_spacing_km(lon, lat) == pytest.approx(11.132, rel=1e-3)


def test_crop_to_box_keeps_box_plus_margin(grids):
    lon, lat, _, _ = grids
    c_lon, c_lat, c_field = _crop_to_box(lon, lat, lon * 0, (40.5, 41.0, -10.5, -10.0))
    assert c_lon.shape == c_lat.shape == c_field.shape
    assert c_lon.min() < 40.5 and c_lon.max() > 41.0


def test_crop_to_box_outside_domain_raises(grids):
    lon, lat, _, _ = grids
    with pytest.raises(ValueError):
        _crop_to_box(lon, lat, lon, (0, 1, 0, 1))


def test_plot_method_figure_requires_box(grids):
    lon, lat, lr_lon, lr_lat = grids
    with pytest.raises(ValueError):
        plot_method_figure(lon, lat, lr_lon, lr_lat, lon, lr_lon, lon, corners="auto")


@patch("matplotlib.figure.Figure.savefig")
def test_plot_method_figure_saves_png_and_pdf(mock_savefig, grids, tmp_path):
    lon, lat, lr_lon, lr_lat = grids
    speed, lr_speed = np.ones_like(lon), np.ones_like(lr_lon)
    output = tmp_path / "method.png"

    plot_method_figure(
        lon,
        lat,
        lr_lon,
        lr_lat,
        speed,
        lr_speed,
        speed,
        corners=(40.3, 41.5, -10.7, -9.5),
        output_path=output,
    )

    saved = [call.args[0] for call in mock_savefig.call_args_list]
    assert saved == [output, output.with_suffix(".pdf")]
