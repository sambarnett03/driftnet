from unittest.mock import patch

import numpy as np
import pytest

from driftnet.plotting import _domain_outline, plot_speed_map, surface_speed, velocity_to_t_points


@pytest.fixture
def mock_coord_data():
    lon, lat = np.meshgrid(np.arange(40.0, 45.0), np.arange(-10.0, -5.0))
    return {"u_lon": lon + 0.5, "u_lat": lat, "v_lon": lon, "v_lat": lat + 0.5}


def test_velocity_to_t_points_uniform_flow(mock_coord_data):
    u = np.full((5, 5), 3.0)
    v = np.full((5, 5), 4.0)
    lon, lat, u_t, v_t = velocity_to_t_points(mock_coord_data, u, v)

    assert lon.shape == lat.shape == u_t.shape == v_t.shape == (4, 4)
    np.testing.assert_allclose(lon[0], np.arange(41.0, 45.0))
    np.testing.assert_allclose(lat[:, 0], np.arange(-9.0, -5.0))
    np.testing.assert_allclose(surface_speed(u_t, v_t), 5.0)


def test_velocity_to_t_points_averages_faces(mock_coord_data):
    u = np.tile(np.arange(5.0), (5, 1))  # u increases eastward
    v = np.zeros((5, 5))
    _, _, u_t, _ = velocity_to_t_points(mock_coord_data, u, v)
    np.testing.assert_allclose(u_t[0], [0.5, 1.5, 2.5, 3.5])


def test_velocity_to_t_points_keeps_time_axis(mock_coord_data):
    u = np.ones((3, 5, 5))
    _, _, u_t, v_t = velocity_to_t_points(mock_coord_data, u, u)
    assert u_t.shape == v_t.shape == (3, 4, 4)


def test_surface_speed_masks_land():
    speed = surface_speed(np.array([[0.0, 1.0]]), np.array([[0.0, 0.0]]))
    assert np.isnan(speed[0, 0])
    assert speed[0, 1] == 1.0


def test_plot_speed_map_shape_mismatch():
    with pytest.raises(ValueError):
        plot_speed_map(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((3, 3)), output_path=None)


@patch("matplotlib.figure.Figure.savefig")
def test_plot_speed_map_saves_png_and_pdf(mock_savefig, mock_coord_data, tmp_path):
    lon, lat, u_t, v_t = velocity_to_t_points(mock_coord_data, np.ones((5, 5)), np.ones((5, 5)))
    output = tmp_path / "speed.png"
    fig, ax = plot_speed_map(lon, lat, surface_speed(u_t, v_t), output_path=output)

    saved = [call.args[0] for call in mock_savefig.call_args_list]
    assert saved == [output, output.with_suffix(".pdf")]
    assert ax.get_title() == "Surface current speed"


def test_domain_outline_traces_grid_edge(mock_coord_data):
    lon, lat = mock_coord_data["v_lon"], mock_coord_data["u_lat"]
    outline_lon, outline_lat = _domain_outline(lon, lat)

    assert outline_lon.min() == lon.min() and outline_lon.max() == lon.max()
    assert outline_lat.min() == lat.min() and outline_lat.max() == lat.max()
    assert (outline_lon[0], outline_lat[0]) == (outline_lon[-1], outline_lat[-1])
