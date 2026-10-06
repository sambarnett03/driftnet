import numpy as np
import pytest

from driftnet.plotting import (
    animate_speed_series,
    frame_to_frame_change,
    plot_flicker,
    plot_frame_strip,
)


@pytest.fixture
def series():
    lon, lat = np.meshgrid(np.arange(40.0, 41.0, 0.1), np.arange(-11.0, -10.0, 0.1))
    times = np.datetime64("2010-01-01T00:00") + np.arange(4) * np.timedelta64(30, "m")
    steady = np.ones((4, *lon.shape))
    rng = np.random.default_rng(0)
    speeds = {"Ground truth": steady, "Noisy": steady + rng.normal(0, 0.1, steady.shape)}
    return lon, lat, times, speeds


def test_frame_to_frame_change_is_zero_for_steady_field():
    np.testing.assert_allclose(frame_to_frame_change(np.ones((3, 4, 4))), [0.0, 0.0])


def test_frame_to_frame_change_measures_step():
    speed = np.stack([np.zeros((2, 2)), np.full((2, 2), 0.5)])
    np.testing.assert_allclose(frame_to_frame_change(speed), [0.5])


def test_frame_to_frame_change_ignores_land():
    speed = np.stack([np.array([[0.0, np.nan]]), np.array([[0.2, np.nan]])])
    np.testing.assert_allclose(frame_to_frame_change(speed), [0.2])


def test_plot_flicker_labels_means(series, tmp_path):
    _, _, times, speeds = series
    fig = plot_flicker(times, speeds, output_path=tmp_path / "flicker.png")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[0] == "Ground truth (mean 0.000)"
    assert labels[1].startswith("Noisy (mean 0.1")
    assert (tmp_path / "flicker.png").exists()


def test_plot_frame_strip_grid(series, tmp_path):
    lon, lat, times, speeds = series
    fig = plot_frame_strip(lon, lat, times, speeds, n_columns=3, output_path=tmp_path / "strip.png")
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["t + 0 h", "t + 0.5 h", "t + 1 h"]
    assert (tmp_path / "strip.png").exists()


def test_animate_speed_series_writes_gif(series, tmp_path):
    lon, lat, times, speeds = series
    path = animate_speed_series(lon, lat, times, speeds, dpi=40, output_path=tmp_path / "anim.gif")
    assert path == tmp_path / "anim.gif"
    assert path.stat().st_size > 0
