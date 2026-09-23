from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from numpy.typing import ArrayLike, NDArray

Bounds = tuple[float, float, float, float]
BoundsLike = Sequence[float]
CornerPoint = tuple[float, float]
CornerPoints = Sequence[CornerPoint]
Corners = BoundsLike | CornerPoints


def _parse_corners(corners: Corners | None) -> Bounds | None:
    """Return lon/lat bounds from either bounds or four corner points."""
    if corners is None:
        return None

    corners_array = np.asarray(corners, dtype=float)

    if corners_array.shape == (4,):
        lon_min, lon_max, lat_min, lat_max = corners_array
        return float(lon_min), float(lon_max), float(lat_min), float(lat_max)

    if corners_array.shape == (4, 2):
        lons = corners_array[:, 0]
        lats = corners_array[:, 1]
        return (
            float(np.nanmin(lons)),
            float(np.nanmax(lons)),
            float(np.nanmin(lats)),
            float(np.nanmax(lats)),
        )

    raise ValueError(
        "corners must be either "
        "(lon_min, lon_max, lat_min, lat_max) or "
        "[(lon1, lat1), ..., (lon4, lat4)]."
    )


def _get_extent(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    corners: Corners | None = None,
    padding: float = 2,
) -> Bounds:
    """Return plot extent, using corners if supplied."""
    corner_bounds = _parse_corners(corners)

    if corner_bounds is not None:
        return corner_bounds

    return (
        float(np.nanmin(lon) - padding),
        float(np.nanmax(lon) + padding),
        float(np.nanmin(lat) - padding),
        float(np.nanmax(lat) + padding),
    )


def _clip_to_extent(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    extent: Bounds,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Clip coordinates and velocity components to the given extent."""
    lon_min, lon_max, lat_min, lat_max = extent

    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    return lon[mask], lat[mask], u[mask], v[mask]


def _downsample(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    stride: int,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Downsample coordinates and velocity components."""
    return (
        lon[::stride, ::stride],
        lat[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
    )


def _add_map_features(ax: GeoAxes) -> None:
    """Add standard map features to a Cartopy axis."""
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, zorder=0)


def _get_gridline_values(
    min_value: float,
    max_value: float,
    interval: float,
) -> NDArray[np.floating]:
    """Return gridline values covering the requested interval."""
    start = np.floor(min_value / interval) * interval
    stop = np.ceil(max_value / interval) * interval

    return np.arange(start, stop + interval, interval)


def _add_gridlines(
    ax: GeoAxes,
    extent: Bounds,
    gridline_interval: float | None,
) -> None:
    """Add labelled gridlines, optionally at a fixed degree interval."""
    gridlines = ax.gridlines(draw_labels=True)

    if gridline_interval is None:
        return

    if gridline_interval <= 0:
        raise ValueError("gridline_interval must be positive.")

    lon_min, lon_max, lat_min, lat_max = extent

    gridlines.xlocator = mticker.FixedLocator(
        _get_gridline_values(lon_min, lon_max, gridline_interval).tolist()
    )
    gridlines.ylocator = mticker.FixedLocator(
        _get_gridline_values(lat_min, lat_max, gridline_interval).tolist()
    )


def _plot_quiver_component(
    ax: GeoAxes,
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    extent: Bounds,
    stride: int,
    scale: float | None,
) -> Quiver:
    """Downsample, clip, and plot one quiver component."""
    lon_q, lat_q, u_q, v_q = _downsample(lon, lat, u, v, stride)
    lon_q, lat_q, u_q, v_q = _clip_to_extent(
        lon_q,
        lat_q,
        u_q,
        v_q,
        extent,
    )

    return ax.quiver(
        lon_q,
        lat_q,
        u_q,
        v_q,
        transform=ccrs.PlateCarree(),
        scale=scale,
    )


def plot_velocity_quiver(
    coord_data: dict,
    u_input: ArrayLike | None = None,
    v_input: ArrayLike | None = None,
    corners: Corners | None = None,
    stride: int = 10,
    title: str = "Surface velocity",
    scale: float | None = None,
    figsize: tuple[float, float] = (8, 8),
    output_path: str | Path | None = "images/velocity_field.png",
    gridline_interval: float | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot velocity vectors, optionally clipped to a lon/lat box.

    Parameters
    ----------
    coord_data : dict

    u_input, v_input : array-like, optional
        2D velocity components. At least one must be supplied. If only one
        component is supplied, the missing component is plotted as zero.

    corners : tuple or list, optional
        Clip box corners. May be either:

        ``(lon_min, lon_max, lat_min, lat_max)``

        or four corner points:

        ``[(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)]``

        The function uses the min/max longitude and latitude of the corners.

    stride : int, default 10
        Plot every ``stride`` grid point.

    title : str, default "Surface velocity"
        Plot title.

    scale : float, optional
        Quiver scale. Larger values make arrows smaller.

    figsize : tuple, default (8, 8)
        Figure size.

    output_path : str or pathlib.Path, optional
        Path where the figure is saved. If ``None``, the figure is not saved.

    gridline_interval : float, optional
        Interval in degrees between labelled longitude and latitude gridlines.
        For example, ``gridline_interval=0.02`` draws gridlines every
        0.02 degrees.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    if u_input is None and v_input is None:
        raise TypeError("Received None for both u_input and v_input.")

    if stride < 1:
        raise ValueError("stride must be a positive integer.")

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    ax = cast(GeoAxes, plt.axes(projection=projection))

    if u_input is not None:
        nlat, nlon = np.array(u_input).shape

    else:
        nlat, nlon = np.array(v_input).shape

    u_lon = coord_data["u_lon"]
    u_lat = coord_data["u_lat"]
    v_lon = coord_data["v_lon"]
    v_lat = coord_data["v_lat"]

    base_lat, base_lon = u_lon.shape

    res_lat = base_lat // nlat
    res_lon = base_lon // nlon

    u_lon = u_lon[::res_lon, ::res_lon]
    v_lon = v_lon[::res_lon, ::res_lon]
    u_lat = u_lat[::res_lat, ::res_lat]
    v_lat = v_lat[::res_lat, ::res_lat]

    extent_lons = []
    extent_lats = []

    if u_input is not None:
        extent_lons.append(u_lon)
        extent_lats.append(u_lat)

    if v_input is not None:
        extent_lons.append(v_lon)
        extent_lats.append(v_lat)

    lon_for_extent = np.concatenate([lon.ravel() for lon in extent_lons])
    lat_for_extent = np.concatenate([lat.ravel() for lat in extent_lats])
    extent = _get_extent(lon_for_extent, lat_for_extent, corners=corners)

    ax.set_extent(extent, crs=projection)
    _add_map_features(ax)

    if u_input is not None:
        u = np.asarray(u_input)
        v_zero = np.zeros_like(u)

        _plot_quiver_component(
            ax=ax,
            lon=u_lon,
            lat=u_lat,
            u=u,
            v=v_zero,
            extent=extent,
            stride=stride,
            scale=scale,
        )

    if v_input is not None:
        v = np.asarray(v_input)
        u_zero = np.zeros_like(v)

        _plot_quiver_component(
            ax=ax,
            lon=v_lon,
            lat=v_lat,
            u=u_zero,
            v=v,
            extent=extent,
            stride=stride,
            scale=scale,
        )

    _add_gridlines(
        ax=ax,
        extent=extent,
        gridline_interval=gridline_interval,
    )

    ax.set_title(title)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, ax


def velocity_to_t_points(
    coord_data: dict,
    u: ArrayLike,
    v: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray, NDArray]:
    """
    Interpolate C-grid u and v onto shared cell centres (T points).

    Assumes NEMO staggering: ``u[j, i]`` sits on the east face and ``v[j, i]``
    on the north face of cell ``(j, i)``. The first row and column are dropped
    because they have no west/south neighbour.

    Works on NumPy or Dask arrays, with any number of leading (e.g. time) axes.

    Returns
    -------
    lon, lat, u_t, v_t
        2D T-point coordinates and the interpolated velocity components.
    """
    u = cast(NDArray, u)
    v = cast(NDArray, v)

    u_t = 0.5 * (u[..., 1:, 1:] + u[..., 1:, :-1])
    v_t = 0.5 * (v[..., 1:, 1:] + v[..., :-1, 1:])

    # T points share their row with u points and their column with v points.
    lon = np.asarray(coord_data["v_lon"])[1:, 1:]
    lat = np.asarray(coord_data["u_lat"])[1:, 1:]

    return lon, lat, u_t, v_t


def surface_speed(u_t: ArrayLike, v_t: ArrayLike) -> NDArray[np.floating]:
    """Return current speed, with land (u = v = 0 or NaN) set to NaN."""
    u_t = np.asarray(u_t, dtype=float)
    v_t = np.asarray(v_t, dtype=float)

    speed = np.hypot(u_t, v_t)
    speed[(u_t == 0) & (v_t == 0)] = np.nan

    return speed


def plot_speed_map(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    speed: NDArray[np.floating],
    corners: Corners | None = None,
    title: str | None = "Surface current speed",
    cmap: str = "viridis",
    vmax: float | None = None,
    figsize: tuple[float, float] = (10, 10),
    font_size: float = 14,
    dpi: int = 300,
    output_path: str | Path | None = "images/surface_speed.png",
) -> tuple[Figure, Axes]:
    """
    Plot a map of surface current speed, styled for posters.

    Parameters
    ----------
    lon, lat : 2D arrays
        Coordinates of each speed value (e.g. from ``velocity_to_t_points``).

    speed : 2D array
        Current speed in m/s. NaNs (land) are left blank under the land mask.

    corners : tuple or list, optional
        Map extent, in any form accepted by ``plot_velocity_quiver``.
        Defaults to the full model domain.

    cmap : str, default "viridis"
        Sequential colormap. "cmo.speed" works if ``cmocean`` is imported.

    vmax : float, optional
        Top of the colour scale. Defaults to the 99th percentile of speed so a
        few extreme cells do not wash out the rest of the map.

    dpi : int, default 300
        Resolution of the saved PNG. A PDF is also saved alongside it, with the
        speed field rasterised and the text/coastlines kept as vectors.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    speed = np.asarray(speed)

    if not (lon.shape == lat.shape == speed.shape):
        raise ValueError("lon, lat and speed must all have the same shape.")

    if vmax is None:
        vmax = float(np.nanpercentile(speed, 99))

    extent = _get_extent(lon, lat, corners=corners, padding=0)
    projection = ccrs.PlateCarree()
    land_colour = "#d9d9d9"

    with plt.rc_context({"font.size": font_size}):
        fig = plt.figure(figsize=figsize)
        ax = cast(GeoAxes, plt.axes(projection=projection))
        ax.set_extent(extent, crs=projection)
        # Model land cells (NaN) that Natural Earth misses should still read as land.
        ax.set_facecolor(land_colour)

        mesh = ax.pcolormesh(
            lon,
            lat,
            np.ma.masked_invalid(speed),
            transform=projection,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            shading="nearest",
            rasterized=True,
        )

        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=land_colour, zorder=2)
        ax.coastlines(resolution="10m", linewidth=0.6, color="#404040", zorder=3)

        gridlines = ax.gridlines(
            draw_labels=True, linewidth=0.4, color="white", alpha=0.4, linestyle="--"
        )
        gridlines.top_labels = False
        gridlines.right_labels = False

        colorbar = fig.colorbar(
            mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8, extend="max"
        )
        colorbar.set_label("Surface current speed (m s$^{-1}$)")
        colorbar.outline.set_visible(False)

        if title is not None:
            ax.set_title(title, fontweight="bold")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")

    return fig, ax
