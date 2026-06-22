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

from driftnet.data import degrade_coords

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
    ax: GeoAxes | Axes | None = None,
) -> tuple[Figure, GeoAxes]:
    """
    Plot velocity vectors, optionally clipped to a lon/lat box.
    """
    if u_input is None and v_input is None:
        raise TypeError("Received None for both u_input and v_input.")

    if stride < 1:
        raise ValueError("stride must be a positive integer.")

    projection = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = cast(GeoAxes, plt.axes(projection=projection))
    else:
        ax = cast(GeoAxes, ax)
        raw_fig = ax.figure

        # SubFigures have a '.figure' attribute pointing to their parent Figure.
        # We traverse up in case of nested SubFigures until we hit the root Figure.
        while not isinstance(raw_fig, Figure) and hasattr(raw_fig, "figure"):
            raw_fig = raw_fig.figure

        if not isinstance(raw_fig, Figure):
            raise TypeError("Could not resolve the root matplotlib Figure from the provided ax.")

        fig = raw_fig  # Type checker now knows this is strictly a Figure

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

    u_lon = degrade_coords(u_lon, res_lon, "u")
    v_lon = degrade_coords(v_lon, res_lon, "v")

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


def plot_cgrid_subset(coord_data, i_range=(0, 10), j_range=(0, 10)):
    """
    Plots a subset of the C-grid data.
    i_range and j_range define the slice to visualize.
    """

    # Slice the data
    u_lon = coord_data["u_lon"][j_range[0] : j_range[1], i_range[0] : i_range[1]]
    u_lat = coord_data["u_lat"][j_range[0] : j_range[1], i_range[0] : i_range[1]]

    v_lon = coord_data["v_lon"][j_range[0] : j_range[1], i_range[0] : i_range[1]]
    v_lat = coord_data["v_lat"][j_range[0] : j_range[1], i_range[0] : i_range[1]]

    rho_lon = coord_data["rho_lon"][j_range[0] : j_range[1], i_range[0] : i_range[1]]
    rho_lat = coord_data["rho_lat"][j_range[0] : j_range[1], i_range[0] : i_range[1]]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotting the points
    ax.scatter(rho_lon, rho_lat, color="blue", label="Rho points", marker="o", s=100, alpha=0.6)
    ax.scatter(u_lon, u_lat, color="red", label="U points", marker=">", s=100, alpha=0.6)
    ax.scatter(v_lon, v_lat, color="green", label="V points", marker="^", s=100, alpha=0.6)

    ax.set_title(f"C-Grid Visualization (Subset {i_range}, {j_range})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.savefig("images/cgrid.png")


def plot_velocity_heatmap(
    ax, lon, lat, vel, add_map_features=True, cmap="RdBu_r", vmin=None, vmax=None, corners=None
):
    """
    Plots a heatmap for velocity data on a given axis, optionally adding a zoomed-in Cartopy map.

    Parameters:
    - ax: matplotlib axis (must be a Cartopy GeoAxes if add_map_features is True)
    - lon: 2D numpy array of longitudes
    - lat: 2D numpy array of latitudes
    - vel: 2D numpy array of velocities (u or v)
    - add_map_features: bool, whether to add coastlines, land features, and zoom to the data extent
    - cmap: colormap to use (RdBu_r is good for diverging velocities where 0 is white)
    - vmin, vmax: limits for the colorbar (optional)
    - corners: Optional bounds for the map extent, handled via _get_extent

    Returns:
    - mesh: The QuadMesh object returned by pcolormesh (useful for adding a colorbar later)
    """

    if corners is None:
        corners = [40, 42.5, -20, -17.5]

    # Resolution matching
    nlat, nlon = vel.shape
    base_lat, base_lon = lon.shape

    res_lat = base_lat // nlat
    res_lon = base_lon // nlon

    lon = lon[::res_lon, ::res_lon]
    lat = lat[::res_lat, ::res_lat]

    # Calculate extent using the corners argument or data bounds
    # (Assuming _get_extent is available in this module's scope, just like in quiver)
    extent = _get_extent(lon.ravel(), lat.ravel(), corners=corners)

    if add_map_features:
        # Add geographical features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)

        # Zoom to the calculated extent based on corners/data
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Plot the data using PlateCarree projection so Cartopy knows how to map it
        mesh = ax.pcolormesh(
            lon,
            lat,
            vel,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading="auto",
            zorder=1,
            vmin=vmin,
            vmax=vmax,
        )

        # Optional: Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        # Standard matplotlib plot without Cartopy mapping
        mesh = ax.pcolormesh(lon, lat, vel, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)

        # Apply the extent to standard x/y limits: [lon_min, lon_max, lat_min, lat_max]
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    return mesh
