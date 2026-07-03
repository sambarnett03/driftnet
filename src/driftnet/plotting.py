from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from numpy.typing import ArrayLike, NDArray

from driftnet.data import degrade_coords

Bounds = tuple[float, float, float, float]
BoundsLike = Sequence[float]
CornerPoint = tuple[float, float]
CornerPoints = Sequence[CornerPoint]
Corners = BoundsLike | CornerPoints

def _parse_corners(corners: Corners | None | str) -> Bounds | None | str:
    """Return lon/lat bounds from either bounds or four corner points."""
    if corners is None:
        return None

    if corners == 'auto':
        return 'auto'

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
        "corners must be either 'auto', None, "
        "(lon_min, lon_max, lat_min, lat_max) or "
        "[(lon1, lat1), ..., (lon4, lat4)]."
    )

def _get_extent(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    corners: Corners | None | str = None,
    padding: float = 2.0,
) -> Bounds | None:
    """Return plot extent, using corners if supplied. Returns None for full map."""
    corner_bounds = _parse_corners(corners)

    if corner_bounds == 'auto':
        return (
            float(np.nanmin(lon) - padding),
            float(np.nanmax(lon) + padding),
            float(np.nanmin(lat) - padding),
            float(np.nanmax(lat) + padding),
        )

    # This will return either the explicit bounds tuple, or None
    return cast(Bounds | None, corner_bounds)


def _clip_to_extent(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    extent: Bounds | None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Clip coordinates and velocity components to the given extent."""
    if extent is None:
        return lon, lat, u, v

    lon_min, lon_max, lat_min, lat_max = extent

    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    return lon[mask], lat[mask], u[mask], v[mask]


def _block_average_2d(
    arr: NDArray[np.floating],
    row_stride: int,
    col_stride: int
) -> NDArray[np.floating]:
    """Helper to block-average a single 2D array with independent row/col strides."""
    if row_stride <= 1 and col_stride <= 1:
        return arr

    h, w = arr.shape
    h_trunc = h - (h % row_stride)
    w_trunc = w - (w % col_stride)
    arr_trunc = arr[:h_trunc, :w_trunc]

    return arr_trunc.reshape(
        h_trunc // row_stride,
        row_stride,
        w_trunc // col_stride,
        col_stride
    ).mean(axis=(1, 3))


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
    """
    Downsample coordinates and velocity components by block averaging.
    Uses fast NumPy reshaping to calculate the mean of non-overlapping blocks.
    """
    if stride <= 1:
        return lon, lat, u, v

    return (
        _block_average_2d(lon, stride, stride),
        _block_average_2d(lat, stride, stride),
        _block_average_2d(u, stride, stride),
        _block_average_2d(v, stride, stride),
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
    extent: Bounds | None,
    gridline_interval: float | None,
) -> None:
    """Add labelled gridlines, optionally at a fixed degree interval."""
    gridlines = ax.gridlines(draw_labels=True)

    if gridline_interval is None or extent is None:
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
    extent: Bounds | None,
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
    coord_data: dict[str, Any],
    u_input: ArrayLike | None = None,
    v_input: ArrayLike | None = None,
    corners: Corners | None | str = None,
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

        while not isinstance(raw_fig, Figure) and hasattr(raw_fig, "figure"):
            raw_fig = raw_fig.figure

        if not isinstance(raw_fig, Figure):
            raise TypeError("Could not resolve the root matplotlib Figure from the provided ax.")

        fig = raw_fig

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

    # Degrade the coordinate grids to match the input velocity resolution via block averaging
    u_lon = _block_average_2d(u_lon, res_lat, res_lon)
    v_lon = _block_average_2d(v_lon, res_lat, res_lon)
    u_lat = _block_average_2d(u_lat, res_lat, res_lon)
    v_lat = _block_average_2d(v_lat, res_lat, res_lon)

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

    if extent is not None:
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


def plot_cgrid_subset(
    coord_data: dict[str, Any],
    i_range: tuple[int, int] = (0, 10),
    j_range: tuple[int, int] = (0, 10)
) -> None:
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
    ax: GeoAxes | Axes,
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    vel: NDArray[np.floating],
    add_map_features: bool = True,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    corners: Corners | None | str = None,
) -> QuadMesh:
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
        corners = [40.0, 42.5, -20.0, -17.5]

    # Resolution matching
    nlat, nlon = vel.shape
    base_lat, base_lon = lon.shape

    res_lat = base_lat // nlat
    res_lon = base_lon // nlon

    lon = _block_average_2d(lon, res_lat, res_lon)
    lat = _block_average_2d(lat, res_lat, res_lon)

    # Calculate extent using the corners argument or data bounds
    extent = _get_extent(lon.ravel(), lat.ravel(), corners=corners)

    if add_map_features:
        # Cast to GeoAxes so Pylance knows Cartopy methods are available
        geo_ax = cast(GeoAxes, ax)

        # Add geographical features
        geo_ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)

        # Zoom to the calculated extent based on corners/data
        if extent is not None:
            geo_ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Plot the data using PlateCarree projection so Cartopy knows how to map it
        mesh = geo_ax.pcolormesh(
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
        gl = geo_ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        # Standard matplotlib plot without Cartopy mapping
        mesh = ax.pcolormesh(lon, lat, vel, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)

        # Apply the extent to standard x/y limits: [lon_min, lon_max, lat_min, lat_max]
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

    return mesh


def _normalize_trajectories(
    data: NDArray[np.floating] | Sequence[NDArray[np.floating]] | None
) -> list[NDArray[np.floating]]:
    """Helper to safely convert varying inputs into a strict list of 1D arrays."""
    if data is None:
        return []
    if isinstance(data, np.ndarray):
        return list(data) if data.ndim == 2 else [data]
    return list(data)


def plot_side_by_side_trajectories(
    ds_gt: xr.Dataset,
    ds_pred: xr.Dataset,
    grid_coords_path: Path | str,
    lons_gt: NDArray[np.floating] | Sequence[NDArray[np.floating]],
    lats_gt: NDArray[np.floating] | Sequence[NDArray[np.floating]],
    lons_pred: NDArray[np.floating] | Sequence[NDArray[np.floating]],
    lats_pred: NDArray[np.floating] | Sequence[NDArray[np.floating]],
    ds_interp: xr.Dataset | None = None,
    lons_interp: NDArray[np.floating] | Sequence[NDArray[np.floating]] | None = None,
    lats_interp: NDArray[np.floating] | Sequence[NDArray[np.floating]] | None = None,
    time_index: int = 0,
    corners: Corners | None | str = None,
    padding: float = 2.0,
    save_name: str = 'test.png'
) -> None:
    """
    Plots GT, Predicted, and optionally Interpolated trajectories side-by-side.
    """
    # Load base coordinates
    grid_data = np.load(grid_coords_path)
    base_lon = grid_data['rho_lon']
    base_lat = grid_data['rho_lat']
    base_h, base_w = base_lon.shape

    # Helper function to extract and downsample a dataset
    def _prepare_ds(ds: xr.Dataset):
        u = ds['velocity'].isel(component=0, time_counter=time_index).values
        v = ds['velocity'].isel(component=1, time_counter=time_index).values
        mag = np.sqrt(u**2 + v**2)

        h, w = mag.shape
        res_lat = max(1, base_h // h)
        res_lon = max(1, base_w // w)

        lon_d = _block_average_2d(base_lon, res_lat, res_lon)
        lat_d = _block_average_2d(base_lat, res_lat, res_lon)
        return mag, lon_d, lat_d

    # Prepare datasets
    mag_gt, lon_gt, lat_gt = _prepare_ds(ds_gt)
    mag_pred, lon_pred, lat_pred = _prepare_ds(ds_pred)

    # Initialize interpolation variables so they are always bound
    mag_interp = lon_interp = lat_interp = None
    if ds_interp is not None:
        mag_interp, lon_interp, lat_interp = _prepare_ds(ds_interp)

    # Normalize trajectory inputs to strict lists (now guaranteed to never be None)
    _lons_gt = _normalize_trajectories(lons_gt)
    _lats_gt = _normalize_trajectories(lats_gt)
    _lons_pred = _normalize_trajectories(lons_pred)
    _lats_pred = _normalize_trajectories(lats_pred)
    _lons_interp = _normalize_trajectories(lons_interp)
    _lats_interp = _normalize_trajectories(lats_interp)

    # --- CALCULATE MAP EXTENT ---
    # Safe list addition, even if _lons_interp is an empty list []
    all_lons = _lons_gt + _lons_pred + _lons_interp
    all_lats = _lats_gt + _lats_pred + _lats_interp

    extent = _get_extent(np.concatenate(all_lons), np.concatenate(all_lats), corners=corners, padding=padding)

    # Dynamically scale width based on number of panels
    has_interp = ds_interp is not None and len(_lons_interp) > 0 and len(_lats_interp) > 0
    num_panels = 3 if has_interp else 2

    fig, axes_raw = plt.subplots(1, num_panels, figsize=(8 * num_panels, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = [cast(GeoAxes, ax) for ax in (axes_raw if num_panels > 1 else [axes_raw])]

    # Apply common styling
    for ax in axes:
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    # --- Panel 1: Ground Truth ---
    ax0 = axes[0]
    ax0.set_title("Ground Truth")
    pcm = ax0.pcolormesh(lon_gt, lat_gt, mag_gt, transform=ccrs.PlateCarree(), cmap="viridis", shading="auto")
    for i, (lons, lats) in enumerate(zip(_lons_gt, _lats_gt)):
        ax0.plot(lons, lats, transform=ccrs.PlateCarree(), color="white", linewidth=2, label="Track" if i == 0 else None, zorder=3)
        ax0.scatter(lons[0], lats[0], color="green", marker="o", transform=ccrs.PlateCarree(), zorder=4)
        ax0.scatter(lons[-1], lats[-1], color="red", marker="X", transform=ccrs.PlateCarree(), zorder=4)
    ax0.legend(loc="upper right")

    # --- Panel 2: ML Prediction ---
    ax1 = axes[1]
    ax1.set_title("ML Predicted")
    ax1.pcolormesh(lon_pred, lat_pred, mag_pred, transform=ccrs.PlateCarree(), cmap="viridis", shading="auto")
    for i, (lons, lats) in enumerate(zip(_lons_pred, _lats_pred)):
        ax1.plot(lons, lats, transform=ccrs.PlateCarree(), color="white", linewidth=2, label="Track" if i == 0 else None, zorder=3)
        ax1.scatter(lons[0], lats[0], color="green", marker="o", transform=ccrs.PlateCarree(), zorder=4)
        ax1.scatter(lons[-1], lats[-1], color="red", marker="X", transform=ccrs.PlateCarree(), zorder=4)
    ax1.legend(loc="upper right")

    # --- Panel 3: Interpolated (Optional) ---
    if has_interp and lon_interp is not None and lat_interp is not None and mag_interp is not None:
        ax2 = axes[2]
        ax2.set_title("Bilinear Interpolation")
        ax2.pcolormesh(lon_interp, lat_interp, mag_interp, transform=ccrs.PlateCarree(), cmap="viridis", shading="auto")
        for i, (lons, lats) in enumerate(zip(_lons_interp, _lats_interp)):
            ax2.plot(lons, lats, transform=ccrs.PlateCarree(), color="white", linewidth=2, label="Track" if i == 0 else None, zorder=3)
            ax2.scatter(lons[0], lats[0], color="green", marker="o", transform=ccrs.PlateCarree(), zorder=4)
            ax2.scatter(lons[-1], lats[-1], color="red", marker="X", transform=ccrs.PlateCarree(), zorder=4)
        ax2.legend(loc="upper right")

    cbar = fig.colorbar(pcm, ax=axes_raw, orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Velocity Magnitude (m/s)")

    output_dir = Path('images')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / save_name, bbox_inches="tight")
    plt.close(fig)


def plot_overlapping_trajectories_on_neutral_map(
    lons_gt: NDArray[np.floating],
    lats_gt: NDArray[np.floating],
    lons_pred: NDArray[np.floating],
    lats_pred: NDArray[np.floating],
    extent: Bounds,
) -> None:
    """
    Plots both sets of trajectories on a single map without a confusing velocity background.
    extent: [min_lon, max_lon, min_lat, max_lat]
    """
    fig, ax_raw = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    ax = cast(GeoAxes, ax_raw)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="azure")
    ax.add_feature(cfeature.LAND, facecolor="tan", edgecolor="black", zorder=2)

    gl = ax.gridlines(draw_labels=True, linestyle=":", alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False

    # Plot Tracks
    ax.plot(
        lons_gt,
        lats_gt,
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=2.5,
        label="Ground Truth",
    )
    ax.plot(
        lons_pred,
        lats_pred,
        transform=ccrs.PlateCarree(),
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="ML Predicted",
    )

    # Mark Start point
    ax.scatter(
        lons_gt[0],
        lats_gt[0],
        color="green",
        marker="o",
        s=100,
        transform=ccrs.PlateCarree(),
        label="Start Point",
        zorder=4,
    )

    ax.set_title("Trajectory Comparison: GT vs ML Prediction")
    ax.legend(loc="upper right")

    plt.show()