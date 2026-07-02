from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
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



def plot_side_by_side_trajectories(
    ds_gt: xr.Dataset,
    ds_pred: xr.Dataset,
    grid_coords_path: Path,
    lons_gt: np.ndarray,
    lats_gt: np.ndarray,
    lons_pred: np.ndarray,
    lats_pred: np.ndarray,
    time_index: int = 0,
    corners=None, # <--- NEW optional argument added
):
    """
    Plots GT and Predicted trajectories side-by-side over their respective velocity fields.
    Assumes lons/lats are 1D arrays of a single particle's history, or 2D arrays (particles, time_counter).
    Dynamically block-averages coordinate grids to match velocity field resolutions.
    """

    # Extract components and calculate magnitude
    u_gt = ds_gt['velocity'].isel(component=0, time_counter=time_index).values
    v_gt = ds_gt['velocity'].isel(component=1, time_counter=time_index).values
    mag_gt = np.sqrt(u_gt**2 + v_gt**2)

    u_pred = ds_pred['velocity'].isel(component=0, time_counter=time_index).values
    v_pred = ds_pred['velocity'].isel(component=1, time_counter=time_index).values
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)

    # Load base coordinates
    grid_data = np.load(grid_coords_path)
    base_lon = grid_data['rho_lon']
    base_lat = grid_data['rho_lat']
    base_h, base_w = base_lon.shape

    # Calculate strides and downsample coordinates for Ground Truth
    gt_h, gt_w = mag_gt.shape
    res_lat_gt = max(1, base_h // gt_h)
    res_lon_gt = max(1, base_w // gt_w)

    lon_gt = _block_average_2d(base_lon, res_lat_gt, res_lon_gt)
    lat_gt = _block_average_2d(base_lat, res_lat_gt, res_lon_gt)

    # Calculate strides and downsample coordinates for ML Prediction
    pred_h, pred_w = mag_pred.shape
    res_lat_pred = max(1, base_h // pred_h)
    res_lon_pred = max(1, base_w // pred_w)

    lon_pred = _block_average_2d(base_lon, res_lat_pred, res_lon_pred)
    lat_pred = _block_average_2d(base_lat, res_lat_pred, res_lon_pred)

    # --- CALCULATE EXTENT FOR ZOOM ---
    # Calculates a bounding box padded by 2 degrees around the GT trajectory,
    # or uses explicit corners if provided.
    extent = _get_extent(lons_gt, lats_gt, corners=corners, padding=2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    # Common plot settings
    for ax in axes:
        # Apply the zoom extent directly to the cartopy axes
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    # --- Panel 1: Ground Truth ---
    ax = axes[0]
    ax.set_title("Ground Truth: Velocity & Trajectories")

    # Plot background velocity magnitude using downsampled GT coordinates
    pcm1 = ax.pcolormesh(
        lon_gt,
        lat_gt,
        mag_gt,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        shading="auto"
    )

    # Plot GT trajectory
    if lons_gt.ndim == 1:
        ax.plot(
            lons_gt, lats_gt,
            transform=ccrs.PlateCarree(),
            color="white", linewidth=2, label="GT Track", zorder=3
        )
        ax.scatter(
            lons_gt[0], lats_gt[0],
            color="green", marker="o", transform=ccrs.PlateCarree(), label="Start", zorder=4,
        )
        ax.scatter(
            lons_gt[-1], lats_gt[-1],
            color="red", marker="X", transform=ccrs.PlateCarree(), label="End", zorder=4,
        )
    ax.legend(loc="upper right")

    # --- Panel 2: ML Prediction ---
    ax = axes[1]
    ax.set_title("ML Predicted: Velocity & Trajectories")

    # Plot background velocity magnitude using downsampled Pred coordinates
    ax.pcolormesh(
        lon_pred,
        lat_pred,
        mag_pred,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        shading="auto",
    )

    # Plot Pred trajectory
    if lons_pred.ndim == 1:
        ax.plot(
            lons_pred, lats_pred,
            transform=ccrs.PlateCarree(),
            color="white", linestyle="--", linewidth=2, label="Pred Track", zorder=3
        )
        ax.scatter(
            lons_pred[0], lats_pred[0],
            color="green", marker="o", transform=ccrs.PlateCarree(), zorder=4,
        )
        ax.scatter(
            lons_pred[-1], lats_pred[-1],
            color="red", marker="X", transform=ccrs.PlateCarree(), zorder=4,
        )
    ax.legend(loc="upper right")

    # Add a shared colorbar
    cbar = fig.colorbar(pcm1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Velocity Magnitude (m/s)")

    # Create output directory if it doesn't exist
    output_dir = Path('images')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'test.png', bbox_inches="tight")
    plt.close(fig)


def plot_overlapping_trajectories_on_neutral_map(
    lons_gt: np.ndarray,
    lats_gt: np.ndarray,
    lons_pred: np.ndarray,
    lats_pred: np.ndarray,
    extent: list,
):
    """
    Plots both sets of trajectories on a single map without a confusing velocity background.
    extent: [min_lon, max_lon, min_lat, max_lat]
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    ax = cast(GeoAxes, ax)

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
