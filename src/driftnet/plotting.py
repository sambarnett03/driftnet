import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast, get_args
import math


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from numpy.typing import ArrayLike, NDArray
import matplotlib.colors as mcolors

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.generated_types import ExperimentPathType
from driftnet.utils import _get_valid_spatial_slices, extract_trajectories

Bounds = tuple[float, float, float, float]
BoundsLike = Sequence[float]
CornerPoint = tuple[float, float]
CornerPoints = Sequence[CornerPoint]
Corners = BoundsLike | CornerPoints
MetricType = Literal["euler_distance", "ftle", "velocity_mse"]


def _parse_corners(corners: Corners | None | str) -> Bounds | None | str:
    """Return lon/lat bounds from either bounds or four corner points."""
    if corners is None:
        return None

    if corners == "auto":
        return "auto"

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

    if corner_bounds == "auto":
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
    arr: NDArray[np.floating], row_stride: int, col_stride: int
) -> NDArray[np.floating]:
    """Helper to block-average a single 2D array with independent row/col strides."""
    if row_stride <= 1 and col_stride <= 1:
        return arr

    h, w = arr.shape
    h_trunc = h - (h % row_stride)
    w_trunc = w - (w % col_stride)
    arr_trunc = arr[:h_trunc, :w_trunc]

    return arr_trunc.reshape(
        h_trunc // row_stride, row_stride, w_trunc // col_stride, col_stride
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
    j_range: tuple[int, int] = (0, 10),
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
    grid_coords_path: Path,
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

    grid = np.load(grid_coords_path)
    lat = grid["rho_lat"]
    lon = grid["rho_lon"]

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
        gl = geo_ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
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
    data: NDArray[np.floating] | Sequence[NDArray[np.floating]] | None,
) -> list[NDArray[np.floating]]:
    """Helper to safely convert varying inputs into a strict list of 1D arrays."""
    if data is None:
        return []
    if isinstance(data, np.ndarray):
        return list(data) if data.ndim == 2 else [data]
    return list(data)


def _style_map_axis(ax: GeoAxes, extent: Sequence[float] | None = None) -> None:
    """
    Applies standard Cartopy geographic styling and bounds to an axis.
    """
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def _plot_single_panel(
    ax: GeoAxes, title: str, track_lons: Sequence[np.ndarray], track_lats: Sequence[np.ndarray]
):
    """
    Plots the background velocity field and overlays particle trajectories.
    """
    ax.set_title(title)

    # Plot trajectories
    for i, (lons, lats) in enumerate(zip(track_lons, track_lats, strict=False)):
        ax.plot(
            lons,
            lats,
            transform=ccrs.PlateCarree(),
            color="black",
            linewidth=2,
            label="Track" if i == 0 else None,
            zorder=3,
        )
        # Start marker
        ax.scatter(
            lons[0], lats[0], color="green", marker="o", transform=ccrs.PlateCarree(), zorder=4
        )
        # End marker
        ax.scatter(
            lons[-1], lats[-1], color="red", marker="X", transform=ccrs.PlateCarree(), zorder=4
        )

    legend_elements = [
        Line2D([0], [0], color="b", lw=4, label="Track"),
        Line2D([0], [0], marker="o", color="green", label="Start"),
        Line2D([0], [0], marker="X", color="red", label="End"),
    ]

    ax.legend(handles=legend_elements, loc="upper right")


# Define a literal type for static type checking safety
FieldType = Literal["truth", "predicted", "degraded"]


def plot_multi_experiment_trajectories(
    exp_config: ExperimentConfig,
    exp_names: Sequence[ExperimentPathType] | None = None,
    folder_name: str | Path = "comparison",
    padding: float = 2.0,
) -> None:
    """
    Dynamically plots side-by-side trajectory panels for ground truth and all specified experiments.
    """
    # 1. Resolve experiment names
    if exp_names is None:
        exp_names = get_args(ExperimentPathType)

    if not exp_names:
        raise ValueError("No experiments provided to plot.")

    all_lons_combined: list[NDArray] = []
    all_lats_combined: list[NDArray] = []
    processed_panels_data = []


    # 2. Extract Ground Truth
    # Assuming ground truth is identical across experiments, we pull it from the first one.
    truth_traj_path = Path(exp_config.base) / exp_names[0] / "metrics" / "trajectories_truth.zarr"

    if not truth_traj_path.exists():
        raise FileNotFoundError(f"Could not find Ground Truth trajectories at {truth_traj_path}")

    truth_trajectories = extract_trajectories(truth_traj_path)
    t_lons = _normalize_trajectories(truth_trajectories[0])
    t_lats = _normalize_trajectories(truth_trajectories[1])

    all_lons_combined.extend(t_lons)
    all_lats_combined.extend(t_lats)

    processed_panels_data.append({
        "title": "Ground Truth",
        "lons": t_lons,
        "lats": t_lats
    })

    # 3. Iterate over experiments and extract ML predictions dynamically
    for name in exp_names:
        exp_traj_path = Path(exp_config.base) / name / "metrics" / "trajectories_ml_predicted.zarr"

        if not exp_traj_path.exists():
            raise FileNotFoundError(f"Could not find predicted trajectories at {exp_traj_path}")

        trajectories = extract_trajectories(exp_traj_path)
        _lons = _normalize_trajectories(trajectories[0])
        _lats = _normalize_trajectories(trajectories[1])

        all_lons_combined.extend(_lons)
        all_lats_combined.extend(_lats)

        processed_panels_data.append({
            "title": f"Predicted: {name}",
            "lons": _lons,
            "lats": _lats
        })

    # 4. Global map bounds calculation
    extent = _get_extent(
        np.concatenate(all_lons_combined), np.concatenate(all_lats_combined), padding=padding
    )

    # 5. Dynamic Subplot Setup
    num_panels = len(processed_panels_data)
    fig, axes_raw = plt.subplots(
        1, num_panels, figsize=(8 * num_panels, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Clean array-vs-scalar unpacking for Pyright type checker safety
    axes = [cast(GeoAxes, axes_raw)] if num_panels == 1 else [cast(GeoAxes, ax) for ax in axes_raw]

    # 6. Step through layout allocations and populate panels
    for i, panel in enumerate(processed_panels_data):
        ax = axes[i]
        _style_map_axis(ax, extent)
        _plot_single_panel(
            ax=ax, title=panel["title"], track_lons=panel["lons"], track_lats=panel["lats"]
        )

    # 7. Save out image
    output_dir = Path("images") / str(folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "multi_experiment_trajectories.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Dynamic trajectory plot saved to {save_path}")



def plot_combined_experiment_trajectories(
    exp_config: ExperimentConfig,
    exp_names: Sequence[ExperimentPathType] | None = None,
    folder_name: str | Path = "comparison",
    padding: float = 2.0,
) -> None:
    """
    Plots the ground truth and ML predicted trajectories for all specified
    experiments overlaid on a single map.
    """
    # 1. Resolve experiment names
    if exp_names is None:
        exp_names = get_args(ExperimentPathType)

    if not exp_names:
        raise ValueError("No experiments provided to plot.")

    all_lons_combined: list[NDArray] = []
    all_lats_combined: list[NDArray] = []


    colours = list(mcolors.TABLEAU_COLORS.values())

    # Store trajectory data to plot later: {"label": str, "lons": NDArray, "lats": NDArray}
    trajectories_to_plot = []

    # 2. Extract Ground Truth (from the first experiment)
    truth_traj_path = Path(exp_config.base) / exp_names[0] / "metrics" / "trajectories_truth.zarr"

    if not truth_traj_path.exists():
        raise FileNotFoundError(f"Could not find Ground Truth trajectories at {truth_traj_path}")

    truth_trajectories = extract_trajectories(truth_traj_path)
    t_lons = _normalize_trajectories(truth_trajectories[0])
    t_lats = _normalize_trajectories(truth_trajectories[1])

    all_lons_combined.extend(t_lons)
    all_lats_combined.extend(t_lats)

    trajectories_to_plot.append({
        "label": "Ground Truth",
        "lons": t_lons,
        "lats": t_lats,
        "is_truth": True,
        "colour": 'black'
   })

    # 3. Iterate over experiments and extract ML predictions
    for i, name in enumerate(exp_names):
        exp_traj_path = Path(exp_config.base) / name / "metrics" / "trajectories_ml_predicted.zarr"

        if not exp_traj_path.exists():
            raise FileNotFoundError(f"Could not find predicted trajectories at {exp_traj_path}")

        trajectories = extract_trajectories(exp_traj_path)
        _lons = _normalize_trajectories(trajectories[0])
        _lats = _normalize_trajectories(trajectories[1])

        all_lons_combined.extend(_lons)
        all_lats_combined.extend(_lats)

        trajectories_to_plot.append({
            "label": f"Predicted: {name}",
            "lons": _lons,
            "lats": _lats,
            "is_truth": False,
            "colour": colours[i]
        })

    # 4. Global map bounds calculation
    extent = _get_extent(
        np.concatenate(all_lons_combined), np.concatenate(all_lats_combined), padding=padding
    )

    # 5. Single Map Setup
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax = cast(GeoAxes, ax)

    _style_map_axis(ax, extent)
    ax.set_title("Combined Trajectories Comparison", fontsize=14, pad=15)

    # 6. Plot all collected trajectories onto the single axis
    for item in trajectories_to_plot:
        track_lons = item["lons"]
        track_lats = item["lats"]
        label = item["label"]

        # Styling: Make ground truth stand out (e.g., thicker black line)
        if item["is_truth"]:
            linewidth = 2.5
            zorder = 5  # Ensure truth is drawn on top
            alpha = 1.0
            style = 'dashed'
        else:
            linewidth = 1.5
            zorder = 4
            alpha = 0.8
            style = 'solid'

        # Iterate through the list of arrays just like in _plot_single_panel
        for i, (lons, lats) in enumerate(zip(track_lons, track_lats, strict=False)):
            ax.plot(
                lons,
                lats,
                label=label if i == 0 else None,  # Only add label once per experiment
                color=item['colour'],
                linestyle=style,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                transform=ccrs.PlateCarree()
            )
            # Start marker
            ax.scatter(
                lons[0], lats[0], color="green", marker="o", transform=ccrs.PlateCarree(), zorder=zorder+1
            )
            # End marker
            ax.scatter(
                lons[-1], lats[-1], color="red", marker="X", transform=ccrs.PlateCarree(), zorder=zorder+1
            )

    # Add legend outside the plot to avoid covering trajectories
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False, title="Experiments")
    # 7. Save out image
    output_dir = Path("images") / str(folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "combined_experiment_trajectories.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Combined trajectory plot saved to {save_path}")



def plot_multi_experiment_speed_heatmaps(
    data_config: DataConfig,
    exp_config: ExperimentConfig,
    exp_names: Sequence[ExperimentPathType] | None = None,
    folder_name: str | Path = "comparison",
    time_idx: int = 0,
    corners: Any = None,
    padding: float = 2.0,
) -> None:
    """
    Dynamically plots speed heatmaps for Ground Truth and multiple experiments.
    Automatically calculates an optimal grid layout to scale nicely with the number of experiments.
    """
    # 1. Resolve experiment names
    if exp_names is None:
        exp_names = get_args(ExperimentPathType)

    if not exp_names:
        raise ValueError("No experiments provided to plot.")

    # Get datetime corresponding to time_idx from the ground truth dataset
    times = xr.open_zarr(exp_config.model_predictions).time_counter.values
    datetime_str = times[time_idx]

    # 2. Load base coordinates and extract spatial slices
    x_slice, y_slice = _get_valid_spatial_slices(data_config)
    base_lons = np.load(data_config.grid_params)["rho_lon"][y_slice, x_slice]
    base_lats = np.load(data_config.grid_params)["rho_lat"][y_slice, x_slice]

    all_lons_combined: list[np.ndarray] = []
    all_lats_combined: list[np.ndarray] = []
    processed_panels_data = []

    global_vmin = float("inf")
    global_vmax = float("-inf")

    # Helper function to process datasets
    def process_dataset(ds_path: Path | str, title: str, needs_slice: bool):
        nonlocal global_vmin, global_vmax

        ds = xr.open_zarr(ds_path)
        if needs_slice:
            ds = ds.isel(x=x_slice, y=y_slice)

        # Isolate the specific time step for the heatmap
        ds = ds.sel({"time_counter": datetime_str})

        # Extract U and V to calculate scalar speed
        u = ds["velocity"].isel(component=0)
        v = ds["velocity"].isel(component=1)
        speed = ((u**2 + v**2) ** 0.5).values

        all_lons_combined.append(base_lons.flatten())
        all_lats_combined.append(base_lats.flatten())

        # Track global min/max for a shared colorbar
        global_vmin = min(global_vmin, np.nanmin(speed))
        global_vmax = max(global_vmax, np.nanmax(speed))

        processed_panels_data.append(
            {"title": title, "lons": base_lons, "lats": base_lats, "speed": speed}
        )

    # 3. Process Ground Truth
    process_dataset(
        ds_path=data_config.original_res,
        title="Ground Truth",
        needs_slice=True
    )

    # 4. Process all requested ML Experiments
    # We infer the prediction filename from the exp_config to ensure we load the correct file
    pred_filename = Path(exp_config.model_predictions).name

    for name in exp_names:
        exp_pred_path = Path(exp_config.base) / name / pred_filename

        if not exp_pred_path.exists():
            raise FileNotFoundError(f"Could not find predicted dataset at {exp_pred_path}")

        process_dataset(
            ds_path=exp_pred_path,
            title=f"Predicted: {name}",
            needs_slice=False
        )

    # 5. Global map bounds calculation
    extent = _get_extent(
        np.concatenate(all_lons_combined),
        np.concatenate(all_lats_combined),
        padding=padding,
        corners=corners,
    )

    # 6. Dynamic Subplot Setup (Scales beautifully with panel count)
    num_panels = len(processed_panels_data)

    if num_panels == 1:
        nrows, ncols = 1, 1
    elif num_panels == 2:
        nrows, ncols = 1, 2
    elif num_panels == 3:
        nrows, ncols = 1, 3
    elif num_panels == 4:
        nrows, ncols = 2, 2
    else:
        # Generic fallback for larger numbers (e.g., 5 panels -> 2x3, 7 panels -> 3x3)
        ncols = math.ceil(math.sqrt(num_panels))
        nrows = math.ceil(num_panels / ncols)

    fig, axes_raw = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 8 * nrows), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Flatten safely for unified iteration
    axes_flat = np.atleast_1d(axes_raw).flatten()

    # 7. Step through layout allocations and populate panels
    mesh = None
    for i, ax_raw in enumerate(axes_flat):
        ax = cast(GeoAxes, ax_raw)

        # Turn off axes that don't have data (e.g., the 6th panel in a 2x3 grid with only 5 datasets)
        if i >= num_panels:
            ax.axis("off")
            continue

        panel = processed_panels_data[i]
        _style_map_axis(ax, extent)
        ax.set_title(panel["title"], fontsize=14, pad=10)

        # Plot the heatmap
        mesh = ax.pcolormesh(
            panel["lons"],
            panel["lats"],
            panel["speed"],
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            vmin=global_vmin,
            vmax=global_vmax,
            shading="auto",
        )

    # Add a shared colorbar across all subplots
    if mesh is not None:
        cbar = fig.colorbar(
            mesh, ax=axes_flat, orientation="horizontal", shrink=0.5, pad=0.08, aspect=40
        )
        cbar.set_label("Speed", fontsize=12)

    # 8. Save out image
    output_dir = Path("images") / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "zoomed_" if corners is not None else ""
    save_path = output_dir / f"{prefix}multi_exp_speed_heatmap_t{time_idx}.png"

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Dynamic speed heatmap saved to {save_path}")


def _plot_multi_experiment(
    data: dict,
    times: NDArray,
    metric_name: MetricType,
    folder_name: str | Path,
    file_name: str = "lagrangian_divergence_metrics",
):
    # os.makedirs(folder_name, exist_ok=True)
    # Initialize the plot with the same sizing as the reference
    plt.figure(figsize=(10, 6))

    # Iterate through the dictionary to plot each experiment
    # The dictionary keys will act as the labels for the legend
    for label, values in data.items():
        plt.plot(times, values, label=label, linewidth=2.5)

    # Apply standard styling, titles, and labels
    if metric_name == "euler_distance":
        plt.title("Lagrangian Separation Distance from Ground Truth")
        plt.xlabel("Advection Time (Hours)")
        plt.ylabel("Mean Separation Distance (km)")

    elif metric_name == "ftle":
        plt.title("FTLE from Ground Truth")
        plt.xlabel("Advection Time (Hours)")
        plt.ylabel("FTLE (km)")
        plt.yscale('log', base=np.e)

    elif metric_name == "velocity_mse":
        plt.title("MSE in speeds from Ground Truth")
        plt.xlabel("Advection Time (Hours)")
        plt.ylabel("Mean Square Error ($ms^{-1}$)")

    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    # Handle directory creation and saving the figure
    Path(f"images/{folder_name}").mkdir(parents=True, exist_ok=True)
    plot_path = f"images/{folder_name}/{metric_name}.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plot successfully saved to {plot_path}")

    # Close the plot to free up memory
    plt.close()


def plot_several_experiments(
    exp_config: ExperimentConfig,
    metrics_to_plot: Sequence[MetricType] | None = None,
    exp_names: Sequence[ExperimentPathType] | None = None,
    out_folder: Path = Path("comparison")
):

    if exp_names is None:
        exp_names = get_args(ExperimentPathType)

    if metrics_to_plot is None:
        metrics_to_plot = get_args(MetricType)

    metric_registry = {
        "euler_distance": {
            "csv_name": "distance.csv",
            "title": "Distance between particles",
            "heading": "Mean_ML_Error",
        },
        "ftle": {"csv_name": "ftle.csv", "title": "FTLE", "heading": "ML_Lyapunov_Exponent"},
        "velocity_mse": {
            "csv_name": "velocity_mse.csv",
            "title": "FTLE",
            "heading": "MSE_speed_ML",
        },
    }

    for metric in metrics_to_plot:
        if metric not in metric_registry:
            raise KeyError(f"metric {metric} not recognised - add to metric_registry")

        cfg = metric_registry[metric]

        data = {}
        time_hours = np.array([])

        for i, name in enumerate(exp_names):
            file_path = Path(exp_config.base) / name / "metrics" / cfg["csv_name"]

            if not file_path.exists():
                raise FileNotFoundError(f"Could not find results at {file_path}")

            df = pl.read_csv(file_path)

            if i == 0:  # calculate times (only need to do this once since all same)
                df = df.with_columns(pl.col("time").str.to_datetime(strict=False))

                time_hours = (df["time"] - df["time"][0]).dt.total_minutes() / 60.0

            data[name] = df[cfg["heading"]]

        _plot_multi_experiment(data, time_hours, metric, out_folder)
