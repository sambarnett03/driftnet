from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from numpy.typing import NDArray
from parcels import AdvectionRK4, FieldSet, JITParticle, ParticleFile, ParticleSet

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.utils import _get_valid_spatial_slices, append_mean_row


def haversine_distance(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Calculates the Haversine distance in kilometers between two sets of lon/lat points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def DeleteOutOfBounds(particle, fieldset, time):
    """OceanParcels V3 Kernel to delete particles that hit land or go out of bounds."""
    if particle.state >= 50:
        particle.delete()


def prepare_fieldset(
    zarr_path: Path,
    grid_path: Path,
    time_coords: np.ndarray,
    x_slice: slice | None = None,
    y_slice: slice | None = None,
) -> FieldSet:
    """Wraps the custom velocity Zarr stores into an OceanParcels-compliant FieldSet."""
    ds = xr.open_zarr(zarr_path).sel(time_counter=time_coords)
    grid = np.load(grid_path)

    # Cache the original full dimensions of the geographic grid
    full_ny, full_nx = grid["rho_lat"].shape

    # 1. ALWAYS slice the coordinate grid matrices to match the target trimmed size
    if y_slice is not None and x_slice is not None:
        rho_lon = grid["rho_lon"][y_slice, x_slice]
        rho_lat = grid["rho_lat"][y_slice, x_slice]
    else:
        rho_lon = grid["rho_lon"]
        rho_lat = grid["rho_lat"]

    # 2. ONLY slice ds if it's the untrimmed ground truth (matches full grid dimensions)
    if x_slice is not None and ds.sizes["x"] == full_nx:
        ds = ds.isel(x=x_slice)

    if y_slice is not None and ds.sizes["y"] == full_ny:
        ds = ds.isel(y=y_slice)

    # Calculate the local grid spacing for every cell (handles curvilinear grids gracefully)
    dlon = np.gradient(rho_lon, axis=1)
    dlat = np.gradient(rho_lat, axis=0)

    # Shift half an increment left (longitude) and down (latitude) to get corners
    lon_corner = rho_lon - (dlon / 2.0)
    lat_corner = rho_lat - (dlat / 2.0)

    time = ds.time_counter.values

    # Create DataArrays explicitly linking BOTH to the corner grid
    U_da = xr.DataArray(
        ds.velocity.isel(component=0).data,
        coords={"time": time, "lat": (("y", "x"), lat_corner), "lon": (("y", "x"), lon_corner)},
        dims=["time", "y", "x"],
        name="U",
    )
    V_da = xr.DataArray(
        ds.velocity.isel(component=1).data,
        coords={"time": time, "lat": (("y", "x"), lat_corner), "lon": (("y", "x"), lon_corner)},
        dims=["time", "y", "x"],
        name="V",
    )

    parcels_ds = xr.Dataset({"U": U_da, "V": V_da})

    variables = {"U": "U", "V": "V"}
    dimensions = {
        "U": {"lon": "lon", "lat": "lat", "time": "time"},
        "V": {"lon": "lon", "lat": "lat", "time": "time"},
    }

    fieldset = FieldSet.from_xarray_dataset(
        parcels_ds, variables, dimensions, mesh="spherical", allow_time_extrapolation=True
    )

    # Use getattr to bypass rigid static type checking for runtime attributes
    fieldset.U.interp_method = "cgrid_velocity"  # type: ignore
    fieldset.V.interp_method = "cgrid_velocity"  # type: ignore

    return fieldset


def run_parcels_simulation(
    name: str,
    fieldset: FieldSet,
    start_lons: np.ndarray,
    start_lats: np.ndarray,
    start_times: np.ndarray,
    runtime: np.timedelta64,
    out_dir: Path,
) -> Path:
    """Executes the advection simulation for a given FieldSet."""
    print(f"Running Lagrangian advection for {name}...")

    pset = ParticleSet(
        fieldset=fieldset, pclass=JITParticle, lon=start_lons, lat=start_lats, time=start_times
    )

    out_file = out_dir / f"trajectories_{name}.zarr"
    pfile = ParticleFile(str(out_file), particleset=pset, outputdt=np.timedelta64(1, "h"))

    pset.execute(
        [AdvectionRK4, DeleteOutOfBounds],
        runtime=runtime,
        dt=np.timedelta64(5, "m"),  # Internal integration timestep
        output_file=pfile,
        verbose_progress=True,
    )
    return out_file


def _setup_experiment(
    exp_config: ExperimentConfig,
    start_time: str | np.datetime64 | None = None,
    duration_days: int = 20,
):

    out_dir = exp_config.metrics

    ds_pred = xr.open_zarr(exp_config.model_predictions)
    all_test_times = ds_pred.time_counter.values

    # 1. Resolve the start and end time window strictly for Pyright
    actual_start_time: np.datetime64
    if start_time is None:
        actual_start_time = np.datetime64(all_test_times[0])
    else:
        actual_start_time = np.datetime64(start_time)

    end_time = actual_start_time + np.timedelta64(duration_days, "D")

    # Slice the temporal domain to exactly match the requested simulation duration
    time_mask = (all_test_times >= actual_start_time) & (all_test_times <= end_time)
    test_times = all_test_times[time_mask]

    if len(test_times) == 0:
        raise ValueError(
            f"No time steps found between {actual_start_time} and {end_time} in the ML predictions."
        )

    runtime = test_times[-1] - test_times[0]

    print(f"Simulation window: {actual_start_time} to {test_times[-1]}")
    print(
        f"Test period isolated: {len(test_times)} steps."
        f"Total runtime: {runtime.astype('timedelta64[h]')}."
    )

    return test_times, runtime, out_dir


def _setup_fieldsets(data_config: DataConfig, exp_config: ExperimentConfig, test_times: NDArray):
    x_slice, y_slice = _get_valid_spatial_slices(data_config)

    print("Initializing OceanParcels FieldSets...")
    fs_truth = prepare_fieldset(
        data_config.original_res, data_config.grid_params_path, test_times, x_slice, y_slice
    )
    fs_pred = prepare_fieldset(
        exp_config.model_predictions, data_config.grid_params_path, test_times, x_slice, y_slice
    )
    fs_interp = prepare_fieldset(
        data_config.interpolated, data_config.grid_params_path, test_times, x_slice, y_slice
    )

    return fs_truth, fs_pred, fs_interp


def _setup_test_particles(data_config: DataConfig, test_times: NDArray):
    x_slice, y_slice = _get_valid_spatial_slices(data_config)

    ds_orig_lazy = xr.open_zarr(data_config.original_res)
    ds_truth = ds_orig_lazy.sel(time_counter=test_times).isel(x=x_slice, y=y_slice)
    u_initial = ds_truth.velocity.isel(component=0, time_counter=0).values

    # Load and trim the geographic grid to match the dataset dimensions
    grid = np.load(data_config.grid_params_path)
    rho_lon = grid["rho_lon"][y_slice, x_slice]
    rho_lat = grid["rho_lat"][y_slice, x_slice]

    valid_mask = (u_initial != 0.0) & (~np.isnan(u_initial))
    y_idx, x_idx = np.where(valid_mask)

    # Subsample to track ~100 particles uniformly across the domain
    step = max(1, len(y_idx) // 10)

    # Release the particles at the correctly trimmed rho (cell center) locations
    start_lons = rho_lon[y_idx[::step], x_idx[::step]]
    start_lats = rho_lat[y_idx[::step], x_idx[::step]]
    start_times_array = np.repeat(test_times[0], len(start_lons))

    return start_lons, start_lats, start_times_array


def get_connectivity_metrics(exp_config: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Computes trajectory divergence metrics.
    Returns:
        df_compare: Raw DataFrame containing per-trajectory, per-timestep errors.
        agg_df: Aggregated DataFrame containing mean errors per timestep.
    """
    truth_file = exp_config.metrics / "trajectories_truth.zarr"
    pred_file = exp_config.metrics / "trajectories_ml.zarr"
    interp_file = exp_config.metrics / "trajectories_interpolated.zarr"

    # Open using Xarray and drop into Polars via Pandas
    ds_truth = xr.open_zarr(truth_file)
    df_truth = (
        pl.from_pandas(ds_truth.to_dataframe().reset_index())
        .rename({"lon": "lon_t", "lat": "lat_t"})
        .drop(["obs", "z"])
    )

    df_pred = xr.open_zarr(pred_file)
    df_pred = (
        pl.from_pandas(df_pred.to_dataframe().reset_index())
        .rename({"lon": "lon_p", "lat": "lat_p"})
        .drop(["obs", "z"])
    )

    df_interp = xr.open_zarr(interp_file)
    df_interp = (
        pl.from_pandas(df_interp.to_dataframe().reset_index())
        .rename({"lon": "lon_i", "lat": "lat_i"})
        .drop(["obs", "z"])
    )

    # Normalize trajectory IDs in case they drifted during simulation setup
    df_pred = df_pred.with_columns(
        (pl.col("trajectory") - pl.col("trajectory").min()).alias("trajectory")
    )
    df_interp = df_interp.with_columns(
        (pl.col("trajectory") - pl.col("trajectory").min()).alias("trajectory")
    )

    # Merge on particle ID and Time
    df_compare = df_truth.join(df_pred, on=["time", "trajectory"]).join(
        df_interp, on=["time", "trajectory"]
    )

    # Calculate Haversine Distances
    ml_error = haversine_distance(
        df_compare["lon_t"].to_numpy(),
        df_compare["lat_t"].to_numpy(),
        df_compare["lon_p"].to_numpy(),
        df_compare["lat_p"].to_numpy(),
    )
    interp_error = haversine_distance(
        df_compare["lon_t"].to_numpy(),
        df_compare["lat_t"].to_numpy(),
        df_compare["lon_i"].to_numpy(),
        df_compare["lat_i"].to_numpy(),
    )

    df_compare = df_compare.with_columns(
        [pl.Series("ML_Error_km", ml_error), pl.Series("Interp_Error_km", interp_error)]
    )

    # Create the aggregated dataframe for time-series plotting
    agg_df = (
        df_compare.group_by("time")
        .agg(
            [
                pl.col("ML_Error_km").mean().alias("Mean_ML_Error"),
                pl.col("Interp_Error_km").mean().alias("Mean_Interp_Error"),
            ]
        )
        .sort("time")
    )

    # Save the aggregated results
    Path("results").mkdir(parents=True, exist_ok=True)
    append_mean_row(agg_df).write_csv(exp_config.metrics / "distance.csv")

    # Return BOTH the granular trajectory data and the aggregated data
    return df_compare, agg_df


def plot_connectivity_results(
    agg_df: pl.DataFrame, folder_name: str | Path, file_name: str = "lagrangian_divergence_metrics"
):

    time_hours = (agg_df["time"] - agg_df["time"][0]).dt.total_minutes() / 60.0

    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, agg_df["Mean_ML_Error"], label="ML Super-Res", linewidth=2.5, color="red")
    plt.plot(
        time_hours,
        agg_df["Mean_Interp_Error"],
        label="Bilinear Interpolation",
        linewidth=2.5,
        color="blue",
        linestyle="--",
    )
    plt.title("Lagrangian Separation Distance from Ground Truth")
    plt.xlabel("Advection Time (Hours)")
    plt.ylabel("Mean Separation Distance (km)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    Path(f"images/{folder_name}").mkdir(parents=True, exist_ok=True)
    plot_path = f"images/{folder_name}/{file_name}.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plot successfully saved to {plot_path}")
