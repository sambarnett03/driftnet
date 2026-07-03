import numpy as np
import polars as pl
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Explicit imports to help Pyright resolve the types statically
from parcels import (
    FieldSet,
    ParticleSet,
    ParticleFile,
    JITParticle,
    StatusCode,
    AdvectionRK4
)

from driftnet.config import DataConfig, ExperimentConfig


def haversine_distance(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Calculates the Haversine distance in kilometers between two sets of lon/lat points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def DeleteOutOfBounds(particles, fieldset):
    """OceanParcels V4 Kernel to delete particles that hit land or go out of bounds."""
    # V4 uses vectorized particle arrays, so we use boolean masking instead of if-statements
    out_of_bounds = particles.state >= 50
    particles[out_of_bounds].state = StatusCode.Delete


def prepare_fieldset(zarr_path: Path, grid_path: Path, time_coords: np.ndarray) -> FieldSet:
    """Wraps the custom velocity Zarr stores into an OceanParcels-compliant FieldSet."""
    # Load dataset and restrict to the exact test-set time coordinates
    ds = xr.open_zarr(zarr_path).sel(time_counter=time_coords)
    grid = np.load(grid_path)

    # Extract native C-grid coordinates for U and V components separately
    lon_u, lat_u = grid['u_lon'], grid['u_lat']
    lon_v, lat_v = grid['v_lon'], grid['v_lat']
    time = ds.time_counter

    # Create DataArrays explicitly linking to their respective C-grid faces
    U_da = xr.DataArray(
        ds.velocity.isel(component=0).data,
        coords={'time': time, 'lat_u': (('y', 'x'), lat_u), 'lon_u': (('y', 'x'), lon_u)},
        dims=['time', 'y', 'x'],
        name='U'
    )
    V_da = xr.DataArray(
        ds.velocity.isel(component=1).data,
        coords={'time': time, 'lat_v': (('y', 'x'), lat_v), 'lon_v': (('y', 'x'), lon_v)},
        dims=['time', 'y', 'x'],
        name='V'
    )

    parcels_ds = xr.Dataset({'U': U_da, 'V': V_da})

    # Instruct FieldSet to map each variable to its specific coordinate grid
    variables = {'U': 'U', 'V': 'V'}
    dimensions = {
        'U': {'lon': 'lon_u', 'lat': 'lat_u', 'time': 'time'},
        'V': {'lon': 'lon_v', 'lat': 'lat_v', 'time': 'time'}
    }

    # Initialize FieldSet
    return FieldSet.from_xarray_dataset(
        parcels_ds,
        variables,
        dimensions,
        mesh='spherical',
        allow_time_extrapolation=True
    )


def run_parcels_simulation(name: str, fieldset: FieldSet, start_lons: np.ndarray,
                           start_lats: np.ndarray, start_times: np.ndarray,
                           runtime: np.timedelta64, out_dir: Path) -> Path:
    """Executes the advection simulation for a given FieldSet."""
    print(f"Running Lagrangian advection for {name}...")

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=JITParticle,
        lon=start_lons,
        lat=start_lats,
        time=start_times
    )

    out_file = out_dir / f"trajectories_{name}.parquet"

    # Providing 'particleset=pset' to safely satisfy older Pyright type-stubs
    pfile = ParticleFile(str(out_file), particleset=pset, outputdt=np.timedelta64(1, 'h'))

    pset.execute(
        [AdvectionRK4, DeleteOutOfBounds],
        runtime=runtime,
        dt=np.timedelta64(15, 'm'), # Internal integration timestep
        output_file=pfile,
        verbose_progress=True
    )
    return out_file


def run_lagrangian_diagnostics(data_config: DataConfig, exp_config: ExperimentConfig):
    """Main entry point to evaluate ML upscaling using Lagrangian particle tracking."""

    out_dir = exp_config.base_path / "model_data"

    # 1. Isolate the exact time coordinates inferred by the ML model
    ds_pred = xr.open_zarr(exp_config.model_predictions)
    test_times = ds_pred.time_counter.values
    runtime = test_times[-1] - test_times[0]

    print(f"Test period isolated: {len(test_times)} steps. Total runtime: {runtime.astype('timedelta64[h]')} hours.")

    # 2. Prepare FieldSets
    print("Initializing OceanParcels FieldSets...")
    fs_truth = prepare_fieldset(data_config.original_res, data_config.grid_params_path, test_times)
    fs_pred = prepare_fieldset(exp_config.model_predictions, data_config.grid_params_path, test_times)
    fs_interp = prepare_fieldset(data_config.interpolated, data_config.grid_params_path, test_times)

    # 3. Choose starting locations (drop particles in valid open ocean, avoiding land)
    # Pulling directly from Zarr prevents Pyright dynamic-attribute errors on the FieldSet
    ds_truth = xr.open_zarr(data_config.original_res).sel(time_counter=test_times)
    u_initial = ds_truth.velocity.isel(component=0).data[0, :, :]
    grid = np.load(data_config.grid_params_path)

    # Valid ocean mask (where velocity is non-zero/not NaN)
    valid_mask = (u_initial != 0.0) & (~np.isnan(u_initial))
    y_idx, x_idx = np.where(valid_mask)

    # Subsample to track ~100 particles uniformly across the domain
    step = max(1, len(y_idx) // 100)

    # Even on a C-grid, we release the particles at the rho (cell center) locations
    start_lons = grid['rho_lon'][y_idx[::step], x_idx[::step]]
    start_lats = grid['rho_lat'][y_idx[::step], x_idx[::step]]
    start_times = np.repeat(test_times[0], len(start_lons))

    print(f"Dropping {len(start_lons)} particles into the domain...")

    # 4. Run simulations
    truth_file = run_parcels_simulation("truth", fs_truth, start_lons, start_lats, start_times, runtime, out_dir)
    pred_file = run_parcels_simulation("ml_predicted", fs_pred, start_lons, start_lats, start_times, runtime, out_dir)
    interp_file = run_parcels_simulation("interpolated", fs_interp, start_lons, start_lats, start_times, runtime, out_dir)

    # 5. Compute Connectivity Error (Separation Distance)
    print("Calculating divergence metrics...")
    df_truth = pl.read_parquet(truth_file).rename({"lon": "lon_t", "lat": "lat_t"})
    df_pred = pl.read_parquet(pred_file).rename({"lon": "lon_p", "lat": "lat_p"})
    df_interp = pl.read_parquet(interp_file).rename({"lon": "lon_i", "lat": "lat_i"})

    # Merge on particle ID and Time
    df_compare = df_truth.join(df_pred, on=["particle_id", "time"]).join(df_interp, on=["particle_id", "time"])

    # Calculate Haversine Distances
    ml_error = haversine_distance(
        df_compare["lon_t"].to_numpy(), df_compare["lat_t"].to_numpy(),
        df_compare["lon_p"].to_numpy(), df_compare["lat_p"].to_numpy()
    )
    interp_error = haversine_distance(
        df_compare["lon_t"].to_numpy(), df_compare["lat_t"].to_numpy(),
        df_compare["lon_i"].to_numpy(), df_compare["lat_i"].to_numpy()
    )

    df_compare = df_compare.with_columns([
        pl.Series("ML_Error_km", ml_error),
        pl.Series("Interp_Error_km", interp_error)
    ])

    # 6. Aggregate and Plot Results
    agg_df = df_compare.group_by("time").agg([
        pl.col("ML_Error_km").mean().alias("Mean_ML_Error"),
        pl.col("Interp_Error_km").mean().alias("Mean_Interp_Error")
    ]).sort("time")

    time_hours = (agg_df["time"] - agg_df["time"][0]).dt.total_minutes() / 60.0

    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, agg_df["Mean_ML_Error"], label="ML Super-Res", linewidth=2.5, color="red")
    plt.plot(time_hours, agg_df["Mean_Interp_Error"], label="Bilinear Interpolation", linewidth=2.5, color="blue", linestyle="--")
    plt.title("Lagrangian Separation Distance from Ground Truth")
    plt.xlabel("Advection Time (Hours)")
    plt.ylabel("Mean Separation Distance (km)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    plot_path = exp_config.base_path / "lagrangian_divergence_metrics.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plot successfully saved to {plot_path}")