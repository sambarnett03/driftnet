from pathlib import Path
import shutil

import numpy as np
import polars as pl

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.metrics.ftle import add_cross_field_lyapunov
from driftnet.metrics.lagrange import (
    _setup_experiment,
    _setup_fieldsets,
    _setup_test_particles,
    get_connectivity_metrics,
    run_parcels_simulation,
)
from driftnet.metrics.mse import calculate_velocity_mse
from driftnet.plotting import (
    plot_multi_experiment_trajectories,
    plot_multi_experiment_speed_heatmaps,
    plot_several_experiments,
    plot_combined_experiment_trajectories
)

def compute_trajectories(
    data_config: DataConfig,
    exp_config: ExperimentConfig,
    start_time: str | np.datetime64 | None = None,
    duration_days: int = 20,
):
    """Main entry point to evaluate ML upscaling using Lagrangian particle tracking."""

    # Setup times for experiment
    test_times, runtime, out_dir = _setup_experiment(exp_config, start_time, duration_days)

    # Prepare FieldSets
    fs_truth, fs_pred = _setup_fieldsets(data_config, exp_config, test_times)

    # Choose starting locations (drop particles in valid open ocean, avoiding land)
    start_lons, start_lats, start_times_array = _setup_test_particles(data_config, test_times)
    print(f"Dropping {len(start_lons)} particles into the domain...")

    # Run simulations
    shutil.copytree(
        '/gws/ssde/j25b/oxford_es/sbarnett/driftnet/experiments/default_experiment/baseline_trial/metrics/trajectories_truth.zarr',
        exp_config.metrics / 'trajectories_truth.zarr'
    )

    run_parcels_simulation(
        "ml_predicted", fs_pred, start_lons, start_lats, start_times_array, runtime, out_dir
    )


def save_metrics(data_config: DataConfig, exp_config: ExperimentConfig, _plot=False):
    df_compare, df_agg = get_connectivity_metrics(exp_config)

    df_lyap, lyap_agg = add_cross_field_lyapunov(
        exp_config, df_compare, error_cols=["ML_Error_km"]
    )

    calculate_velocity_mse(data_config, exp_config)


def plot_metrics(data_config: DataConfig, exp_config: ExperimentConfig):
    # Plot experiment results across all trials
    plot_several_experiments(exp_config)

    # Plot trajectories
    plot_multi_experiment_trajectories(exp_config)
    plot_combined_experiment_trajectories(exp_config)

    # Plot heat map
    plot_multi_experiment_speed_heatmaps(
        data_config,
        exp_config,
        corners=[40.0, 42.5, -20.0, -17.5]
    )


def print_final_metrics(exp_config):
    # 1. Read the CSVs
    mses = pl.read_csv(exp_config.metrics / "velocity_mse.csv")
    distance = pl.read_csv(exp_config.metrics / "distance.csv")
    ftle = pl.read_csv(exp_config.metrics / "ftle.csv")

    # 2. Extract the 'Average' row for each dataframe
    avg_mse_row = mses.filter(pl.col("time") == "Average")
    avg_dist_row = distance.filter(pl.col("time") == "Average")
    avg_ftle_row = ftle.filter(pl.col("time") == "Average")

    # 3. Pull the exact scalar values using .item()
    average_mse = avg_mse_row.select("MSE_speed_ML").item()

    # Assuming your distance column is named 'Mean_ML_Error' based on earlier code
    average_distance = avg_dist_row.select("Mean_ML_Error").item()

    # Assuming your FTLE column is named 'ML_Lyapunov_Exponent' based on earlier code
    average_ftle = avg_ftle_row.select("ML_Lyapunov_Exponent").item()

    # 4. Print the results
    print(f"Average MSE in velocity fields: {average_mse:.4f}")
    print(f"Average error in distance between pairs: {average_distance:.4f} km")
    print(f"Average FTLE between pairs: {average_ftle:.6f} days^-1")


def evaluate_predictions(data_config: DataConfig, exp_config: ExperimentConfig):
    # compute_trajectories(data_config, exp_config)
    # save_metrics(data_config, exp_config, _plot=True)
    plot_metrics(data_config, exp_config)
    print_final_metrics(exp_config)
