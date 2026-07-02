import pickle

import numpy as np
import xarray as xr

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.diagnostics import evaluate_and_extract_paths
from driftnet.plotting import plot_side_by_side_trajectories


def run_lagrangian_diagnostics(config_data: DataConfig, config_pred: ExperimentConfig):

    ds_orig = xr.open_dataset(config_data.original_res)
    ds_pred = xr.open_dataset(config_pred.model_predictions)

    # 1. Run the simulation
    df_metrics, paths = evaluate_and_extract_paths(
        ds_orig,
        ds_pred,
        config_data.grid_params_path,
        seed_particles=[(-15, 60)],
        start_time=np.datetime64("1995-09-09T18:00:00"),
        baseline_duration_hours=48,
        safety_margin_hours=4.0,
    )

    print('Lagrangian simulation complete')

    # 2. Save your validation metrics to disk
    df_metrics.to_csv("results/ml_lagrangian_evaluation.csv", index=False)

    with open("results/trajectory_paths.pkl", "wb") as f:
        pickle.dump(paths, f)

    print('Successfully saved results of simulation')
    print('-' * 20)

    # 3. Plot the first particle's trajectory paths using the plotting functions from before
    plot_trajectories(config_data, config_pred)

    print('Plot generated')



def plot_trajectories(config_data: DataConfig, config_pred: ExperimentConfig):

    # 1. Load the pre-calculated paths
    print("Loading trajectory paths...")
    with open("results/trajectory_paths.pkl", "rb") as f:
        paths = pickle.load(f)

    # 2. Open the datasets (required by the plotting function to overlay backgrounds)
    print("Opening datasets...")
    ds_orig = xr.open_dataset(config_data.original_res)
    ds_pred = xr.open_dataset(config_pred.model_predictions)

    # 3. Experiment with your plotting!
    print("Generating plot...")
    plot_side_by_side_trajectories(
        ds_gt=ds_orig,
        ds_pred=ds_pred,
        grid_coords_path=config_data.grid_params_path,
        lons_gt=paths[0]["lons_gt"],
        lats_gt=paths[0]["lats_gt"],
        lons_pred=paths[0]["lons_pred"],
        lats_pred=paths[0]["lats_pred"],
    )
    print("Done!")
