import numpy as np
import xarray as xr

from driftnet.config import DataConfig
from driftnet.diagnostics import evaluate_and_extract_paths
from driftnet.plotting import plot_side_by_side_trajectories


def run_lagrangian_diagnostics(config: DataConfig):

    ds_orig = xr.open_dataset(config.original_res)
    ds_pred = xr.open_dataset(config.degraded_res)

    # 1. Run the simulation
    df_metrics, paths = evaluate_and_extract_paths(
        ds_orig,
        ds_pred,
        seed_particles=[(32.5, -64.8)],
        start_time=np.datetime64("2026-06-01T00:00:00"),
        baseline_duration_hours=48.0,
        safety_margin_hours=12.0,
    )

    # 2. Save your validation metrics to disk
    df_metrics.to_csv("ml_lagrangian_evaluation.csv", index=False)

    # 3. Plot the first particle's trajectory paths using the plotting functions from before
    plot_side_by_side_trajectories(
        ds_gt=ds_orig,
        ds_pred=ds_pred,
        lons_gt=paths[0]["lons_gt"],
        lats_gt=paths[0]["lats_gt"],
        lons_pred=paths[0]["lons_pred"],
        lats_pred=paths[0]["lats_pred"],
    )
