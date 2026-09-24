import xarray as xr
import numpy as np
from pathlib import Path
from driftnet.config import ExperimentConfig

def compute_ensemble_mean(
    exp_config: ExperimentConfig,
    trial_names: list[str],
    output_trial_name: str = "ensemble_mean"
) -> Path:
    """
    Loads the predictions from multiple trials within the same experiment,
    averages the velocity fields, and saves the result as a new trial.
    """
    base_path = Path(exp_config.base) / exp_config.exp_name

    datasets = []
    for trial in trial_names:
        trial_zarr_path = base_path / trial / "predictions.zarr"
        if not trial_zarr_path.exists():
            raise FileNotFoundError(f"Missing predictions for trial: {trial_zarr_path}")

        print(f"Queueing {trial}...")
        # Open lazily using Dask (doesn't load the full arrays into RAM)
        ds = xr.open_zarr(trial_zarr_path)
        datasets.append(ds)

    print(f"Concatenating {len(datasets)} trials...")
    # Combine along a new 'ensemble' dimension
    ds_combined = xr.concat(datasets, dim="ensemble")

    print("Calculating the ensemble mean computational graph...")
    # Compute the mean over the ensemble dimension
    ds_mean = ds_combined.mean(dim="ensemble")

    # Ensure we stay in float32 for storage/memory efficiency (mean() sometimes promotes to float64)
    ds_mean["velocity"] = ds_mean["velocity"].astype(np.float32)

    # Setup output directory structure to match standard trials
    output_dir = base_path / output_trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zarr_path = output_dir / "predictions.zarr"

    print(f"Executing Dask compute and saving to {output_zarr_path} (This may take a moment)...")

    # Re-apply chunking logic to match standard driftnet outputs
    ds_mean = ds_mean.chunk({"time_counter": 48, "component": -1, "y": -1, "x": -1})

    # Add some metadata for future reference
    ds_mean.attrs = {
        "description": f"Ensemble mean of {len(trial_names)} diffusion trials",
        "trials_used": ", ".join(trial_names)
    }

    # Trigger the Dask compute and write directly to disk in chunks
    ds_mean.to_zarr(output_zarr_path, mode="w")

    print(f"Ensemble mean successfully saved as trial: '{output_trial_name}'!")
    return output_zarr_path