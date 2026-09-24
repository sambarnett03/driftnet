import argparse
import random
from pathlib import Path

import numpy as np
import torch

from driftnet.config import MasterConfig, ExperimentConfig
from metrics.diagnostics import save_metrics
from figures.figure_plotting import plot_metrics
from ml.inference_script import inference_over_test_set


def set_seeds(seed: int):
    """Ensures deterministic noise generation across PyTorch and Numpy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Generate Diffusion Ensemble via Seeds")
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--num_members", type=int, default=10)
    args = parser.parse_args()

    config = MasterConfig.load_from_yaml(args.config)

    # 1. Locate the pre-trained weights we want to reuse
    source_weights = Path(config.experiment.base) / "diffusioncfg" / "trial1" / "model_data" / "best_weights.pth"
    if not source_weights.exists():
        raise FileNotFoundError(f"Source weights not found at {source_weights}")

    exp_names = []
    nice_labels = []

    # =========================================================================
    # PART 1: GENERATE ENSEMBLE PREDICTIONS
    # =========================================================================
    print(f"--- Generating {args.num_members} Ensemble Members ---")

    for i in range(args.num_members):
        trial_name = f"ensemble_member_{i}"
        exp_names.append(f"diffusioncfg/{trial_name}")
        nice_labels.append(f"Member {i}")

        # Create an isolated config mapping to this member's specific folder
        member_config = ExperimentConfig(
            exp_name="diffusioncfg",
            trial_name=trial_name,
            base=config.experiment.base,
            guidance_scale=config.experiment.guidance_scale
        )

        # Force the config to load the weights from trial1!
        member_config.model_weights = source_weights

        # Skip logic: Don't regenerate if the Zarr store is already finished
        if member_config.model_predictions.exists():
            print(f"[{trial_name}] Predictions already exist. Skipping generation.")
        else:
            print(f"\n[{trial_name}] Generating predictions with Seed {42 + i}...")
            set_seeds(42 + i)
            inference_over_test_set(config.data, config.hyperparameters, member_config)

    # =========================================================================
    # PART 2: COMPUTE KINETIC ENERGY SPECTRUM
    # =========================================================================
    # Add the interpolation baseline for comparison
    baseline_name = "interpolate/baseline_trial"
    exp_names.append(baseline_name)
    nice_labels.append("Linear Interpolation (Baseline)")

    print("\n--- Computing Kinetic Energy Spectra ---")

    # Only run FFTs for members that haven't been computed yet
    metrics_to_run = []
    for exp in exp_names:
        spectrum_csv = Path(config.experiment.base) / exp / "metrics" / "spectrum.csv"
        if not spectrum_csv.exists():
            metrics_to_run.append(exp)

    if metrics_to_run:
        print(f"Calculating spectra for {len(metrics_to_run)} trials...")
        save_metrics(
            config.data,
            config.experiment,
            exp_names=metrics_to_run,
            metrics_to_calc=["kinetic_energy_spectrum"]
        )
    else:
        print("All spectra already computed! Skipping calculations.")

    # =========================================================================
    # PART 3: PLOT THE RESULTS
    # =========================================================================
    print("\n--- Plotting Ensemble Kinetic Energy Spectra ---")
    plot_metrics(
        data_config=config.data,
        exp_config=config.experiment,
        exp_names=exp_names,
        metrics_to_plot=["kinetic_energy_spectrum"],
        legend_labels=nice_labels
    )


if __name__ == "__main__":
    main()