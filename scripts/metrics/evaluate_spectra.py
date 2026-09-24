import argparse
from pathlib import Path

import numpy as np
import polars as pl

from driftnet.config import MasterConfig


def evaluate_spectra(config: MasterConfig, exp_names: list[str], labels: list[str]):
    base_path = Path(config.experiment.base)

    # 1. Find the Ground Truth spectrum from the first valid experiment
    truth_k = None
    truth_ke = None

    for name in exp_names:
        csv_path = base_path / name / "metrics" / "spectrum.csv"
        if csv_path.exists():
            df = pl.read_csv(csv_path)
            if "KE_truth" in df.columns:
                # Filter out k=0 (mean flow) and NaNs
                valid_mask = (df["wavenumber"].to_numpy() > 0) & (~np.isnan(df["KE_truth"].to_numpy()))
                truth_k = df["wavenumber"].to_numpy()[valid_mask]
                truth_ke = df["KE_truth"].to_numpy()[valid_mask]
                break

    if truth_ke is None:
        print("Error: Could not find 'KE_truth' in any of the provided spectrum.csv files.")
        return

    # Print Table Header
    print(f"\n{'Experiment':<35} | {'Log R-Squared (R²)':<18} | {'Log RMSE':<10}")
    print("-" * 68)

    # 2. Iterate over all experiments and calculate metrics
    for name, label in zip(exp_names, labels):
        csv_path = base_path / name / "metrics" / "spectrum.csv"
        if not csv_path.exists():
            print(f"{label:<35} | {'Missing CSV':<18} | {'N/A':<10}")
            continue

        df = pl.read_csv(csv_path)

        valid_mask = (df["wavenumber"].to_numpy() > 0) & (~np.isnan(df["KE_pred"].to_numpy()))
        pred_k = df["wavenumber"].to_numpy()[valid_mask]
        pred_ke = df["KE_pred"].to_numpy()[valid_mask]

        # Intersect wavenumbers in case the low-res baseline has fewer points
        common_k, truth_indices, pred_indices = np.intersect1d( # type: ignore
            truth_k, pred_k, return_indices=True # type: ignore
        )

        if len(common_k) == 0:
            print(f"{label:<35} | {'No common wavenumbers':<18} | {'N/A':<10}")
            continue

        # Extract matching spectra
        y_true = truth_ke[truth_indices]
        y_pred = pred_ke[pred_indices]

        # Ensure strictly positive values for log10 conversion
        nonzero = (y_true > 0) & (y_pred > 0)
        y_true = y_true[nonzero]
        y_pred = y_pred[nonzero]

        # 3. Convert to Log Space
        log_true = np.log10(y_true)
        log_pred = np.log10(y_pred)

        # 4. Calculate R-squared (Coefficient of Determination)
        ss_res = np.sum((log_true - log_pred) ** 2)
        ss_tot = np.sum((log_true - np.mean(log_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # 5. Calculate Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((log_true - log_pred) ** 2))

        print(f"{label:<35} | {r_squared:<18.4f} | {rmse:<10.4f}")

    print("\nNote: R² and RMSE are calculated on log10(KE) to equally weight high-frequency turbulence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Spectral Metrics")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    config = MasterConfig.load_from_yaml(args.config)

    # Use the same names and labels from your plotting script
    exp_names = ['interpolate/baseline_trial', 'resblock/l1',  'diffusioncfg/trial1', 'diffusioncfg/ensemble_mean']

    nice_labels = [
        "Linear Interpolation (Baseline)",
        "U-Net",
        "Diffusion Single Trial",
        "Diffusion Ensemble Mean"
    ]

    evaluate_spectra(config, exp_names, nice_labels)