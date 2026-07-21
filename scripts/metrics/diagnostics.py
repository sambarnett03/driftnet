import polars as pl
from collections.abc import Sequence

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.metrics.ftle import add_cross_field_lyapunov
from driftnet.metrics.lagrange import get_connectivity_metrics
from driftnet.metrics.mse import calculate_velocity_mse
from driftnet.plotting import (
    plot_combined_experiment_trajectories,
    plot_multi_experiment_speed_heatmaps,
    plot_multi_experiment_trajectories,
    plot_several_experiments,
)
from driftnet.generated_types import ExperimentPathType

from driftnet.metrics.spectrum import write_spectra_to_csv
from driftnet.generated_types import ExperimentPathType




def save_metrics(
    data_config: DataConfig,
    exp_config: ExperimentConfig,
    exp_names: Sequence[ExperimentPathType]
):
    """
    Iterates over all specified experiments and saves their respective metrics.
    """
    for name in exp_names:
        print(f"\n{'='*50}\nComputing metrics for: {name}\n{'='*50}")

        # Extract exp_name and trial_name from the path string (e.g., "interpolate/100000particles")
        parts = str(name).split("/")
        e_name = parts[0]
        t_name = parts[1] if len(parts) > 1 else "baseline_trial"

        # Create a localized config for this specific experiment loop
        current_exp_config = ExperimentConfig(
            base=exp_config.base,
            exp_name=e_name,
            trial_name=t_name
        )

        # 1. Lagrangian Metrics
        print("Calculating Lagrangian connectivity metrics...")
        df_compare, df_agg = get_connectivity_metrics(current_exp_config)

        print("Calculating FTLE...")
        df_lyap, lyap_agg = add_cross_field_lyapunov(current_exp_config, df_compare, error_cols=["ML_Error_km"])

        # 2. Eulerian Metrics
        print("Calculating Eulerian velocity MSE...")
        calculate_velocity_mse(data_config, current_exp_config)

        # 3. Spectral Metrics
        print('Calculating FFT')
        write_spectra_to_csv(data_config, current_exp_config)

    print("\nAll metrics across all experiments successfully saved!")



def plot_metrics(data_config: DataConfig, exp_config: ExperimentConfig,
                 exp_names: Sequence[ExperimentPathType] | None = None):

    # Plot experiment results across all trials
    plot_several_experiments(exp_config, exp_names=exp_names,
                             metrics_to_plot=['euler_distance', 'ftle',
                                              'kinetic_energy_spectrum',
                                              'velocity_nmse'])

    # Plot trajectories
    # plot_multi_experiment_trajectories(exp_config, exp_names)
    # plot_combined_experiment_trajectories(exp_config, exp_names)

    # Plot heat map
    # plot_multi_experiment_speed_heatmaps(
    #     data_config, exp_config, corners=[40.0, 42.5, -20.0, -17.5],
    # )




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
    exp_names : Sequence[ExperimentPathType]
    exp_names = ['default_experiment/baseline_trial']

    plot_metrics(data_config, exp_config, exp_names=exp_names)
    print_final_metrics(exp_config)
