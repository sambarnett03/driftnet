
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast, get_args

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
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.generated_types import ExperimentPathType
from driftnet.utils import _get_valid_spatial_slices, extract_trajectories
from driftnet.generated_types import MetricType


def plot_metrics(
    data_config: DataConfig,
    exp_config: ExperimentConfig,
    exp_names: Sequence[ExperimentPathType],
    metrics_to_plot: Sequence[MetricType],
    legend_labels: Sequence[str] | None = None
):
    # Plot experiment results across all trials
    plot_several_experiments(
        exp_config,
        exp_names=exp_names,
        metrics_to_plot=metrics_to_plot,
        legend_labels=legend_labels  # <--- Forwarding the new argument
    )



def _plot_multi_experiment(
    data: dict,
    x_values: np.ndarray,
    metric_name: MetricType,
    folder_name: str | Path,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Aesthetically pleasing colors (tab10)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Standard titles and labels
    labels_map = {
        "euler_distance": ("Lagrangian Separation Distance", "Advection Time (Hours)", "Mean Separation Distance (km)"),
        "ftle": ("Finite-Time Lyapunov Exponent (FTLE)", "Advection Time (Hours)", "FTLE (days$^{-1}$)"),
        "velocity_mse": ("Velocity Mean Square Error", "Advection Time (Hours)", "MSE ($m^2s^{-2}$)"),
        "velocity_nmse": ("Normalized Velocity MSE", "Advection Time (Hours)", "Normalized MSE"),
    }
    title, xlabel, ylabel = labels_map.get(
        metric_name, (metric_name.replace("_", " ").title(), "Time", "Value")
    )

    min_len = len(x_values)
    for values in data.values():
        if len(values) < min_len:
            min_len = len(values)

    # Plot each experiment
    color_idx = 0
    for label, values in data.items():
        if label == "Ground Truth":
            color = "black"
            linestyle = "--"
            linewidth = 3.0
            zorder = 10
        else:
            color = colors[color_idx % len(colors)]
            linestyle = "-"
            linewidth = 2.5
            zorder = 5
            color_idx += 1

        ax.plot(
            x_values[: min_len - 1],
            values[: min_len - 1],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.9,
            zorder=zorder
        )

    # Clean formatting
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    if metric_name == 'velocity_nmse':
        ax.set_ylim(bottom=0, top=0.0006)

    # Remove top/right borders and add a clean grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")

    ax.legend(fontsize=11, frameon=False, loc="best")

    # Save
    Path(f"images/{folder_name}").mkdir(parents=True, exist_ok=True)
    plot_path = Path(f"/home/users/sbarnett/documents/driftnet/images/{folder_name}/{metric_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plot successfully saved to {plot_path}")
    plt.close()


def _plot_kinetic_energy_spectrum(
    data: dict,
    x_values: np.ndarray,
    metric_name: MetricType,
    folder_name: str | Path,
):
    valid = x_values > 0
    k = x_values[valid]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    color_idx = 0
    for label, values in data.items():
        vals = np.array(values)[valid]
        if label == "Ground Truth":
            ax.loglog(k, vals, label=label, color="black", linewidth=3.0, zorder=10)
        else:
            c = colors[color_idx % len(colors)]
            ax.loglog(k, vals, label=label, color=c, linestyle="-", linewidth=2.0, alpha=0.85, zorder=5)
            color_idx += 1

    ax.set_title("Kinetic Energy Spectrum", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Wavenumber, $k$ (cycles / m)", fontsize=13)
    ax.set_ylabel("Kinetic Energy Density", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", linestyle="--", alpha=0.4, color="gray")

    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    def k_to_km(k_val):
        k_val = np.asarray(k_val, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(k_val == 0, np.inf, 1.0 / (k_val * 1000.0))

    def km_to_k(km_val):
        km_val = np.asarray(km_val, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(km_val == 0, np.inf, 1.0 / (km_val * 1000.0))

    secax = ax.secondary_xaxis("top", functions=(k_to_km, km_to_k))
    secax.set_xlabel("Wavelength (km)", fontsize=13)
    secax.set_xticks([10, 50, 100, 500])
    secax.xaxis.set_major_formatter(ScalarFormatter())
    secax.spines["top"].set_visible(False)

    plot_path = Path(f"/home/users/sbarnett/documents/driftnet/images/{folder_name}/{metric_name}.png")
    os.makedirs(plot_path.parent, exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plot successfully saved to {plot_path}")
    plt.close()


def _plot_distance_distribution(
    data: dict,
    x_values: np.ndarray,
    metric_name: MetricType,
    folder_name: str | Path,
):
    os.makedirs(f"images/{folder_name}", exist_ok=True)
    sample_df = next(iter(data.values()))
    time_cols = [c for c in sample_df.columns if c.startswith("dist_t_")]

    num_times = len(time_cols)
    fig, axes = plt.subplots(1, num_times, figsize=(6 * num_times, 5), sharey=False)
    if num_times == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, col in enumerate(time_cols):
        ax = axes[idx]
        t_idx = col.split("_")[-1]

        # 1. Pool all values for this time step to find a robust x-max
        all_vals_for_time = []
        for df in data.values():
            vals = df[col].drop_nulls().to_numpy()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                all_vals_for_time.extend(vals)

        if not all_vals_for_time:
            continue

        # Use the 98th percentile to cut off extreme runaway particles (long tails)
        max_val = np.percentile(all_vals_for_time, 98)

        if max_val == 0 or np.isnan(max_val):
            continue

        x_grid = np.linspace(0, max_val, 200)
        local_max_y = 0  # Track max density for THIS subplot only

        color_idx = 0
        for label, df in data.items():
            vals = df[col].drop_nulls().to_numpy()
            vals = vals[~np.isnan(vals)]

            if len(vals) < 2 or np.var(vals) == 0:
                continue

            # Calculate KDE
            kde = gaussian_kde(vals)
            kde_vals = kde(x_grid)

            c = colors[color_idx % len(colors)]

            ax.plot(x_grid, kde_vals, linewidth=2.5, label=label, color=c)
            ax.fill_between(x_grid, kde_vals, alpha=0.15, color=c)

            # Update local y-max
            local_max_y = max(local_max_y, np.max(kde_vals))
            color_idx += 1

        ax.set_title(f"Separation Variance (t={t_idx} hours)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Separation Distance (km)", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Density / Probability", fontsize=13)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.5, color="gray")

        # 2. Set axes limits independently for this subplot
        ax.set_xlim(0, max_val)
        if local_max_y > 0:
            ax.set_ylim(0, local_max_y * 1.15)

        if idx == num_times - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=11)

    plot_path = Path(f"/home/users/sbarnett/documents/driftnet/images/{folder_name}/{metric_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plot successfully saved to {plot_path}")
    plt.close()


def plot_several_experiments(
    exp_config: ExperimentConfig,
    metrics_to_plot: Sequence[MetricType] | None = None,
    exp_names: Sequence[ExperimentPathType] | None = None,
    legend_labels: Sequence[str] | None = None,
):
    if exp_names is None:
        exp_names = get_args(ExperimentPathType)
    if metrics_to_plot is None:
        metrics_to_plot = get_args(MetricType)

    if legend_labels is not None:
        if len(legend_labels) != len(exp_names):
            raise ValueError(f"legend_labels ({len(legend_labels)}) must match exp_names ({len(exp_names)}) length")
    else:
        legend_labels = [str(name) for name in exp_names]

    metric_registry = {
        "euler_distance": {"csv_name": "distance.csv", "heading": "Mean_ML_Error", "x_col": "time", "plot_fn": _plot_multi_experiment},
        "ftle": {"csv_name": "ftle.csv", "heading": "ML_Lyapunov_Exponent", "x_col": "time", "plot_fn": _plot_multi_experiment},
        "velocity_mse": {"csv_name": "velocity_mse.csv", "heading": "MSE_speed_ML", "x_col": "time", "plot_fn": _plot_multi_experiment},
        "velocity_nmse": {"csv_name": "velocity_mse.csv", "heading": "NMSE_speed_ML", "x_col": "time", "plot_fn": _plot_multi_experiment},
        "kinetic_energy_spectrum": {"csv_name": "spectrum.csv", "heading": "KE_pred", "truth_heading": "KE_truth", "x_col": "wavenumber", "plot_fn": _plot_kinetic_energy_spectrum},
        "distance_distribution": {"csv_name": "distance_distribution.csv", "heading": "ALL", "x_col": None, "plot_fn": _plot_distance_distribution},
    }

    for metric in metrics_to_plot:
        if metric not in metric_registry:
            raise KeyError(f"metric {metric} not recognised")

        cfg = metric_registry[metric]
        data = {}
        x_values = np.array([])

        for i, (name, label) in enumerate(zip(exp_names, legend_labels)):
            file_path = Path(exp_config.base) / name / "metrics" / cfg["csv_name"]
            if not file_path.exists():
                raise FileNotFoundError(f"Could not find results at {file_path}")

            df = pl.read_csv(file_path)

            if i == 0:
                if cfg.get("x_col") == "time":
                    df_time = df.with_columns(pl.col("time").str.to_datetime(strict=False))
                    x_values = ((df_time["time"] - df_time["time"][0]).dt.total_minutes() / 60.0).to_numpy()
                elif cfg.get("x_col") is not None:
                    x_values = df[cfg["x_col"]].to_numpy()

                if "truth_heading" in cfg and cfg["truth_heading"] in df.columns:
                    data["Ground Truth"] = df[cfg["truth_heading"]].to_numpy()

            if cfg.get("heading") == "ALL":
                data[label] = df
            else:
                data[label] = df[cfg["heading"]].to_numpy()

        plot_fn = cfg.get("plot_fn", _plot_multi_experiment)
        plot_fn(data, x_values, metric, Path("comparison"))