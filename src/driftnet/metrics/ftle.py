from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from driftnet.config import ExperimentConfig
from driftnet.utils import append_mean_row


def _extract_burn_in_time(df: pl.DataFrame) -> Any | None:
    """
    Finds the second unique timestamp to use as the baseline
    for calculating trajectory error growth rates.
    """
    unique_times = df["time"].unique().sort()
    return unique_times[1] if len(unique_times) >= 2 else None


def _get_baseline_errors(
    df: pl.DataFrame, burn_in_time: Any, error_cols: list[str]
) -> pl.DataFrame:
    """
    Isolates the initial track separation values for each particle
    at the designated burn-in step.
    """
    select_exprs = [pl.col("trajectory")] + [
        pl.col(col).alias(f"{col}_baseline") for col in error_cols
    ]
    return df.filter(pl.col("time") == burn_in_time).select(select_exprs)


def _compute_growth_exponent_expr(
    current_error_col: str, baseline_col: str, delta_t_col: str
) -> pl.Expr:
    """
    Constructs the Polars expression to safely compute the exponential
    divergence rate, protecting against zero/negative divisions and logs.
    """
    return (
        pl.when((pl.col(delta_t_col) > 0) & (pl.col(baseline_col) > 0))
        .then(
            (1.0 / pl.col(delta_t_col)) * (pl.col(current_error_col) / pl.col(baseline_col)).log()
        )
        .otherwise(None)
    )


def add_cross_field_lyapunov(
    exp_config: ExperimentConfig, df_compare: pl.DataFrame, error_cols: list[str] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Main orchestrator that splits the cross-field Lyapunov calculation
    into modular data processing stages.
    """
    if error_cols is None:
        error_cols = ["ML_Error_km", "Interp_Error_km"]

    # 1. Resolve the burn-in reference step (t1)
    t1 = _extract_burn_in_time(df_compare)

    # 2. Extract baseline denominator boundaries for tracking pairs
    baseline_df = _get_baseline_errors(df_compare, t1, error_cols)

    # 3. Merge baseline context back into the main evaluation DataFrame
    df_lyap = df_compare.join(baseline_df, on="trajectory", how="left")

    # 4. Append elapsed delta time tracking (scaled in days)
    df_lyap = df_lyap.with_columns(
        ((pl.col("time") - t1).dt.total_minutes() / (60.0 * 24.0)).alias("delta_t_days")
    )

    # 5. Dynamically project exponents for all targeted error columns
    exponent_exprs = []
    for col in error_cols:
        # Generates a clean output name, e.g., 'ML_Lyapunov_Exponent'
        clean_prefix = col.replace("_Error_km", "")
        out_col_name = f"{clean_prefix}_Lyapunov_Exponent"

        expr = _compute_growth_exponent_expr(
            current_error_col=col, baseline_col=f"{col}_baseline", delta_t_col="delta_t_days"
        ).alias(out_col_name)

        exponent_exprs.append(expr)

    df_lyap = df_lyap.with_columns(exponent_exprs)

    agg_lyap_df = (
        df_lyap.group_by("time")
        .agg([pl.col("ML_Lyapunov_Exponent").mean(), pl.col("Interp_Lyapunov_Exponent").mean()])
        .sort("time")
    )

    append_mean_row(agg_lyap_df).write_csv(exp_config.metrics / "ftle.csv")

    return df_lyap, agg_lyap_df


def plot_ftle_results(agg_df: pl.DataFrame, folder_name: str | Path):

    time_hours = (agg_df["time"] - agg_df["time"][0]).dt.total_minutes() / 60.0

    plt.figure(figsize=(10, 6))
    plt.plot(
        time_hours, agg_df["ML_Lyapunov_Exponent"], label="ML Super-Res", linewidth=2.5, color="red"
    )
    plt.plot(
        time_hours,
        agg_df["Interp_Lyapunov_Exponent"],
        label="Bilinear Interpolation",
        linewidth=2.5,
        color="blue",
        linestyle="--",
    )
    plt.title("FTLE from Ground Truth")
    plt.xlabel("Advection Time (Hours)")
    plt.ylabel("FTLE (km)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    Path(f"images/{folder_name}").mkdir(parents=True, exist_ok=True)
    plot_path = f"images/{folder_name}/layp.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plot successfully saved to {plot_path}")
