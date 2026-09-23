from dask.base import compute
import polars as pl
import xarray as xr

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.utils import _get_valid_spatial_slices, append_mean_row


def _extract_velocity_components(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Extracts the U and V components and calculates the scalar speed.
    Leverages Dask lazy arrays under the hood.
    """
    # Use isel to safely grab component 0 (u) and 1 (v)
    u = ds["velocity"].isel(component=0)
    v = ds["velocity"].isel(component=1)

    # Lazy computation of velocity magnitude (speed)
    speed = (u**2 + v**2) ** 0.5

    return u, v, speed


def _load_and_align_ground_truth(data_config: DataConfig, eval_times: xr.DataArray) -> xr.Dataset:
    """
    Loads the ground truth zarr, slices it spatially to match the domain,
    and slices it temporally to match the model evaluation period.
    """
    ds_gt = xr.open_zarr(data_config.original_res)

    # Spatial clipping
    x_slice, y_slice = _get_valid_spatial_slices(data_config)
    ds_gt = ds_gt.isel(x=x_slice, y=y_slice)

    # Temporal clipping (align to the exact timestamps in the ML dataset)
    ds_gt = ds_gt.sel(time_counter=eval_times)

    return ds_gt


def _calculate_spatial_errors(pred_da: xr.DataArray, truth_da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Builds the Dask graph to compute the Mean Squared Error and Normalized MSE.
    Returns two 1D DataArrays indexed by time_counter.
    """
    # 1. Standard MSE
    squared_error = (pred_da - truth_da) ** 2
    mse = squared_error.mean(dim=["x", "y"])

    # 2. Normalized MSE (NMSE)
    # We square the truth data first to find the max energy/variance.
    # This automatically handles negative velocity values natively and matches the MSE units (m^2/s^2).
    max_val_sq = (truth_da ** 2).max(dim=["x", "y"])

    # Safe division: Replace 0 with a tiny epsilon to prevent division-by-zero errors
    max_val_sq = xr.where(max_val_sq == 0, 1e-10, max_val_sq)

    nmse = mse / max_val_sq

    return mse, nmse


def calculate_velocity_mse(data_config: DataConfig, exp_config: ExperimentConfig) -> pl.DataFrame:
    """
    Main orchestrator to calculate U, V, and Speed MSE & NMSE between Ground Truth,
    ML Predicted, and Bilinear Interpolated fields.
    """
    print("Loading datasets and building MSE & NMSE compute graph...")

    # 1. Load predicted dataset
    ds_pred = xr.open_zarr(exp_config.model_predictions)

    # 2. Get the evaluation timestamps to align the Ground Truth
    eval_times = ds_pred["time_counter"]
    ds_gt = _load_and_align_ground_truth(data_config, eval_times)

    # 3. Extract u, v, and speed components for all fields
    u_gt, v_gt, speed_gt = _extract_velocity_components(ds_gt)
    u_ml, v_ml, speed_ml = _extract_velocity_components(ds_pred)

    # 4. Build the lazy MSE & NMSE graphs
    mse_u, nmse_u = _calculate_spatial_errors(u_ml, u_gt)
    mse_v, nmse_v = _calculate_spatial_errors(v_ml, v_gt)
    mse_s, nmse_s = _calculate_spatial_errors(speed_ml, speed_gt)

    print("Executing Dask compute (this may take a moment)...")

    # 5. Trigger Dask to process ALL graphs in a single optimized pass
    (
        mse_u_val, nmse_u_val,
        mse_v_val, nmse_v_val,
        mse_s_val, nmse_s_val
    ) = compute(mse_u, nmse_u, mse_v, nmse_v, mse_s, nmse_s)

    df_mse = pl.DataFrame(
        {
            "time": eval_times.values,
            "MSE_u_ML": mse_u_val.values,
            "NMSE_u_ML": nmse_u_val.values,
            "MSE_v_ML": mse_v_val.values,
            "NMSE_v_ML": nmse_v_val.values,
            "MSE_speed_ML": mse_s_val.values,
            "NMSE_speed_ML": nmse_s_val.values,
        }
    )

    # 6. Append the global average row
    df_mse_final = append_mean_row(df_mse, time_col="time")

    # 7. Save and return
    out_path = exp_config.metrics / "velocity_mse.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_mse_final.write_csv(out_path)

    print(f"MSE & NMSE calculation complete. Saved to {out_path}")
    return df_mse_final