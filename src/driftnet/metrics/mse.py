import polars as pl
import xarray as xr

# Assuming these imports match your existing setup
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
    Loads the 4TB ground truth zarr, slices it spatially to match the domain,
    and slices it temporally to match the model evaluation period.
    """
    ds_gt = xr.open_zarr(data_config.original_res)

    # Spatial clipping
    x_slice, y_slice = _get_valid_spatial_slices(data_config)
    ds_gt = ds_gt.isel(x=x_slice, y=y_slice)

    # Temporal clipping (align to the exact timestamps in the ML dataset)
    ds_gt = ds_gt.sel(time_counter=eval_times)

    return ds_gt


def _calculate_spatial_mse(pred_da: xr.DataArray, truth_da: xr.DataArray) -> xr.DataArray:
    """
    Builds the Dask graph to compute the Mean Squared Error over the spatial dimensions.
    Returns a 1D DataArray indexed by time_counter.
    """
    squared_error = (pred_da - truth_da) ** 2

    # Calculate the mean over x and y, leaving a time series
    return squared_error.mean(dim=["x", "y"])


def calculate_velocity_mse(data_config: DataConfig, exp_config: ExperimentConfig) -> pl.DataFrame:
    """
    Main orchestrator to calculate U, V, and Speed MSE between Ground Truth,
    ML Predicted, and Bilinear Interpolated fields.
    """
    print("Loading datasets and building MSE compute graph...")

    # 1. Load predicted and interpolated datasets
    ds_pred = xr.open_zarr(exp_config.model_predictions)
    ds_interp = xr.open_zarr(data_config.interpolated)

    # 2. Get the evaluation timestamps to align the Ground Truth
    eval_times = ds_pred["time_counter"]
    ds_gt = _load_and_align_ground_truth(data_config, eval_times)

    # 3. Extract u, v, and speed components for all three fields
    u_gt, v_gt, speed_gt = _extract_velocity_components(ds_gt)
    u_ml, v_ml, speed_ml = _extract_velocity_components(ds_pred)
    u_in, v_in, speed_in = _extract_velocity_components(ds_interp)

    # 4. Build the lazy MSE graphs
    mse_u_ml = _calculate_spatial_mse(u_ml, u_gt)
    mse_v_ml = _calculate_spatial_mse(v_ml, v_gt)
    mse_s_ml = _calculate_spatial_mse(speed_ml, speed_gt)

    mse_u_in = _calculate_spatial_mse(u_in, u_gt)
    mse_v_in = _calculate_spatial_mse(v_in, v_gt)
    mse_s_in = _calculate_spatial_mse(speed_in, speed_gt)

    print("Executing Dask compute (this may take a moment)...")

    # 5. Trigger Dask to process the graph and return numpy arrays
    # Calling .compute() tells Dask to actually execute the math across workers
    df_mse = pl.DataFrame(
        {
            "time": eval_times.values,
            "MSE_u_ML": mse_u_ml.compute().values,
            "MSE_v_ML": mse_v_ml.compute().values,
            "MSE_speed_ML": mse_s_ml.compute().values,
            "MSE_u_Interp": mse_u_in.compute().values,
            "MSE_v_Interp": mse_v_in.compute().values,
            "MSE_speed_Interp": mse_s_in.compute().values,
        }
    )

    # 6. Append the global average row (using the function from our previous step)
    df_mse_final = append_mean_row(df_mse, time_col="time")

    # 7. Save and return
    out_path = exp_config.metrics / "velocity_mse.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_mse_final.write_csv(out_path)

    print(f"MSE calculation complete. Saved to {out_path}")
    return df_mse_final
