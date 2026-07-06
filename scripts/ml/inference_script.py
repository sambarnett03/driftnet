import numpy as np
import xarray as xr
import zarr

from driftnet.config import DataConfig, ExperimentConfig, HyperparametersConfig
from driftnet.ml.dataset import OceanDownscaleSet, _read_indices_from_csv
from driftnet.ml.inference import stream_inference


def inference_over_test_set(
    data_config: DataConfig, hyper_config: HyperparametersConfig, exp_config: ExperimentConfig
):

    print("Initializing datasets...")
    csv_path = data_config.splits / "test_indices.csv"
    test_indices = _read_indices_from_csv(csv_path)

    test_dataset = OceanDownscaleSet(data_config, hyper_config, time_indices=test_indices)

    output_zarr_path = exp_config.model_predictions
    print(f"Streaming structured predictions directly to {output_zarr_path}...")

    total_loss = 0.0
    num_batches = 0
    current_sample_idx = 0

    # 1. Consume the generator batch-by-batch
    for preds_batch, batch_loss in stream_inference(
        test_set=test_dataset, weights_path=exp_config.model_weights, num_workers=4
    ):
        total_loss += batch_loss
        num_batches += 1
        batch_size = preds_batch.shape[0]
        current_sample_idx += batch_size

        # Wrap this specific batch in an Xarray Dataset
        # Ensure predictions are float32 to match target
        ds_batch = xr.Dataset(
            data_vars={
                "velocity": (
                    ["time_counter", "component", "y", "x"],
                    preds_batch.astype(np.float32),
                )
            },
            coords={
                "component": np.array([0, 1], dtype=np.int64),
            },
        )

        # 2. Write or Append to Zarr
        if num_batches == 1:
            ds_batch.attrs = {
                "description": "2x Strict Ocean Super-Resolution U-Net Predictions",
                "checkpoint_used": str(exp_config.model_weights),
            }
            # Chunk along the new dimension names
            ds_batch = ds_batch.chunk({"time_counter": 48, "component": -1, "y": -1, "x": -1})
            ds_batch.to_zarr(output_zarr_path, mode="w")  # type: ignore
        else:
            ds_batch.to_zarr(output_zarr_path, append_dim="time_counter")

    # 3. Final calculations
    avg_test_loss = total_loss / num_batches
    print(f"Inference stream complete! Total samples predicted: {current_sample_idx}.")
    print(f"Final Test MSE: {avg_test_loss:.6f}")

    # 4. Patch the exact time coordinates into the completed Zarr store
    print("Aligning true time coordinates with predictions...")
    with xr.open_zarr(data_config.original_res) as ds_truth:
        # Extract the exact number of timestamps needed
        true_time_coords = ds_truth.time_counter[test_indices].values

    # Write the time_counter coordinates into the Zarr store using Xarray append mode
    ds_time = xr.Dataset(coords={"time_counter": true_time_coords})
    ds_time.to_zarr(output_zarr_path, mode="a")  # type: ignore

    # Patch the MSE via raw zarr
    z_store = zarr.open(str(output_zarr_path), mode="a")
    z_store.attrs["test_mse"] = float(avg_test_loss)
    print("Successfully patched Zarr metadata with final MSE and true time coordinates.")
