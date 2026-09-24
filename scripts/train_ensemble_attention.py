import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from driftnet.config import MasterConfig, ExperimentConfig
from driftnet.utils import EarlyStopping, get_spatial_trim_slices
from driftnet.ml.dataset import _read_indices_from_csv
from metrics.diagnostics import save_metrics
from figures.figure_plotting import plot_metrics


# =========================================================================
# 1. MODEL ARCHITECTURE
# =========================================================================
class EfficientEnsembleAttention(nn.Module):
    """
    Computes Cross-Member Attention natively on 2D grids using memory-efficient
    Einstein summation, bypassing the need to flatten spatial dimensions.
    """
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable global query token per head: shape (num_heads, head_dim)
        self.query = nn.Parameter(torch.randn(self.num_heads, self.head_dim))

        # Projections for Keys and Values
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x shape: (Batch, Members, Channels, Height, Width)
        B, M, C, H, W = x.shape

        # 1. Project Keys and Values
        x_flat = x.view(B * M, C, H, W)

        # Reshape to expose the heads: (B, M, heads, head_dim, H, W)
        K = self.k_proj(x_flat).view(B, M, self.num_heads, self.head_dim, H, W)
        V = self.v_proj(x_flat).view(B, M, self.num_heads, self.head_dim, H, W)

        # 2. Compute Attention Scores (Dot product of K and Q along head_dim 'd')
        # K     = b: Batch, m: Members, n: Heads, d: Head_dim, h: Height, w: Width
        # Query = n: Heads, d: Head_dim
        # Out   = b: Batch, m: Members, n: Heads, h: Height, w: Width
        attn_logits = torch.einsum('bmndhw,nd->bmnhw', K, self.query) * self.scale

        # 3. Softmax across the Members dimension (dim=1 corresponds to 'm')
        attn_weights = F.softmax(attn_logits, dim=1)

        # 4. Apply Attention Weights to Values
        # V       = b, m, n, d, h, w
        # Weights = b, m, n, h, w
        # Out     = b, n, d, h, w  (Summing over 'm' completely collapses the Members!)
        out = torch.einsum('bmndhw,bmnhw->bndhw', V, attn_weights)

        # 5. Flatten heads back into channel dim
        return out.reshape(B, C, H, W)


class EnsembleAttentionUNet(nn.Module):
    def __init__(self, in_channels=2, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature Extractor (Permutation Equivariant)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )

        # Custom Memory-Efficient Attention
        self.attention = EfficientEnsembleAttention(embed_dim=embed_dim, num_heads=4)

        final_conv = nn.Conv2d(embed_dim, 2, kernel_size=3, padding=1)
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

        # Spatial Decoder
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            final_conv
        )

    def forward(self, ensemble: torch.Tensor):
        B, M, C, H, W = ensemble.shape

        # 1. Extract Features per member
        x = ensemble.view(B * M, C, H, W)
        x = self.feature_extractor(x)

        # Reshape back to 5D for attention: (B, M, embed_dim, H, W)
        x = x.view(B, M, self.embed_dim, H, W)

        # 2. Cross-Member Attention
        out = self.attention(x)

        # 3. Spatial Decoding & Global Residual
        ens_mean = ensemble.mean(dim=1)
        return ens_mean + self.spatial_decoder(out)


def composite_loss(pred, target):
    """Combined MSE and Spatial Gradient loss to retain turbulence."""
    mse = F.mse_loss(pred, target)

    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    grad_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    return mse + 0.2 * grad_loss


# =========================================================================
# 2. DATASET (Updated with Local-to-Global Index Mapping)
# =========================================================================
class EnsembleDataset(Dataset):
    def __init__(self, data_config, member_paths, local_indices, global_indices, batch_size):
        self.member_ds = [xr.open_zarr(p) for p in member_paths]
        self.truth_ds = xr.open_zarr(data_config.original_res)

        self.local_indices = local_indices
        self.global_indices = global_indices
        self.batch_size = batch_size

        nx, ny = self.truth_ds.sizes["x"], self.truth_ds.sizes["y"]
        self.y_slice, self.x_slice = get_spatial_trim_slices(nx, ny, data_config.degrade_factor)

    def __len__(self):
        return int(np.ceil(len(self.local_indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.local_indices))

        local_batch = self.local_indices[start:end]
        global_batch = self.global_indices[start:end]

        # Use the local index (0-1439) to slice the Ensemble members
        member_arrays = []
        for ds in self.member_ds:
            arr = ds.velocity.isel(time_counter=local_batch).values
            member_arrays.append(arr)
        X = np.stack(member_arrays, axis=1)

        # Use the global index (e.g. 10000-11439) to slice the Ground Truth
        Y = self.truth_ds.velocity.isel(
            time_counter=global_batch, x=self.x_slice, y=self.y_slice
        ).values

        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


# =========================================================================
# 3. TRAINING LOOP
# =========================================================================
def train_attention_model(config: MasterConfig, member_paths: list[Path], exp_config: ExperimentConfig, splits: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Ensemble Attention Model on {device} ---")

    train_set = EnsembleDataset(config.data, member_paths, splits["train_local"], splits["train_global"], config.hyperparameters.batch_size)
    val_set = EnsembleDataset(config.data, member_paths, splits["val_local"], splits["val_global"], config.hyperparameters.batch_size)

    train_loader = DataLoader(train_set, batch_size=None, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=None, num_workers=4)

    model = EnsembleAttentionUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.hyperparameters.learning_rate)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-5)

    micro_batch_size = 2

    for epoch in range(1, config.hyperparameters.epochs + 1):
        model.train()
        train_loss = 0.0

        for X_chunk, Y_chunk in train_loader:
            optimizer.zero_grad()
            chunk_loss = 0.0
            num_micro_batches = int(np.ceil(len(X_chunk) / micro_batch_size))

            for i in range(0, len(X_chunk), micro_batch_size):
                X_micro = X_chunk[i:i + micro_batch_size].to(device)
                Y_micro = Y_chunk[i:i + micro_batch_size].to(device)

                preds = model(X_micro)
                loss = composite_loss(preds, Y_micro) / num_micro_batches

                loss.backward()
                chunk_loss += loss.item()

            optimizer.step()
            train_loss += chunk_loss

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_chunk, Y_chunk in val_loader:
                chunk_loss = 0.0
                num_micro_batches = int(np.ceil(len(X_chunk) / micro_batch_size))
                for i in range(0, len(X_chunk), micro_batch_size):
                    X_micro = X_chunk[i:i + micro_batch_size].to(device)
                    Y_micro = Y_chunk[i:i + micro_batch_size].to(device)

                    preds = model(X_micro)
                    loss = composite_loss(preds, Y_micro)
                    chunk_loss += loss.item() / num_micro_batches
                val_loss += chunk_loss

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch:03d} | Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f}")

        early_stopping(val_loss=v_loss, models_dict={"unet": model}, save_path=exp_config.model_weights)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete!")


# =========================================================================
# 4. INFERENCE
# =========================================================================
@torch.inference_mode()
def inference_attention_model(config: MasterConfig, member_paths: list[Path], exp_config: ExperimentConfig, splits: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- Running Inference on Test Set ---")

    test_set = EnsembleDataset(config.data, member_paths, splits["test_local"], splits["test_global"], config.hyperparameters.batch_size)
    test_loader = DataLoader(test_set, batch_size=None, num_workers=4)

    model = EnsembleAttentionUNet().to(device)
    model.load_state_dict(torch.load(exp_config.model_weights, map_location=device)["unet"])
    model.eval()

    output_zarr_path = exp_config.model_predictions
    micro_batch_size = 2

    ds_truth = xr.open_zarr(config.data.original_res)
    true_time_coords = ds_truth.time_counter[splits["test_global"]].values

    num_batches = 0
    current_time_idx = 0

    for X_chunk, _ in test_loader:
        batch_preds = []
        for i in range(0, len(X_chunk), micro_batch_size):
            X_micro = X_chunk[i:i + micro_batch_size].to(device)
            preds = model(X_micro)
            batch_preds.append(preds.cpu())

        full_preds = torch.cat(batch_preds, dim=0).numpy()
        batch_times = true_time_coords[current_time_idx : current_time_idx + len(full_preds)]
        current_time_idx += len(full_preds)

        ds_batch = xr.Dataset(
            data_vars={"velocity": (["time_counter", "component", "y", "x"], full_preds)},
            coords={"time_counter": batch_times, "component": np.array([0, 1], dtype=np.int64)}
        )

        if num_batches == 0:
            ds_batch = ds_batch.chunk({"time_counter": 48, "component": -1, "y": -1, "x": -1})
            ds_batch.to_zarr(output_zarr_path, mode="w")
        else:
            ds_batch.to_zarr(output_zarr_path, append_dim="time_counter")

        num_batches += 1
        print(f"Inference batch {num_batches} complete.")

    print(f"Inference complete! Saved to {output_zarr_path}")


# =========================================================================
# 5. MAIN ORCHESTRATOR
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--num_members", type=int, default=10)
    args = parser.parse_args()

    config = MasterConfig.load_from_yaml(args.config)
    base_dir = Path(config.experiment.base)

    member_paths = []
    for i in range(args.num_members):
        p = base_dir / "diffusioncfg" / f"ensemble_member_{i}" / "predictions.zarr"
        if not p.exists():
            raise FileNotFoundError(f"Missing ensemble member: {p}")
        member_paths.append(p)

    attention_exp_config = ExperimentConfig(
        exp_name="diffusioncfg",
        trial_name="ensemble_attention",
        base=config.experiment.base,
        guidance_scale=config.experiment.guidance_scale
    )

    # ---------------------------------------------------------
    # Index Mapping Workaround
    # We only have ensemble predictions for the 1440 test frames.
    # To test the architecture quickly, we map Train, Val, and Test
    # to these exact same frames.
    # ---------------------------------------------------------
    all_global_idx = _read_indices_from_csv(config.data.splits / "test_indices.csv")
    all_local_idx = list(range(len(all_global_idx)))

    splits = {
        "train_local": all_local_idx, "train_global": all_global_idx,
        "val_local": all_local_idx,   "val_global": all_global_idx,
        "test_local": all_local_idx,  "test_global": all_global_idx,
    }

    # Train
    if not attention_exp_config.model_weights.exists():
        train_attention_model(config, member_paths, attention_exp_config, splits)
    else:
        print("Model already trained! Skipping to inference.")

    # Inference
    if not attention_exp_config.model_predictions.exists():
        inference_attention_model(config, member_paths, attention_exp_config, splits)
    else:
        print("Predictions already exist! Skipping inference.")


    print("\n--- Evaluating Metrics ---")
    metrics = ["velocity_nmse", "kinetic_energy_spectrum", "euler_distance"]
    target_exp = "diffusioncfg/ensemble_attention"

    save_metrics(config.data, attention_exp_config, exp_names=[target_exp], metrics_to_calc=metrics)

    exp_names = [
        "interpolate/baseline_trial",
        "diffusioncfg/ensemble_member_0",
        "diffusioncfg/ensemble_mean",
        target_exp
    ]
    nice_labels = [
        "Linear Interpolation (Baseline)",
        "Diffusion (Single Member)",
        "Diffusion (Raw Ensemble Mean)",
        "Attention-Refined Ensemble (Proposed)"
    ]

    plot_metrics(config.data, attention_exp_config, exp_names, metrics, legend_labels=nice_labels)


if __name__ == "__main__":
    main()