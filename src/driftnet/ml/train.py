import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from typing import Tuple, Dict, List

from driftnet.ml.utils import EarlyStopping, print_gpu_stats
from driftnet.config import HyperparametersConfig, ExperimentConfig

# Enable TF32 math on the A100 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    micro_batch_size: int = 24, # <-- Fits inside 40GB A100 VRAM
    debug: bool=False
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (X_chunk, Y_chunk) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        chunk_loss = 0.0
        # How many micro-batches are in this chunk? (e.g., 48 / 12 = 4)
        num_micro_batches = int(np.ceil(len(X_chunk) / micro_batch_size))

        for i in range(0, len(X_chunk), micro_batch_size):
            X_micro = X_chunk[i : i + micro_batch_size].to(device, non_blocking=True)
            Y_micro = Y_chunk[i : i + micro_batch_size].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(X_micro)
                loss = criterion(predictions, Y_micro)
                loss = loss / num_micro_batches

            loss.backward()
            chunk_loss += loss.item()

        # Update weights only after processing the whole 48-item chunk
        optimizer.step()
        running_loss += chunk_loss

        # Log stats every 50 batches
        if debug is True:
            if batch_idx % 5 == 0:
                print_gpu_stats(batch_idx, len(dataloader))
                # Reset max memory tracked after printing to see true peaks per window
                torch.cuda.reset_peak_memory_stats(device)


    return running_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device,
    micro_batch_size: int = 24
) -> float:
    model.eval()
    running_loss = 0.0

    for X_chunk, Y_chunk in dataloader:
        chunk_loss = 0.0
        num_micro_batches = int(np.ceil(len(X_chunk) / micro_batch_size))

        for i in range(0, len(X_chunk), micro_batch_size):
            X_micro = X_chunk[i : i + micro_batch_size].to(device, non_blocking=True)
            Y_micro = Y_chunk[i : i + micro_batch_size].to(device, non_blocking=True)

            # (Optional but recommended) channels_last format for validation too!
            X_micro = X_micro.to(memory_format=torch.channels_last)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(X_micro)
                loss = criterion(predictions, Y_micro)

            chunk_loss += (loss.item() / num_micro_batches)

        running_loss += chunk_loss

    return running_loss / len(dataloader)



def train_model(
    model: nn.Module,
    hyper_config: HyperparametersConfig,
    exp_config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]]]:


    num_epochs = hyper_config.epochs
    batch_size = hyper_config.batch_size
    micro_batch_size = hyper_config.micro_batch_size
    lr = hyper_config.learning_rate
    save_dir = exp_config.base_path

    history = {"train_loss": [], "val_loss": []}
    temp_checkpoint = save_dir / "temp" / "weights.pth"
    temp_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Initialize your early stopping utility
    early_stopping = EarlyStopping(patience=5, min_delta=1e-5)

    # --- 4. The Loop ---
    print("Starting training loop...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, micro_batch_size=micro_batch_size)
        val_loss = validate_one_epoch(model, val_loader, criterion, device, micro_batch_size=micro_batch_size)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        with open(save_dir / "history.json", "w", encoding="utf-8") as fp:
            json.dump(history, fp)

        early_stopping(val_loss=val_loss, models_dict={"unet": model}, save_path=temp_checkpoint)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    best_model = torch.load(temp_checkpoint)
    model.load_state_dict(best_model["unet"])
    print("Loaded the best synchronized model weights from early stopping.")

    torch.save(best_model, save_dir / 'best_weights.pth')

    return model, history
