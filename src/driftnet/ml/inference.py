from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from driftnet.ml.models import Strict2xOceanUNet


@torch.inference_mode()
def stream_inference(
    test_set: Dataset, weights_path: Path, num_workers: int = 4
) -> Iterator[tuple[np.ndarray, float]]:
    """
    Streams inference over the test dataset using the best saved model weights.

    Yields:
        A tuple containing:
        - A NumPy array of batch predictions [Batch_Size, Channels, H, W]
        - The batch loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Reconstruct and compile the model structure
    model = Strict2xOceanUNet(n_channels=2, n_classes=2, base_features=32)
    model = model.to(device)

    print("Compiling model for inference optimization...")
    model = torch.compile(model)  # type: ignore

    # Load the best saved weights
    print(f"Loading weights from {weights_path}...")

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["unet"])  # type: ignore
    model.eval()  # type: ignore

    # Create optimized DataLoader (shuffle=False is crucial)
    # Using batch_size=None since test_set is already providing batches
    test_loader = DataLoader(
        test_set, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    criterion = nn.MSELoss()

    print(f"Starting inference stream over {len(test_loader)} batches...")

    for batch_idx, (X_batch, Y_batch) in enumerate(test_loader):
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)

        # Yield the numpy array and the loss value, then discard from memory
        yield predictions.float().cpu().numpy(), loss.item()

        if batch_idx % 10 == 0:
            print(f"batch number {batch_idx} completed")
