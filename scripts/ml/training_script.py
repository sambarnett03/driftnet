import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from driftnet.config import DataConfig, ExperimentConfig, HyperparametersConfig

# Import your reusable building block from your source code
from driftnet.ml.dataset import get_train_val_test_datasets
from driftnet.ml.models import Strict2xOceanUNet
from driftnet.ml.train import train_model


def train_downscale(
    hyper_config: HyperparametersConfig, data_config: DataConfig, exp_config: ExperimentConfig
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        data_config.degraded_res, data_config.original_res, hyper_config.batch_size
    )

    # Note: batch_size=None tells PyTorch the dataset handles batching itself!
    train_loader = DataLoader(
        train_dataset, batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=4
    )

    # --- 3. Initialize Model, Loss, Optimizer ---
    model = Strict2xOceanUNet(n_channels=2, n_classes=2, base_features=32)
    model = model.to(device)

    print("Compiling model for A100 architecture optimization...")
    model = torch.compile(model)

    # Using Mean Squared Error (L2 loss) as standard for super-resolution / regression tasks
    criterion = nn.MSELoss()

    model, history = train_model(
        model,  # type: ignore
        hyper_config,
        exp_config,
        train_loader,
        val_loader,
        criterion,
        device,
    )
