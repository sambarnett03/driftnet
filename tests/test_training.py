from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from driftnet.ml.train import train_model, train_one_epoch, validate_one_epoch

# --- Dummy Models and Data for Testing ---


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x):
        # Flatten for the linear layer just to make the dummy work
        x = x.view(x.size(0), -1)
        return self.layer(x)


def get_dummy_dataloader(chunk_size=4, input_shape=(10,), num_batches=2) -> DataLoader:
    """Creates a real PyTorch DataLoader containing random dummy data."""
    total_samples = chunk_size * num_batches

    # Generate the full dataset tensors
    x = torch.randn(total_samples, *input_shape)
    y = torch.randn(total_samples, 2)

    dataset = TensorDataset(x, y)

    # batch_size=chunk_size simulates the large chunks your custom dataset yields
    return DataLoader(dataset, batch_size=chunk_size)


# --- Test Cases ---


@patch("torch.autocast")  # Patch autocast to prevent errors on CPU-only test runners
def test_train_one_epoch(mock_autocast):
    # Setup
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    device = torch.device("cpu")  # Force CPU for testing
    dataloader = get_dummy_dataloader(chunk_size=4, input_shape=(10,), num_batches=2)

    # Execute
    loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        micro_batch_size=2,  # Exactly half the chunk size to test the micro-batch loop
        debug=False,
    )

    # Assert
    assert isinstance(loss, float)
    assert loss >= 0.0
    mock_autocast.assert_called_with(device_type="cuda", dtype=torch.bfloat16)


@patch("torch.autocast")
def test_validate_one_epoch(mock_autocast):
    # Setup
    model = DummyModel()
    criterion = nn.MSELoss()
    device = torch.device("cpu")

    # Using shape (1, 10, 1, 1) so channels_last formatting doesn't throw dimension errors
    dataloader = get_dummy_dataloader(chunk_size=4, input_shape=(10, 1, 1), num_batches=2)

    # Execute
    loss = validate_one_epoch(
        model=model, dataloader=dataloader, criterion=criterion, device=device, micro_batch_size=2
    )

    # Assert
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert not model.training  # Ensure model was put into eval mode


@patch("driftnet.ml.train.train_one_epoch")
@patch("driftnet.ml.train.validate_one_epoch")
@patch("driftnet.ml.train.EarlyStopping")
def test_train_model(mock_early_stopping_cls, mock_validate, mock_train, tmp_path):
    # Setup
    model = DummyModel()
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    train_loader = get_dummy_dataloader()
    val_loader = get_dummy_dataloader()

    # Mock configs
    hyper_config = MagicMock()
    hyper_config.epochs = 2
    hyper_config.batch_size = 48
    hyper_config.micro_batch_size = 24
    hyper_config.learning_rate = 1e-3

    exp_config = MagicMock()
    exp_config.base_path = tmp_path  # Use pytest's tmp_path for safe file writing

    # FIX: Define the model_weights path explicitly so parent directory checks pass
    exp_config.model_weights = tmp_path / "model_weights.pt"

    # Configure mock returns
    mock_train.return_value = 0.5
    mock_validate.return_value = 0.4

    # Mock early stopping instance to NOT trigger immediately
    mock_early_stopping_instance = MagicMock()
    mock_early_stopping_instance.early_stop = False
    mock_early_stopping_cls.return_value = mock_early_stopping_instance

    # Mock torch.load to return a fake state dict so the final reload works
    with patch("torch.load", return_value={"unet": model.state_dict()}):
        # Execute
        returned_model, history = train_model(
            model=model,
            hyper_config=hyper_config,
            exp_config=exp_config,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

    # Assert
    assert mock_train.call_count == 2  # Called for 2 epochs
    assert mock_validate.call_count == 2
    assert "train_loss" in history
    assert "val_loss" in history
    assert history["train_loss"] == [0.5, 0.5]
    assert history["val_loss"] == [0.4, 0.4]
    assert (tmp_path / "history.json").exists()
