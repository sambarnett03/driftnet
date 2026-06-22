import pytest
import torch

from driftnet.ml.models import Strict2xOceanUNet  # Adjust import based on your exact path


@pytest.fixture
def unet_model():
    """Fixture to initialize the U-Net model with lightweight settings for testing."""
    return Strict2xOceanUNet(n_channels=2, n_classes=2, base_features=8)


def test_unet_dimensions_double_exactly(unet_model):
    """
    GIVEN a low-resolution tensor of shape [1, 2, 605, 1072]
    WHEN passed through the Strict2xOceanUNet forward pass
    THEN the output tensor must double in spatial dimensions exactly to [1, 2, 1210, 2144]
    """
    # Arrange: Create mock input tensor mimicking your exact low-res data shape
    batch_size = 1
    channels = 2
    low_res_height = 605
    low_res_width = 1072

    x_low_res = torch.randn(batch_size, channels, low_res_height, low_res_width)

    # Act: Run forward pass
    predictions = unet_model(x_low_res)

    # Assert: Verify the exact output dimension sizes
    expected_height = low_res_height * 2  # 1210
    expected_width = low_res_width * 2  # 2144

    assert predictions.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {predictions.shape[0]}"
    )
    assert predictions.shape[1] == channels, (
        f"Expected channels {channels}, got {predictions.shape[1]}"
    )
    assert predictions.shape[2] == expected_height, (
        f"Expected height {expected_height}, got {predictions.shape[2]}"
    )
    assert predictions.shape[3] == expected_width, (
        f"Expected width {expected_width}, got {predictions.shape[3]}"
    )


def test_unet_handles_variable_batch_sizes(unet_model):
    """
    Verifies that the model can handle a batch size of 4 without crashing.
    """
    x_low_res = torch.randn(4, 2, 605, 1072)
    predictions = unet_model(x_low_res)

    assert predictions.shape[0] == 4
