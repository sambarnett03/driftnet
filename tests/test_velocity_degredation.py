import numpy as np
import pytest

from driftnet.data import _degrade_u, _degrade_v, _validate_dimensions


def test_validate_dimensions_success():
    # Setup an array of (2, 10, 20) with n=5
    dummy_array = np.zeros((2, 10, 20))
    coarse_lats, coarse_lons = _validate_dimensions(dummy_array, n=5)
    assert coarse_lats == 2
    assert coarse_lons == 4


def test_validate_dimensions_invalid_shape():
    # Array missing the U/V splitting dimension
    invalid_array = np.zeros((10, 20))
    with pytest.raises(ValueError, match="Input array must have a shape of"):
        _validate_dimensions(invalid_array, n=5)


def test_validate_dimensions_indivisible():
    # 11 lats is not divisible by n=5
    invalid_array = np.zeros((2, 11, 20))
    with pytest.raises(ValueError, match="must be cleanly divisible by n="):
        _validate_dimensions(invalid_array, n=5)


def test_degrade_u_math():
    """
    Test that U looks specifically at the easternmost column
    of the block and takes its vertical average.
    """
    # Create a single 2x2 grid cell block for U
    # Only the Eastern face (column index 1) should matter.
    u_input = np.array(
        [
            [10, 4],  # Row 0
            [20, 6],  # Row 1
        ]
    )
    # Expected: average of the eastern column [4, 6] -> 5.0
    # Inputs expected by function: array, n, coarse_lats, coarse_lons
    result = _degrade_u(u_input, n=2, coarse_lats=1, coarse_lons=1)

    assert result.shape == (1, 1)
    assert result[0, 0] == 5.0


def test_degrade_v_math():
    """
    Test that V looks specifically at the northernmost row
    of the block and takes its horizontal average.
    """
    # Create a single 2x2 grid cell block for V
    # Only the Northern face (row index 1) should matter.
    v_input = np.array(
        [
            [10, 20],  # Row 0
            [3, 5],  # Row 1 (Northern face)
        ]
    )
    # Expected: average of the northern row [3, 5] -> 4.0
    result = _degrade_v(v_input, n=2, coarse_lats=1, coarse_lons=1)

    assert result.shape == (1, 1)
    assert result[0, 0] == 4.0
