from simulation.av_to_imu import av_to_imu
import os
import sys
import numpy as np

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_av_to_imu_output_shape():
    """Test that output arrays have correct shape (N-1, 3)."""
    n_samples = 10
    att = np.zeros((n_samples, 3))
    vn = np.zeros((n_samples, 3))
    vn[:, 1] = 10.0  # constant northward velocity
    pos = np.zeros((n_samples, 3))

    wm, vm = av_to_imu(att, vn, pos, dt=1.0)

    assert wm.shape == (
        n_samples - 1, 3), f"Expected wm shape {(n_samples - 1, 3)}, got {wm.shape}"
    assert vm.shape == (
        n_samples - 1, 3), f"Expected vm shape {(n_samples - 1, 3)}, got {vm.shape}"


def test_av_to_imu_constant_attitude():
    """Test with constant zero attitude (no rotation)."""
    n_samples = 5
    att = np.zeros((n_samples, 3))
    vn = np.tile([0.0, 10.0, 0.0], (n_samples, 1))
    pos = np.zeros((n_samples, 3))

    wm, vm = av_to_imu(att, vn, pos, dt=1.0)

    # With constant attitude, wm should be small (near-zero rotation increments)
    assert wm.shape == (n_samples - 1, 3)
    assert vm.shape == (n_samples - 1, 3)
    assert np.all(np.isfinite(wm)), "wm contains non-finite values"
    assert np.all(np.isfinite(vm)), "vm contains non-finite values"


def test_av_to_imu_constant_velocity():
    """Test with constant velocity and slowly changing attitude."""
    n_samples = 8
    att = np.tile([0.01, 0.0, 0.0], (n_samples, 1)) * \
        np.arange(n_samples).reshape(-1, 1)
    vn = np.tile([0.0, 10.0, 0.0], (n_samples, 1))
    pos = np.zeros((n_samples, 3))

    wm, vm = av_to_imu(att, vn, pos, dt=0.5)

    assert wm.shape == (n_samples - 1, 3)
    assert vm.shape == (n_samples - 1, 3)
    # wm should be non-zero since attitude is changing
    assert np.any(np.abs(wm) > 1e-8), "wm should have non-zero components"


def test_av_to_imu_invalid_input():
    """Test that invalid inputs raise ValueError."""
    att_bad = np.zeros((10, 2))  # Wrong shape
    vn = np.zeros((10, 3))
    pos = np.zeros((10, 3))

    try:
        av_to_imu(att_bad, vn, pos, 1.0)
        assert False, "Should have raised ValueError for wrong att shape"
    except ValueError:
        pass  # Expected


def test_av_to_imu_short_sequence():
    """Test that sequences with n<2 return empty arrays."""
    att = np.zeros((1, 3))
    vn = np.zeros((1, 3))
    pos = np.zeros((1, 3))

    wm, vm = av_to_imu(att, vn, pos, 1.0)

    assert wm.shape == (0, 3)
    assert vm.shape == (0, 3)
