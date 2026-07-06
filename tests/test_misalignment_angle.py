import numpy as np
from simulation.misalignment_angle import (
    quarternion_add_misalignment_angle,
    quarternion_minus_misalignment_angle,
    get_misalignment_angle_from_quaternion_error,
)
from core.basic import Quaternion


def test_add_and_get_phi():
    q = Quaternion.from_rotvec(np.array([0.1, -0.2, 0.05]))
    phi = np.array([1e-3, -2e-3, 0.5e-3])
    q_added = quarternion_add_misalignment_angle(q, phi)
    assert isinstance(q_added, Quaternion)
    phi_est = get_misalignment_angle_from_quaternion_error(q, q_added)
    assert np.allclose(phi_est, phi, atol=1e-8)


def test_minus_and_get_phi():
    q = Quaternion.from_rotvec(np.array([0.2, 0.1, -0.05]))
    phi = np.array([2e-3, -1e-3, 0.8e-3])
    q_minus = quarternion_minus_misalignment_angle(q, phi)
    phi_est = get_misalignment_angle_from_quaternion_error(q, q_minus)
    assert np.allclose(phi_est, -phi, atol=1e-8)


def test_add_then_minus_roundtrip():
    q = Quaternion.from_rotvec(np.array([0.05, -0.03, 0.02]))
    phi = np.array([0.001, 0.0005, -0.0002])
    q_added = quarternion_add_misalignment_angle(q, phi)
    q_restored = quarternion_minus_misalignment_angle(q_added, phi)
    assert np.allclose(q.to_rotvec(), q_restored.to_rotvec(), atol=1e-8)
