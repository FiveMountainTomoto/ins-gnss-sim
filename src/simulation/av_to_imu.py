"""Convert attitude/velocity/position sequences into IMU increments.

This file implements `av_to_imu(att, vn, pos, dt)` adapted from the
provided MATLAB `av2imu` code.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from core.basic.linalg import (
    euler_to_quaternion,
    rotvec_to_quaternion,
    quarternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotvec,
    skew_symmetric,
    euler_to_dir_cos_mat,
)
from core.earth_model import EarthModel


def av_to_imu(att: np.ndarray, vn: np.ndarray, pos: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute IMU angular increments `wm` and specific-force increments `vm`.

    Parameters
    ----------
    att
        Euler attitudes array, shape (N,3), radians.
    vn
        Navigation velocities array, shape (N,3).
    pos
        Positions array, shape (N,3) (lat, lon in radians, h in meters).
    dt
        Time step (s).

    Returns
    -------
    wm, vm
        Arrays of shape (N-1, 3) corresponding to incremental body rotations
        and delta-velocity increments for each interval.
    """

    att = np.asarray(att, dtype=float)
    vn = np.asarray(vn, dtype=float)
    pos = np.asarray(pos, dtype=float)

    if att.ndim != 2 or att.shape[1] != 3:
        raise ValueError("att must be shape (N,3)")
    if vn.shape != att.shape or pos.shape != att.shape:
        raise ValueError("vn and pos must have same shape as att")

    n = att.shape[0]
    if n < 2:
        return np.empty((0, 3)), np.empty((0, 3))

    I3 = np.eye(3)
    wm_prev = np.zeros(3, dtype=float)
    vm_prev = np.zeros(3, dtype=float)

    wm = np.zeros((n - 1, 3), dtype=float)
    vm = np.zeros((n - 1, 3), dtype=float)

    for k in range(1, n):
        # earth parameters at midpoint
        mid_pos = 0.5 * (pos[k - 1] + pos[k])
        mid_vn = 0.5 * (vn[k - 1] + vn[k])
        eth = EarthModel(tuple(mid_pos), mid_vn)

        # quaternion chain: qbb = conj(q_prev) * q(wnin*dt) * q_next
        q_prev = euler_to_quaternion(*att[k - 1])
        q_rot = rotvec_to_quaternion(eth.omega_in_n * dt)
        q_next = euler_to_quaternion(*att[k])
        qbb = quaternion_multiply(quaternion_multiply(
            quarternion_conjugate(q_prev), q_rot), q_next)

        phim = quaternion_to_rotvec(qbb)

        # solve for body rotation increment wm1: (I + skew(1/12*wm_prev)) wm1 = phim
        A_w = I3 + skew_symmetric((1.0 / 12.0) * wm_prev)
        wm1 = np.linalg.solve(A_w, phim)

        # dvnsf = vn_k - vn_k-1 - g_eff * dt
        dvnsf = vn[k] - vn[k - 1] - eth.gravity_eff_n * dt

        Cbn0 = euler_to_dir_cos_mat(*att[k - 1])

        # rhs = Cbn0.T * (I + skew(wnin*dt/2)) * dvnsf - 1/12 * cross(vm_prev, wm1)
        rhs = Cbn0.T @ ((I3 + skew_symmetric(eth.omega_in_n * dt / 2.0)) @ dvnsf)
        rhs = rhs - (1.0 / 12.0) * np.cross(vm_prev, wm1)

        # left matrix A_v = I + 1/2 * skew(1/6*wm_prev + wm1)
        A_v = I3 + 0.5 * skew_symmetric((1.0 / 6.0) * wm_prev + wm1)
        vm1 = np.linalg.solve(A_v, rhs)

        wm[k - 1] = wm1
        vm[k - 1] = vm1

        wm_prev = wm1
        vm_prev = vm1

    return wm, vm


__all__ = ["av_to_imu"]
