"""为 IMU 增量注入常值偏置、随机游走和高斯噪声。"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def imu_add_errors(wm: np.ndarray, vm: np.ndarray,
                   bias_gyro: np.ndarray, rw_gyro: np.ndarray,
                   bias_acc: np.ndarray, rw_acc: np.ndarray,
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """向 IMU 增量注入误差。

    简单模型：
      wm_noisy = wm + bias_gyro + N(0, rw_gyro*sqrt(dt))
      vm_noisy = vm + bias_acc + N(0, rw_acc*sqrt(dt))

    偏置（bias_gyro / bias_acc）在时间步内保持不变，带有独立的随机游走增量。

    Args:
        wm: 形状 (N,3) 的角增量序列（弧度）
        vm: 形状 (N,3) 的速度增量序列（m/s）
        bias_gyro: 初始陀螺常值零偏 (3,)
        rw_gyro: 陀螺角度随机游走谱密度 (3,)（单位与 wm 匹配，乘 sqrt(s) 得到 std）
        bias_acc: 加速度计常值偏置 (3,)
        rw_acc: 加速度计速度随机游走谱密度 (3,)（单位与 vm 匹配）
        dt: 每个样本的时间长度（秒）

    Returns:
        wm_err, vm_err: 注入误差后的序列，与输入形状相同
    """
    wm = np.atleast_2d(wm).astype(float)
    vm = np.atleast_2d(vm).astype(float)

    N = wm.shape[0]
    if vm.shape[0] != N:
        raise ValueError("wm and vm must have same first dimension")

    bias_g = np.asarray(bias_gyro, dtype=float).reshape(1, 3)
    bias_a = np.asarray(bias_acc, dtype=float).reshape(1, 3)

    # 随机游走：每步的偏置增量 ~ N(0, rw * sqrt(dt))
    rwg = np.asarray(rw_gyro, dtype=float).reshape(1, 3)
    rwa = np.asarray(rw_acc, dtype=float).reshape(1, 3)

    # 产生噪声项（测量噪声）和偏置随机游走项
    meas_noise_w = np.random.randn(N, 3) * (rwg * np.sqrt(dt))
    meas_noise_a = np.random.randn(N, 3) * (rwa * np.sqrt(dt))

    # 偏置按采样时间累加
    wm_err = wm + np.repeat(bias_g * dt, N, axis=0) + meas_noise_w
    vm_err = vm + np.repeat(bias_a * dt, N, axis=0) + meas_noise_a

    return wm_err, vm_err


__all__ = ["imu_add_errors"]
