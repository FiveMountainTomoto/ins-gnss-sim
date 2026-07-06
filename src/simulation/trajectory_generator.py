"""Trajectory generator: modernized, clearer implementation of trjprofile.

Functions here translate the provided MATLAB `trjprofile` into a readable
and well-typed Python implementation. The expected maneuver matrix rows are:
    [wx, wy, wz, a_forward, duration]

Notes:
 - `a2mat` in MATLAB corresponds to `euler_to_dir_cos_mat` in this codebase.
 - `EarthModel` is used to obtain curvature radii required for geodetic updates.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from core.basic.linalg import euler_to_dir_cos_mat
from core.earth_model import EarthModel


def trj_profile(
    attitude0: np.ndarray,
    vn0: np.ndarray,
    pos0: np.ndarray,
    maneuvers: np.ndarray,
    dt: float,
    fir_length: int = 21,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据一系列机动动作生成轨迹剖面。

    参数
    ----------
    attitude0
        初始欧拉姿态角（俯仰，滚转，偏航），形状为(3,)。
    vn0
        初始导航坐标系速度 [北向速度, 东向速度, 天向速度]，形状为(3,)。
    pos0
        初始位置 [纬度, 经度, 高度]（纬度/经度单位为弧度，高度单位为米），形状为(3,)。
    maneuvers
        形状为(M, >=5)的数组。每一行: [角速度x, 角速度y, 角速度z, 前向加速度, 持续时间]。
    dt
        采样间隔时间（秒）。
    fir_length
        用于简单低通平滑的FIR滤波器长度（默认21）。

    返回
    -------
    attitude, vn, pos
        形状为(N, 3)的数组，分别包含姿态角（弧度）、速度（m/s）和位置。
    """

    # 基本数据验证
    attitude0 = np.asarray(attitude0, dtype=float).ravel()
    vn0 = np.asarray(vn0, dtype=float).ravel()
    pos0 = np.asarray(pos0, dtype=float).ravel()
    maneuvers = np.asarray(maneuvers, dtype=float)

    if attitude0.shape != (3,) or vn0.shape != (3,) or pos0.shape != (3,):
        raise ValueError("attitude0/vn0/pos0 必须是包含3个元素的一维数组")
    if maneuvers.ndim != 2 or maneuvers.shape[1] < 5:
        raise ValueError("maneuvers 必须是至少包含5列的二维数组")

    # 估算总步数并预分配内存
    total_time = float(np.sum(maneuvers[:, 4]))
    max_steps = int(np.floor(total_time / dt)) + 2

    attitudes = np.zeros((max_steps, 3), dtype=float)  # 姿态历史记录
    velocities = np.zeros((max_steps, 3), dtype=float)  # 速度历史记录
    positions = np.zeros((max_steps, 3), dtype=float)   # 位置历史记录

    # 初始化
    step = 0
    attitudes[0] = attitude0
    velocities[0] = vn0
    positions[0] = pos0

    # 初始时刻机体坐标系下的纵向速度：vb = Cbn.T @ vn0
    Cbn0 = euler_to_dir_cos_mat(*attitude0)  # 初始方向余弦矩阵
    vb0 = Cbn0.T @ vn0
    v_long_body = float(vb0[1])  # 机体纵向速度（前向）

    # 简单的FIR低通滤波器（移动平均）
    fir_len = int(fir_length)
    if fir_len <= 0:
        raise ValueError("fir_length 必须为正数")
    fir_coeff = np.ones(fir_len, dtype=float) / fir_len  # 等权重系数

    # 用于存储最近的[姿态角, 纵向速度]的缓冲区
    buffer = np.tile(np.hstack([attitude0, v_long_body]), (fir_len, 1))

    prev_vn = velocities[0].copy()  # 上一时刻速度
    prev_pos = positions[0].copy()  # 上一时刻位置

    # 遍历所有机动动作
    for maneuver in maneuvers:
        ang_rates = maneuver[0:3]    # 角速度分量 [wx, wy, wz]
        a_forward = float(maneuver[3])  # 前向加速度
        duration = float(maneuver[4])   # 当前机动动作持续时间

        n_steps = int(np.floor(duration / dt + 1e-9))  # 当前动作的步数
        for _ in range(max(1, n_steps)):
            # 更新姿态角和机体纵向速度
            attitude0 = attitude0 + ang_rates * dt
            v_long_body = v_long_body + a_forward * dt

            # 更新缓冲区并计算滤波输出 y = [姿态角, 纵向速度]
            buffer = np.vstack(
                (buffer[1:], np.hstack([attitude0, v_long_body])))
            y = (fir_coeff.reshape(-1, 1) * buffer).sum(axis=0)

            # 存储滤波后的姿态角
            step += 1
            attitudes[step] = y[0:3]

            # 根据姿态计算导航坐标系速度：vn = Cbn @ [0; 纵向速度; 0]
            Cbn_k = euler_to_dir_cos_mat(*attitudes[step])
            vn_k = Cbn_k @ np.array([0.0, y[3], 0.0])  # 仅前向速度分量
            velocities[step] = vn_k

            # 使用中点速度进行位置积分（提高精度）
            vn_mid = 0.5 * (prev_vn + velocities[step])

            # 初始化地球模型（用于曲率半径等计算）
            eth = EarthModel(tuple(prev_pos), vn_mid)

            # 位置增量计算（纬度，经度，高度）
            dlat = vn_mid[1] / eth.rm_plus_h          # 纬度变化率（北向速度）
            dlon = vn_mid[0] / eth.cos_lat_rn_plus_h  # 经度变化率（东向速度）
            dh = vn_mid[2]                            # 高度变化率

            positions[step, 0] = prev_pos[0] + dlat * dt  # 纬度更新
            positions[step, 1] = prev_pos[1] + dlon * dt  # 经度更新
            positions[step, 2] = prev_pos[2] + dh * dt    # 高度更新

            # 更新"上一时刻"变量
            prev_vn = velocities[step].copy()
            prev_pos = positions[step].copy()

    # 截取实际使用部分
    attitudes = attitudes[: step + 1]
    velocities = velocities[: step + 1]
    positions = positions[: step + 1]

    return attitudes, velocities, positions


if __name__ == "__main__":
    # small smoke test
    a0 = np.zeros(3)
    v0 = np.array([0.0, 10.0, 0.0])
    p0 = np.zeros(3)
    maneuvers = np.array([[0.0, 0.0, 0.0, 0.0, 5.0]])
    att, vn, pos = trj_profile(a0, v0, p0, maneuvers, 1.0)
    print(f"Generated {len(att)} samples")
