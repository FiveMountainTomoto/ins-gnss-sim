"""
SINS 误差状态转移矩阵构造器
"""
from __future__ import annotations

import numpy as np
from config import wgs84
from .basic.linalg import skew_symmetric
from .earth_model import EarthModel


def build_sins_error_transition(earth: EarthModel, Cnb: np.ndarray, specific_force_b: np.ndarray) -> np.ndarray:
    """
    构造 15x15 的 SINS 误差态转移矩阵 Ft。

    :param earth: EarthModel 实例，包含位置、速度和相关地球参数
    :param Cnb: 机体到导航系方向余弦矩阵 (3x3)
    :param specific_force_b: 机体系比力向量 fb (3,)
    :return: Ft (15x15)
    """
    tl = earth.tl  # tan(lat)
    secl = 1.0 / earth.cl  # sec(lat)
    inv_rmh = earth.inv_rm_plus_h
    inv_rnh = earth.inv_rn_plus_h
    inv_cl_rnh = earth.inv_cos_lat_rn_plus_h

    inv_rmh2 = inv_rmh * inv_rmh
    inv_rnh2 = inv_rnh * inv_rnh

    v_e = earth.vn[0]
    v_n = earth.vn[1]

    # 构造位置相关的曲率耦合项 Mp1 和 Mp2
    omega_ie_n = earth.omega_ie_n
    Mp1 = np.array([
        [0.0, 0.0, 0.0],
        [-omega_ie_n[2], 0.0, 0.0],
        [omega_ie_n[1], 0.0, 0.0],
    ])

    Mp2 = np.array([
        [0.0, 0.0, v_n * inv_rmh2],
        [0.0, 0.0, -v_e * inv_rnh2],
        [v_e * inv_cl_rnh * secl, 0.0, -v_e * inv_rnh2 * tl],
    ])

    # -omega_in 的反对称矩阵
    Maa = skew_symmetric(-earth.omega_in_n)

    # 姿态到速度耦合
    Mav = np.array([
        [0.0, -inv_rmh, 0.0],
        [inv_rnh, 0.0, 0.0],
        [inv_rnh * tl, 0.0, 0.0],
    ])

    # 姿态到位置耦合
    Map = Mp1 + Mp2

    # 速度到姿态耦合
    Mva = skew_symmetric(Cnb @ specific_force_b)

    # 速度到速度
    Mvv = skew_symmetric(earth.vn) @ Mav - skew_symmetric(earth.omega_en_n)

    # 速度到位置
    Mvp = skew_symmetric(earth.vn) @ (2.0 * Mp1 + Mp2)

    # 重力模型的纬向微分修正
    slcl = earth.sl * earth.cl
    g0 = wgs84.g0

    Mvp[2, 0] = Mvp[2, 0] - g0 * \
        (5.27094e-3 * 2.0 * slcl + 2.32718e-5 * 4.0 * earth.sl2 * slcl)
    Mvp[2, 2] = Mvp[2, 2] + 3.086e-6

    # 位置相关块
    Mpv = np.array([
        [0.0, inv_rmh, 0.0],
        [inv_cl_rnh, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    Mpp = np.array([
        [0.0, 0.0, -v_n * inv_rmh2],
        [v_e * inv_cl_rnh * tl, 0.0, -v_e * inv_rnh2 * secl],
        [0.0, 0.0, 0.0],
    ])

    O33 = np.zeros((3, 3))

    # 组装 15x15 矩阵
    top_row = np.hstack([Maa, Mav, Map, -Cnb, O33])
    mid_row = np.hstack([Mva, Mvv, Mvp, O33, Cnb])
    pos_row = np.hstack([O33, Mpv, Mpp, O33, O33])

    bottom_rows = np.zeros((6, 15))

    Ft = np.vstack([top_row, mid_row, pos_row, bottom_rows])

    return Ft
