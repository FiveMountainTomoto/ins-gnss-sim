from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KalmanFilter:
    """卡尔曼滤波器

    属性说明：
       Q: 过程噪声协方差
       R: 量测噪声协方差
       P: 状态协方差矩阵 P（后验）
       x: 状态向量 x（后验）
       Phi: 状态转移矩阵 Phi
       H: 观测矩阵 H
       Tau: 过程噪声变换矩阵 Tau
    """

    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    x: np.ndarray
    Phi: np.ndarray
    H: np.ndarray
    Tau: np.ndarray

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        P0: np.ndarray,
        Phi: np.ndarray,
        H: np.ndarray,
        Tau: Optional[np.ndarray] = None,
    ) -> None:
        """初始化卡尔曼滤波器。

        参数：
            Q: 过程噪声协方差
            R: 量测噪声协方差
            P0: 初始状态协方差 P_0
            Phi: 状态转移矩阵
            H: 观测矩阵
            Tau: 过程噪声变换矩阵（可选，默认为单位矩阵）
        """
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.P = np.array(P0, dtype=float)

        # 将状态向量初始化为零（列向量）
        _, state_dim = np.atleast_2d(H).shape
        self.x = np.zeros((state_dim, 1), dtype=float)

        self.Phi = np.array(Phi, dtype=float)
        self.H = np.array(H, dtype=float)

        if Tau is None:
            self.Tau = np.eye(state_dim, dtype=float)
        else:
            self.Tau = np.array(
                Tau, dtype=float)

        # 在 update 中计算并填充的占位属性
        self.pre_x: Optional[np.ndarray] = None
        self.pre_P: Optional[np.ndarray] = None
        self.cross_cov: Optional[np.ndarray] = None
        self.inno_cov: Optional[np.ndarray] = None
        self.km_gain: Optional[np.ndarray] = None

    def update(self, measurement: Optional[np.ndarray] = None, time_meas_both: str = "B") -> None:
        """执行一次滤波更新

        参数：
            measurement: 量测向量 Z_k（列向量）。在进行量测更新时必需。
            time_meas_both: 模式，取值 'T'（仅时间更新）、'M'（仅量测更新）、'B'（先预测再校正）。
                - 'T': 仅时间更新（预测）
                - 'M': 仅量测更新（不预测）
                - 'B': 先预测再量测更新
        """
        mode = time_meas_both.upper()
        if mode not in ("T", "M", "B"):
            raise ValueError("time_meas_both must be one of 'T', 'M', or 'B'")

        # 状态预测（时间更新）
        if mode in ("T", "B"):
            self.pre_x = self.Phi @ self.x
            self.pre_P = (
                self.Phi @ self.P @ self.Phi.T
                + self.Tau @ self.Q @ self.Tau.T
            )
        else:  # 'M'
            self.pre_x = self.x
            self.pre_P = self.P

        # 量测更新
        if mode in ("M", "B"):
            if measurement is None:
                raise ValueError(
                    "measurement must be provided for measurement update")
            z = np.array(measurement, dtype=float)
            # 确保为列向量
            if z.ndim == 1:
                z = z.reshape(-1, 1)

            self.cross_cov = self.pre_P @ self.H.T
            self.inno_cov = (
                self.H @ self.cross_cov + self.R
            )
            # 计算卡尔曼增益
            self.km_gain = np.linalg.solve(self.inno_cov, self.cross_cov.T).T

            innovation = z - self.H @ self.pre_x
            self.x = self.pre_x + self.km_gain @ innovation
            self.P = (
                self.pre_P -
                self.km_gain @ self.inno_cov @ self.km_gain.T
            )
        else:  # 'T' only
            self.x = self.pre_x
            self.P = self.pre_P

        # 对称化协方差矩阵以提升数值稳定性
        self.P = (
            self.P + self.P.T) / 2.0


__all__ = ["KalmanFilter"]
