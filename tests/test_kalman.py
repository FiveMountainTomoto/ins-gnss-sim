"""单元测试：验证 `KalmanFilter` 的基本标量更新行为。"""

import numpy as np
import numpy.testing as npt

from core.kalman import KalmanFilter


def test_kalman_scalar_update():
    # 标量系统：x_{k+1} = x_k, z = x + noise
    Q = np.array([[0.0]])
    R = np.array([[1.0]])
    P0 = np.array([[10.0]])
    Phi = np.array([[1.0]])
    H = np.array([[1.0]])
    Tau = np.array([[1.0]])

    kf = KalmanFilter(Q, R, P0, Phi, H, Tau)
    z = np.array([10.0])

    # 执行一次 predict+update
    kf.update(z, time_meas_both='B')

    # 计算期望值（按卡尔曼滤波标准公式）
    P_pred = P0  # Q=0, Phi=1
    K_expected = P_pred @ np.linalg.inv(P_pred + R)
    x_expected = K_expected @ np.array([[10.0]])
    P_expected = P_pred - K_expected @ (P_pred + R) @ K_expected.T

    npt.assert_allclose(kf.x, x_expected, rtol=1e-6, atol=1e-8)
    npt.assert_allclose(kf.P, P_expected, rtol=1e-6, atol=1e-8)
