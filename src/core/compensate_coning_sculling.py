import numpy as np
from typing import Tuple
from config import savage_coefficients


def compensate_coning_sculling(
    delta_theta_series: np.ndarray,
    delta_v_series: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    圆锥/划船误差补偿
    :param delta_theta_series: 陀螺仪增量角序列，形状为 (N, 3)
    :type delta_theta_series: ndarray[N, 3]
    :param delta_v_series: 加速度计增量速度序列，形状为 (N, 3)
    :type delta_v_series: ndarray[N, 3]
    :return: 补偿后的增量角和增量速度
    :rtype: Tuple[ndarray[3,], ndarray[3,]]
    """
    n = delta_theta_series.shape[0]
    if n < 2 or n > 6:
        raise ValueError("order must match input rows and be 2~5")
    coeff = savage_coefficients(n)

    theta_sum = delta_theta_series.sum(axis=0)
    dv_sum = delta_v_series.sum(axis=0)

    theta_mid = coeff @ delta_theta_series[:-1, :]
    dv_mid = coeff @ delta_v_series[:-1, :]

    com_con = np.cross(theta_mid, delta_theta_series[-1, :])
    com_scu = np.cross(theta_mid, delta_v_series[-1, :]) + \
        np.cross(dv_mid, delta_theta_series[-1, :])

    rot_vec_compensated = theta_sum + com_con
    dv_sum_compensated = dv_sum + 0.5 * np.cross(theta_sum, dv_sum) + com_scu
    return rot_vec_compensated, dv_sum_compensated
