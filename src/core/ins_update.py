import numpy as np
from typing import Tuple
from .basic.quaternion import Quaternion
from .compensate_coning_sculling import compensate_coning_sculling
from .earth_model import EarthModel
from .basic.linalg import rotvec_to_dir_cos_mat


def ins_update(quat_n_b: Quaternion, v_n: np.ndarray, position: Tuple[float, float, float], delta_theta_series: np.ndarray, delta_v_series: np.ndarray, dt: float):
    """
    捷联惯导更新算法

    :param quat_n_b: 导航坐标系到机体坐标系的四元数(4,)
    :type quat_n_b: Quaternion
    :param v_n: 导航坐标系下的速度(3,)
    :type v_n: np.ndarray
    :param position: 位置[纬度, 经度, 高度]
    :type position: Tuple[float, float, float]
    :param delta_theta_series: 陀螺仪增量角序列 (N, 3)
    :type delta_theta_series: np.ndarray
    :param delta_v_series: 加速度计增量速度序列 (N, 3)
    :type delta_v_series: np.ndarray
    :param dt: 时间间隔，秒
    :type dt: float
    :return: 更新后的四元数、速度和位置
    :rtype: Tuple[Quaternion, np.ndarray, Tuple[float, float, float]]
    """
    n = delta_theta_series.shape[0]
    total_t = n * dt
    # 圆锥/划船误差补偿
    rotvec, dv = compensate_coning_sculling(
        delta_theta_series, delta_v_series)
    # 地球参数计算
    earth = EarthModel(position, v_n)
    # 速度更新
    dv_n = quat_n_b.rotate(dv)
    mid_theta = -earth.omega_in_n * total_t / 2
    dv_mid = rotvec_to_dir_cos_mat(mid_theta) @ dv_n
    v_new = v_n + dv_mid + earth.gravity_eff_n * total_t
    # 位置更新
    v_mid = (v_n + v_new) / 2
    dlat = v_mid[1] * total_t / earth.rm_plus_h
    dlon = v_mid[0] * total_t / earth.cos_lat_rn_plus_h
    dh = v_mid[2] * total_t
    position_new = (position[0] + dlat,
                    position[1] + dlon,
                    position[2] + dh)
    # 姿态更新
    quat_rot = Quaternion.from_rotvec(rotvec)
    quat_n_new_b = quat_n_b * quat_rot
    quat_n_new_n = Quaternion.from_rotvec(-earth.omega_in_n * total_t)
    quat_new_n_new_b = quat_n_new_n * quat_n_new_b
    return quat_new_n_new_b, v_new, position_new
