import numpy as np
from core.basic import Quaternion


def quarternion_add_misalignment_angle(quaternion: Quaternion, phi: np.ndarray) -> Quaternion:
    """
    四元数加上失准角
    :param quaternion: 四元数（q0, q1, q2, q3）
    :type quaternion: ndarray[4,]
    :param phi: 微小旋转向量
    :type phi: ndarray[3,]
    :return: 更新后的四元数
    :rtype: ndarray[4,]
    """
    if not isinstance(quaternion, Quaternion):
        raise ValueError("Input quaternion must be a Quaternion instance.")
    if phi.shape != (3,):
        raise ValueError("Input phi must be a 3-dimensional vector.")

    phi_quat = Quaternion.from_rotvec(-phi)
    updated_quat = phi_quat * quaternion
    return updated_quat


def quarternion_minus_misalignment_angle(quaternion: Quaternion, phi: np.ndarray) -> Quaternion:
    """
    四元数减去失准角
    :param quaternion: 四元数（q0, q1, q2, q3）
    :type quaternion: ndarray[4,]
    :param phi: 微小旋转向量
    :type phi: ndarray[3,]
    :return: 更新后的四元数
    :rtype: ndarray[4,]
    """
    if not isinstance(quaternion, Quaternion):
        raise ValueError("Input quaternion must be a Quaternion instance.")
    if phi.shape != (3,):
        raise ValueError("Input phi must be a 3-dimensional vector.")

    phi_quat = Quaternion.from_rotvec(phi)
    updated_quat = phi_quat * quaternion
    return updated_quat


def get_misalignment_angle_from_quaternion_error(quat_n: Quaternion, quat_p: Quaternion) -> np.ndarray:
    """
    由导航坐标系四元数和计算导航坐标系四元数计算失准角
    :param quat_n: 导航坐标系四元数（q0, q1, q2, q3）
    :type quat_n: ndarray[4,]
    :param quat_p: 计算导航坐标系四元数（q0, q1, q2, q3）
    :type quat_p: ndarray[4,]
    :return: 失准角等效旋转矢量
    :rtype: ndarray[3,]
    """
    if not isinstance(quat_n, Quaternion):
        raise ValueError("Input quat_n must be a Quaternion instance.")
    if not isinstance(quat_p, Quaternion):
        raise ValueError("Input quat_p must be a Quaternion instance.")

    delta_quat = quat_n * quat_p.conjugate()
    phi = delta_quat.to_rotvec()
    return phi
