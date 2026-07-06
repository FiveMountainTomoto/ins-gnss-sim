import numpy as np


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """
    生成三维向量的反对称矩阵
    :param vector: 输入三维向量
    :type vector: ndarray[3,]
    :return: 反对称矩阵
    :rtype: ndarray[3, 3]
    """
    if vector.shape != (3,):
        raise ValueError("Input vector must be a 3-dimensional vector.")

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def euler_to_dir_cos_mat(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """
    将欧拉角转换为方向余弦矩阵
    :param pitch: 俯仰角（弧度）
    :param roll: 横滚角（弧度）
    :param yaw: 航向角（弧度）
    :return: 方向余弦矩阵
    :rtype: ndarray[3, 3]
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    dcm = np.array([
        [cy*cr-sy*sp*sr, -sy*cp, cy*sr+sy*sp*cr],
        [sy*cr+cy*sp*sr, cy*cp, sy*sr-cy*sp*cr],
        [-cp*sr, sp, cp*cr]
    ])
    return dcm


def dir_cos_mat_to_euler(dcm: np.ndarray) -> tuple:
    """
    将方向余弦矩阵转换为欧拉角
    :param dcm: 方向余弦矩阵
    :type dcm: ndarray[3, 3]
    :return: 欧拉角（横滚角，俯仰角，航向角）
    :rtype: tuple[float, float, float]
    """
    if dcm.shape != (3, 3):
        raise ValueError("Input DCM must be a 3x3 matrix.")

    pitch = np.arcsin(dcm[2, 1])
    if abs(dcm[2, 1]) <= 0.999999:
        roll = -np.arctan2(dcm[2, 0], dcm[2, 2])
        yaw = -np.arctan2(dcm[0, 1], dcm[1, 1])
    else:
        roll = np.arctan2(dcm[0, 2], dcm[0, 0])
        yaw = 0.0
    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将欧拉角转换为四元数
    :param roll: 横滚角（弧度）
    :param pitch: 俯仰角（弧度）
    :param yaw: 航向角（弧度）
    :return: 四元数（q0, q1, q2, q3）
    :rtype: ndarray[4,]
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    q0 = cy*cp*cr-sy*sp*sr
    q1 = cy*sp*cr-sy*cp*sr
    q2 = sy*sp*cr+cy*cp*sr
    q3 = sy*cp*cr+cy*sp*sr

    return np.array([q0, q1, q2, q3])


def quaternion_to_dir_cos_mat(quaternion: np.ndarray) -> np.ndarray:
    """
    将四元数转换为方向余弦矩阵
    :param quaternion: 四元数（q0, q1, q2, q3）
    :type quaternion: ndarray[4,]
    :return: 方向余弦矩阵
    :rtype: ndarray[3, 3]
    """
    if quaternion.shape != (4,):
        raise ValueError("Input quaternion must be a 4-dimensional vector.")

    q0, q1, q2, q3 = quaternion
    dcm = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])
    return dcm


def quaternion_to_euler(quaternion: np.ndarray) -> tuple:
    """
    将四元数转换为欧拉角
    :param quaternion: 四元数（q0, q1, q2, q3）
    :type quaternion: ndarray[4,]
    :return: 欧拉角（横滚角，俯仰角，航向角）
    :rtype: tuple[float, float, float]
    """
    dcm = quaternion_to_dir_cos_mat(quaternion)
    roll, pitch, yaw = dir_cos_mat_to_euler(dcm)
    return roll, pitch, yaw


def dir_cos_mat_to_quaternion(dcm: np.ndarray) -> np.ndarray:
    """
    将方向余弦矩阵转换为四元数
    :param dcm: 方向余弦矩阵
    :type dcm: ndarray[3, 3]
    :return: 四元数（q0, q1, q2, q3）
    :rtype: ndarray[4,]
    """
    if dcm.shape != (3, 3):
        raise ValueError("Input DCM must be a 3x3 matrix.")

    if dcm[0, 0] >= dcm[1, 1]+dcm[2, 2]:
        q1 = 0.5*np.sqrt(1+dcm[0, 0]-dcm[1, 1]-dcm[2, 2])
        q0 = (dcm[2, 1]-dcm[1, 2])/(4*q1)
        q2 = (dcm[0, 1]+dcm[1, 0])/(4*q1)
        q3 = (dcm[0, 2]+dcm[2, 0])/(4*q1)
    elif dcm[1, 1] >= dcm[0, 0]+dcm[2, 2]:
        q2 = 0.5*np.sqrt(1-dcm[0, 0]+dcm[1, 1]-dcm[2, 2])
        q0 = (dcm[0, 2]-dcm[2, 0])/(4*q2)
        q1 = (dcm[0, 1]+dcm[1, 0])/(4*q2)
        q3 = (dcm[1, 2]+dcm[2, 1])/(4*q2)
    elif dcm[2, 2] >= dcm[0, 0]+dcm[1, 1]:
        q3 = 0.5*np.sqrt(1-dcm[0, 0]-dcm[1, 1]+dcm[2, 2])
        q0 = (dcm[1, 0]-dcm[0, 1])/(4*q3)
        q1 = (dcm[0, 2]+dcm[2, 0])/(4*q3)
        q2 = (dcm[1, 2]+dcm[2, 1])/(4*q3)
    else:
        q0 = 0.5*np.sqrt(1+dcm[0, 0]+dcm[1, 1]+dcm[2, 2])
        q1 = (dcm[2, 1]-dcm[1, 2])/(4*q0)
        q2 = (dcm[0, 2]-dcm[2, 0])/(4*q0)
        q3 = (dcm[1, 0]-dcm[0, 1])/(4*q0)
    return np.array([q0, q1, q2, q3])


def rotvec_to_dir_cos_mat(rotvec: np.ndarray) -> np.ndarray:
    """
    将旋转向量转换为方向余弦矩阵
    :param rotvec: 旋转向量
    :type rotvec: ndarray[3,]
    :return: 方向余弦矩阵
    :rtype: ndarray[3, 3]
    """
    if rotvec.shape != (3,):
        raise ValueError("Input rotvec must be a 3-dimensional vector.")
    theta2 = rotvec@rotvec
    if theta2 < 1e-8:
        a = 1-theta2*(1/6-theta2/120)
        b = 0.5-theta2*(1/24-theta2/720)
    else:
        theta = np.sqrt(theta2)
        a = np.sin(theta)/theta
        b = (1-np.cos(theta))/theta2
    skew_mat = skew_symmetric(rotvec)
    dcm = np.eye(3)+a*skew_mat+b*(skew_mat@skew_mat)
    return dcm


def rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """
    将旋转向量转换为四元数
    :param rotvec: 旋转向量
    :type rotvec: ndarray[3,]
    :return: 四元数（q0, q1, q2, q3）
    :rtype: ndarray[4,]
    """
    if rotvec.shape != (3,):
        raise ValueError("Input rotvec must be a 3-dimensional vector.")
    theta2 = rotvec@rotvec
    if theta2 < 1e-8:
        q0 = 1-theta2*(1/8-theta2/384)
        coeff = 0.5-theta2*(1/48-theta2/3840)
    else:
        theta = np.sqrt(theta2)
        q0 = np.cos(theta/2)
        coeff = np.sin(theta/2)/theta
    q1, q2, q3 = coeff*rotvec
    return np.array([q0, q1, q2, q3])


def quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """
    将四元数转换为旋转向量
    :param quaternion: 四元数（q0, q1, q2, q3）
    :type quaternion: ndarray[4,]
    :return: 旋转向量
    :rtype: ndarray[3,]
    """
    if quaternion.shape != (4,):
        raise ValueError("Input quaternion must be a 4-dimensional vector.")
    q0, q1, q2, q3 = quaternion
    if q0 < 0:
        q0, q1, q2, q3 = -q0, -q1, -q2, -q3
    half_theta = np.arccos(q0)
    if half_theta > 1e-20:
        coeff = 2*half_theta/np.sin(half_theta)
    else:
        coeff = 2
    rotvec = coeff * np.array([q1, q2, q3])
    return rotvec


def quarternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """
    计算四元数的共轭四元数
    :param quaternion: 四元数（q0, q1, q2, q3）
    :type quaternion: ndarray[4,]
    :return: 共轭四元数
    :rtype: ndarray[4,]
    """
    if quaternion.shape != (4,):
        raise ValueError("Input quaternion must be a 4-dimensional vector.")
    q0, q1, q2, q3 = quaternion
    return np.array([q0, -q1, -q2, -q3])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    计算两个四元数的乘积
    :param q1: 第一个四元数（q0, q1, q2, q3）
    :type q1: ndarray[4,]
    :param q2: 第二个四元数（q0, q1, q2, q3）
    :type q2: ndarray[4,]
    :return: 四元数乘积
    :rtype: ndarray[4,]
    """
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError("Input quaternions must be 4-dimensional vectors.")
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])


def quaternion_mul_vec(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    四元数乘三维向量
    :param quaternion: 四元数（q0, q1, q2, q3）
    :type quaternion: ndarray[4,]
    :param vector: 三维向量
    :type vector: ndarray[3,]
    :return: 旋转后的三维向量
    :rtype: ndarray[3,]
    """
    if quaternion.shape != (4,):
        raise ValueError("Input quaternion must be a 4-dimensional vector.")
    if vector.shape != (3,):
        raise ValueError("Input vector must be a 3-dimensional vector.")

    q_vec = np.array([0, vector[0], vector[1], vector[2]])
    q_conj = quarternion_conjugate(quaternion)
    q_rotated = quaternion_multiply(
        quaternion_multiply(quaternion, q_vec), q_conj)
    return q_rotated[1:]
