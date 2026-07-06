from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WGS84:
    """
    WGS-84椭球参数
    """
    a: float = 6378137.0  # 长半轴
    f: float = 1 / 298.257223563  # 扁率
    b: float = a * (1 - f)  # 短半轴
    e2: float = f * (2 - f)  # 第一偏心率平方
    ep2: float = e2 / (1 - e2)  # 第二偏心率平方
    omega_ie: float = 7.2921151467e-5  # 地球自转角速度(rad/s)
    g0: float = 9.7803253359  # 赤道上的重力加速度(m/s^2)


def savage_coefficients(order: int) -> np.ndarray:
    """
    圆锥/划船误差补偿系数
    :param order: 多项式阶数（2-6）
    :type order: int
    :return: 多项式系数数组
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    table = {
        2: np.array([2])/3,
        3: np.array([9, 27])/20,
        4: np.array([54, 92, 214])/105,
        5: np.array([250, 525, 650, 1375])/504,
        6: np.array([2315, 4558, 7296, 7834, 15797])/4620,
    }
    if order not in table:
        raise ValueError(
            f"Savage coefficients for order {order} are not defined.")

    return table[order]


@dataclass
class AlgorithmConfig:
    """
    算法配置参数
    """
    coning_sculling_order: int = 2  # 圆锥/划船误差补偿子样数2-6
    ins_update_interval: float = 0.1  # 采样间隔
    gravity_model: str = "grs80"  # 重力模型
    earth_model: str = "wgs84"  # 参考椭球


wgs84 = WGS84()
config = AlgorithmConfig()
