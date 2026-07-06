import numpy as np
from config import wgs84
from typing import Tuple


class EarthModel:
    __slots__ = (
        "omega_in_n",  # 导航坐标系下的总牵连角速度(3,)
        "gravity_eff_n",  # 导航坐标系下的有效重力(3,)
        "rm_plus_h",  # 子午圈曲率半径加高
        "rn_plus_h",  # 卯酉圈曲率半径加高
        "cos_lat_rn_plus_h",  # 纬度余弦值 乘 卯酉圈曲率半径加高
        "gravity_n",  # 导航坐标系下的重力(3,)
        "sl", "cl", "tl", "sl2",  # 纬度相关的三角函数值
        "vn",  # 导航坐标系下的速度(3,)
        "omega_ie_n",  # 导航坐标系下的地球自转角速度(3,)
        "omega_en_n",  # 导航坐标系下的地理牵连角速度(3,)
        "inv_rm_plus_h", "inv_rn_plus_h", "inv_cos_lat_rn_plus_h",  # 曲率相关的倒数
    )

    def __init__(self, position: Tuple[float, float, float], velocity_n: np.ndarray) -> None:
        """
        地球导航参数计算类构造函数

        :param self: 
        :param position: 位置[纬度, 经度, 高度]，弧度
        :type position: Tuple[float, float, float]
        :param velocity_n: 导航坐标系下的速度(3,)
        :type velocity_n: np.ndarray
        """
        lat, lon, h = position
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        tan_lat = np.tan(lat)

        # 计算曲率半径
        e2 = wgs84.e2
        a = wgs84.a
        rn = a / np.sqrt(1 - e2 * sin_lat**2)
        rm = a * (1 - e2) / (1 - e2 * sin_lat**2)**1.5

        self.rm_plus_h = rm + h
        self.rn_plus_h = rn + h
        self.cos_lat_rn_plus_h = cos_lat * self.rn_plus_h

        # 常用三角量
        self.sl = sin_lat
        self.cl = cos_lat
        self.tl = tan_lat
        self.sl2 = sin_lat**2

        # 速度向量
        self.vn = velocity_n

        # 计算总牵连角速度
        omega_ie = wgs84.omega_ie
        # 地球自转在导航系分量
        self.omega_ie_n = omega_ie * np.array([0.0, cos_lat, sin_lat])
        # 地理牵连角速度（由速度与曲率产生）
        self.omega_en_n = np.array([
            -velocity_n[1] / self.rm_plus_h,
            velocity_n[0] / self.rn_plus_h,
            velocity_n[0] / self.rn_plus_h * tan_lat
        ])
        # 总牵连角速度 = 地球自转 + 地理牵连
        self.omega_in_n = self.omega_ie_n + self.omega_en_n

        # 便捷的倒数量（避免调用处反复计算）
        self.inv_rm_plus_h = 1.0 / self.rm_plus_h
        self.inv_rn_plus_h = 1.0 / self.rn_plus_h
        self.inv_cos_lat_rn_plus_h = 1.0 / self.cos_lat_rn_plus_h

        # 计算有效重力
        g0 = wgs84.g0
        # 局地重力模型（GRS80）
        g_lat_h = g0*(1+5.27094e-3*sin_lat**2+2.32718e-5*sin_lat**4)-3.086e-6*h
        self.gravity_n = np.array([0.0, 0.0, -g_lat_h])
        # 有效重力 = 真重力 - 旋转项 (omega_ie_n + omega_in_n) x v
        self.gravity_eff_n = self.gravity_n - \
            np.cross(self.omega_ie_n+self.omega_in_n, velocity_n)
