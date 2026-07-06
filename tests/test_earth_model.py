import numpy as np
from core.earth_model import EarthModel
from config import wgs84


def test_earth_model_equator_zero_velocity():
    """赤道、零速度时的基本一致性测试"""
    position = (0.0, 0.0, 0.0)  # lat, lon, h (radians, radians, meters)
    velocity = np.zeros(3)

    em = EarthModel(position, velocity)

    # 总牵连角速度应为 [0, omega_ie, 0]
    expected_omega = np.array([0.0, wgs84.omega_ie, 0.0])
    assert np.allclose(em.omega_in_n, expected_omega)

    # 有效重力在零速度下等于重力模型值
    expected_gravity = np.array([0.0, 0.0, -wgs84.g0])
    assert np.allclose(em.gravity_eff_n, expected_gravity)

    # 曲率半径相关值
    assert np.isclose(em.rn_plus_h, wgs84.a)
    assert np.isclose(em.rm_plus_h, wgs84.a * (1 - wgs84.e2))
    assert np.isclose(em.cos_lat_rn_plus_h, wgs84.a)
