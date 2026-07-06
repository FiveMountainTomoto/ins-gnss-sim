import numpy as np

from core.earth_model import EarthModel
from core.sins_error_transition import build_sins_error_transition
from core.basic import linalg


def test_sins_error_transition_basic_blocks():
    # 设定一个合理的状态：纬度、经度（弧度）和高度（m）以及导航速度 (ve, vn, vu)
    lat = 0.3
    lon = 0.2
    h = 100.0
    vn = np.array([120.0, 5.0, 0.0])

    earth = EarthModel((lat, lon, h), vn)

    Cnb = np.eye(3)
    fb = np.array([0.1, 0.0, -9.8])

    Ft = build_sins_error_transition(earth, Cnb, fb)

    # 形状正确
    assert Ft.shape == (15, 15)

    # 检查子块：Ft(phi, eb) = -Cnb
    assert np.allclose(Ft[0:3, 9:12], -Cnb)

    # 检查子块：Ft(dv, db) = Cnb
    assert np.allclose(Ft[3:6, 12:15], Cnb)

    # Ft(phi,phi) 应为 skew(-omega_in_n)
    expected_Maa = linalg.skew_symmetric(-earth.omega_in_n)
    assert np.allclose(Ft[0:3, 0:3], expected_Maa)

    # 底部 6 行应为零
    assert np.allclose(Ft[9:15, :], np.zeros((6, 15)))

    # 全矩阵值有限
    assert np.isfinite(Ft).all()
