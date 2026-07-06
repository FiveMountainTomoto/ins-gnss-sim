"""Plotting utilities for INS trajectory and sensor data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional
from config import wgs84


def delta_position(pos: np.ndarray) -> np.ndarray:
    """
    输入 (N,3) [lat,lon,h] rad/m
    输出 (N,3) [north_m, east_m, up_m]
    """
    if pos.shape[0] == 0:
        return np.empty((0, 3))

    Re = wgs84.a
    lat0, lon0, h0 = pos[0, :]

    cos_lat = np.cos(pos[:, 0])

    # 弧度→米
    dlat = (pos[:, 0] - lat0) * Re
    dlon = (pos[:, 1] - lon0) * cos_lat * Re
    dh = pos[:, 2] - h0

    return np.column_stack([dlat, dlon, dh])


# 尝试设置中文字体，优先使用 Windows 常见字体
def _ensure_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Heiti SC",
        "Noto Sans CJK SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name] + \
                plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False
            return name
    # 若未找到，仍然确保负号可显示
    plt.rcParams['axes.unicode_minus'] = False
    return None


# 尝试在模块导入时设置中文字体
_ensure_chinese_font()


def plot_trajectory(time: np.ndarray, att: np.ndarray, vn: np.ndarray,
                    pos: np.ndarray, deg2rad: float = np.pi / 180.0):
    """Plot trajectory results in 4 subplots.

    Parameters
    ----------
    time
        Time vector (N,) in seconds.
    att
        Attitude (N, 3) in radians.
    vn
        Velocity (N, 3) in m/s.
    pos
        Position (N, 3) [lat, lon, h].
    deg2rad
        Conversion factor (rad/deg).
    """
    dpos = delta_position(pos)

    fig = plt.figure(figsize=(12, 10))

    # Attitude
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time, att / deg2rad)
    ax1.grid(True)
    ax1.set_xlabel("时间 / s")
    ax1.set_ylabel("姿态 / °")
    ax1.legend(["滚转", "俯仰", "偏航"])

    # Velocity
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time, vn)
    ax2.grid(True)
    ax2.set_xlabel("时间 / s")
    ax2.set_ylabel("速度 / m/s")
    ax2.legend(["北向 Vn", "东向 Ve", "下向 Vd"])

    # Position increments
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time, dpos)
    ax3.grid(True)
    ax3.set_xlabel("时间 / s")
    ax3.set_ylabel("位置增量 / m")
    ax3.legend(["北向 ΔN", "东向 ΔE", "高度 Δh"])

    # Geographic plot (lat vs lon)
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(pos[:, 1] / deg2rad, pos[:, 0] / deg2rad)
    ax4.plot(pos[0, 1] / deg2rad, pos[0, 0] / deg2rad,
             "ro", markersize=8, label="start")
    ax4.grid(True)
    ax4.set_xlabel("经度 / °")
    ax4.set_ylabel("纬度 / °")
    ax4.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax4.legend(["起点"])

    plt.tight_layout()
    return fig


def plot_imu_data(time: np.ndarray, wm: np.ndarray, vm: np.ndarray,
                  dt: float, deg2rad: float = np.pi / 180.0):
    """Plot IMU gyro and accelerometer data.

    Parameters
    ----------
    time
        Time vector (N,) in seconds (corresponding to increments).
    wm
        Angular increments (N-1, 3) in radians.
    vm
        Velocity increments (N-1, 3) in m/s (approximately).
    dt
        Time step in seconds.
    deg2rad
        Conversion factor (rad/deg).
    """
    fig = plt.figure(figsize=(12, 5))

    # Gyro rates
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(time[1:], wm / dt / deg2rad)
    ax1.grid(True)
    ax1.set_xlabel("时间 / s")
    ax1.set_ylabel("陀螺角速 / °/s")
    ax1.legend(["p (滚转)", "q (俯仰)", "r (偏航)"])

    # Accelerometer
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(time[1:], vm / dt)
    ax2.grid(True)
    ax2.set_xlabel("时间 / s")
    ax2.set_ylabel("加速度 / m/s²")
    ax2.legend(["ax (前向)", "ay (右向)", "az (下向)"])

    plt.tight_layout()
    return fig


__all__ = ["delta_position", "plot_trajectory",
           "plot_imu_data", "plot_compare_avp_xkpk"]


# plot_filter_states 已删除；保持模块只导出原始绘图函数


def plot_compare_avp_xkpk(avp: np.ndarray, xkpk: np.ndarray):
    """按照原始 Matlab 样式绘制真值与估计对比以及方差收敛图。

    参数:
        avp: (N,10) 每行 [phi1,phi2,phi3, vn,ve,vd, lat,lon,h, t]
        xkpk: (N,31) 每行 [x(15), diag(P)(15), t]
    返回:
        matplotlib.figure.Figure
    """
    if avp.size == 0 or xkpk.size == 0:
        return None

    Re = wgs84.a
    tt = avp[:, -1]

    # 状态与方差
    x = xkpk[:, :15]
    pk = np.sqrt(xkpk[:, 15:30])

    # 单位换算因子
    arcmin = 180.0 * 60.0 / np.pi
    dph = np.pi / 180.0 / 3600.0  # rad per deg/h -> 用于反向缩放
    ug = 1e-6 * wgs84.g0

    # 上半部：比较图（3x2）
    fig = plt.figure(figsize=(12, 12))

    ax = plt.subplot(3, 2, 1)
    ax.plot(tt, avp[:, 0:2] * arcmin)
    ax.plot(tt, x[:, 0:2] * arcmin)
    ax.set_title('姿态小角: 误差（球面分/弧分）')
    ax.legend(['真值: φ_E', '真值: φ_N', '估计: φ_E', '估计: φ_N'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 2)
    ax.plot(tt, avp[:, 2] * arcmin)
    ax.plot(tt, x[:, 2] * arcmin)
    ax.set_title('姿态小角: 垂向')
    ax.legend(['真值: φ_U', '估计: φ_U'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 3)
    ax.plot(tt, avp[:, 3:6])
    ax.plot(tt, x[:, 3:6])
    ax.set_title('速度误差 / m/s')
    ax.legend(['真值: Vn', '真值: Ve', '真值: Vd', '估计: Vn', '估计: Ve', '估计: Vd'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 4)
    dpos = delta_position(avp[:, 6:9])
    north_est = x[:, 6] * Re
    east_est = x[:, 7] * np.cos(avp[:, 6]) * Re
    up_est = x[:, 8]
    ax.plot(tt, dpos)
    ax.plot(tt, np.column_stack([north_est, east_est, up_est]))
    ax.set_title('位置误差 / m')
    ax.legend(['真值: 北向', '真值: 东向', '真值: 高度', '估计: 北向', '估计: 东向', '估计: 高度'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 5)
    ax.plot(tt, x[:, 9:12] / dph)
    ax.set_title('陀螺零偏相关项 / deg/h')
    ax.legend(['估计: εx', '估计: εy', '估计: εz'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 6)
    ax.plot(tt, x[:, 12:15] / ug)
    ax.set_title('加计偏置 / ug')
    ax.legend(['估计: bx', '估计: by', '估计: bz'])
    ax.grid(True)

    plt.tight_layout()

    # 下半部：方差图（单独 fig2，保持与原实现一致）
    fig2 = plt.figure(figsize=(12, 12))

    ax = plt.subplot(3, 2, 1)
    ax.plot(tt, pk[:, 0:2] * arcmin)
    ax.set_title('方差: 姿态 φ EN')
    ax.legend(['σ(φ_E)', 'σ(φ_N)'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 2)
    ax.plot(tt, pk[:, 2] * arcmin)
    ax.set_title('方差: 姿态 φ U')
    ax.legend(['σ(φ_U)'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 3)
    ax.plot(tt, pk[:, 3:6])
    ax.set_title('方差: 速度误差')
    ax.legend(['σ(Vn)', 'σ(Ve)', 'σ(Vd)'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 4)
    north_p = pk[:, 6] * Re
    east_p = pk[:, 7] * np.cos(avp[:, 6]) * Re
    up_p = pk[:, 8]
    ax.plot(tt, np.column_stack([north_p, east_p, up_p]))
    ax.set_title('方差: 位置误差 / m')
    ax.legend(['σ(北向)', 'σ(东向)', 'σ(高度)'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 5)
    ax.plot(tt, pk[:, 9:12] / dph)
    ax.set_title('方差: 陀螺相关 / deg/h')
    ax.legend(['σ(εx)', 'σ(εy)', 'σ(εz)'])
    ax.grid(True)

    ax = plt.subplot(3, 2, 6)
    ax.plot(tt, pk[:, 12:15] / ug)
    ax.set_title('方差: 加计偏置 / ug')
    ax.legend(['σ(bx)', 'σ(by)', 'σ(bz)'])
    ax.grid(True)

    plt.tight_layout()
    return fig
